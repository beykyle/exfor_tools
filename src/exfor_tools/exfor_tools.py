import numpy as np
import periodictable
from functools import reduce

import x4i3
from x4i3 import exfor_manager
from x4i3.exfor_reactions import X4Reaction
from x4i3.exfor_column_parsing import (
    X4ColumnParser,
    X4IndependentColumnPair,
    angDistUnits,
    angleParserList,
    baseDataKeys,
    condenseColumn,
    dataTotalErrorKeys,
    energyUnits,
    errorSuffix,
    frameSuffix,
    incidentEnergyParserList,
    noUnits,
    percentUnits,
    resolutionFWSuffix,
    resolutionHWSuffix,
    variableSuffix,
    X4MissingErrorColumnPair,
)

__EXFOR_DB__ = None


def init_exfor_db():
    global __EXFOR_DB__
    if __EXFOR_DB__ is None:
        __EXFOR_DB__ = exfor_manager.X4DBManagerDefault()


def get_db():
    global __EXFOR_DB__
    if __EXFOR_DB__ is None:
        init_exfor_db()
    return __EXFOR_DB__


# these are the supported quantities at the moment
# XS = cross section, A = angle, Ruth = Rutherford cross section, Ay = analyzing power
quantity_matches = {
    "dXS/dA": [["DA"]],
    "dXS/dRuth": [["DA", "RTH"], ["DA", "RTH/REL"]],
    "Ay": [["POL/DA", "ANA"]],
}

quantity_symbols = {
    ("DA",): r"$\frac{d\sigma}{d\Omega}$",
    ("DA", "RTH"): r"$\sigma / \sigma_{R}$",
    ("DA", "RTH/REL"): r"$\sigma / \sigma_{R}$",
    ("POL/DA", "ANA"): r"$A_y$",
}

label_matches = dict(
    zip(
        ["EN", "ANG-ERR", "DATA-ERR", "ANG-CM", "DATA"],
        ["Energy", "d(Angle)", "d(Data)", "Angle", "Data"],
    )
)


energyExParserList = [
    X4MissingErrorColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        None,
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in errorSuffix]
            + ["E-EXC" + s for s in errorSuffix],
            match_units=energyUnits + percentUnits,
        ),
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in resolutionFWSuffix]
            + ["E-EXC" + s for s in resolutionFWSuffix],
            match_units=energyUnits + percentUnits,
            scale_factor=0.5,
        ),
    ),
    X4IndependentColumnPair(
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in variableSuffix]
            + ["E-EXC" + s for s in variableSuffix],
            match_units=energyUnits,
        ),
        X4ColumnParser(
            match_labels=["E-LVL" + s for s in resolutionHWSuffix]
            + ["E-EXC" + s for s in resolutionHWSuffix],
            match_units=energyUnits + percentUnits,
        ),
    ),
]


def sanitize_column(col):
    for i in range(len(col)):
        if col[i] is None:
            col[i] = 0
    return col


def get_exfor_differential_data(
    target,
    projectile,
    quantity,
    product,
    residual,
    energy_range=None,
):
    r"""query EXFOR for all entries satisfying search criteria, and return them
    as a dictionary of entry number to ExforEntryAngularDistribution"""
    A, Z = target
    target_symbol = f"{str(periodictable.elements[Z])}-{A}"

    A, Z = projectile
    if (A, Z) == (1, 0):
        projectile_symbol = "N"
    elif (A, Z) == (1, 1):
        projectile_symbol = "P"
    elif (A, Z) == (2, 1):
        projectile_symbol = "D"
    elif (A, Z) == (3, 1):
        projectile_symbol = "T"
    elif (A, Z) == (4, 2):
        projectile_symbol = "A"
    else:
        projectile_symbol = f"{str(periodictable.elements[Z])}-{A}"
    if product is None:
        product = "*"

    reaction = f"{projectile_symbol},{product}"
    exfor_quantity = quantity_matches[quantity][0][0]
    entries = __EXFOR_DB__.query(
        reaction=reaction, quantity=exfor_quantity, target=target_symbol
    ).keys()

    data_sets = {}
    for entry in entries:
        try:
            data_set = ExforEntryAngularDistribution(
                entry=entry,
                target=target,
                projectile=projectile,
                quantity=quantity,
                residual=residual,
                product=product,
                energy_range=energy_range,
            )
        except ValueError as e:
            print(f"There was an error reading entry {entry}, it will be skipped:")
            print(e)
        if len(data_set.measurements) > 0 and entry not in data_sets:
            data_sets[entry] = data_set

    return data_sets


def sort_measurement_list(measurements, min_num_pts=5):
    # TODO don't condense, instead store a map from unique measurement conditions to list of AngularDistributions
    energies = np.array([m.Elab for m in measurements])
    energies_sorted = energies[np.argsort(energies)]
    measurements_sorted = [measurements[i] for i in np.argsort(energies)]
    vals, idx, cnt = np.unique(energies_sorted, return_counts=True, return_index=True)
    measurements_condensed = []
    for i, c in zip(idx, cnt):
        m = measurements_sorted[i]

        # concatenate data for all sets at the same energy
        data = np.hstack(
            [m.data] + [measurements_sorted[i + j].data for j in range(1, c)]
        )

        # re-sort data by angle
        data = data[:, data[0, :].argsort()]

        measurements_condensed.append(
            AngularDistribution(
                m.Elab, m.dElab, m.energy_units, m.units, m.labels, data
            )
        )

    # sanitize
    for m in measurements_condensed:
        m.data = m.data[:, m.data[0, :].argsort()]
        m.data = m.data[:, np.logical_and(m.data[0, :] >= 0, m.data[0, :] <= 180)]

    return measurements_condensed


def sort_measurements_by_energy(all_entries, min_num_pts=5):
    r"""
    Given a dictionary form EXFOR entry number to ExforEntryAngularDistribution, grabs all
    the ExforEntryAngularDistributionSet's and sorts them by energy, concatenating ones
    that are at the same energy
    """
    measurements = []
    for entry, data in all_entries.items():
        for measurement in data.measurements:
            if measurement.data.shape[1] > min_num_pts:
                measurements.append(measurement)
    return sort_measurement_list(measurements, min_num_pts=min_num_pts)


def parse_differential_data(
    subentry, data_error_columns=["DATA-ERR"], err_treatment="independent"
):
    r"""
    Extract differential cross section (potentially as ratio to Rutherford)
    """
    data_parser = X4ColumnParser(
        match_labels=reduce(
            lambda x, y: x + y,
            [[b + s for s in variableSuffix + frameSuffix] for b in baseDataKeys],
        ),
        match_units=angDistUnits + noUnits,
    )
    match_idxs = data_parser.allMatches(subentry)
    if len(match_idxs) != 1:
        raise ValueError(f"Expected only one DATA column, found {len(match_idxs)}")
    iy = match_idxs[0]
    data_column = data_parser.getColumn(iy, subentry)
    xs_units = data_column[1]
    xs = np.array(data_column[2:], dtype=np.float64)

    # parse errors
    err_col_match = []
    for label in data_error_columns:

        # parse error column
        err_parser = X4ColumnParser(
            match_labels=reduce(
                lambda x, y: x + y,
                [label],
            ),
            match_units=angDistUnits + percentUnits + noUnits,
        )
        icol = err_parser.firstMatch(subentry)
        if icol >= 0:
            err = err_parser.getColumn(icol, subentry)
            err_units = err[1]
            err_data = np.array(sanitize_column(err[2:]), dtype=np.float64)
            # convert to same units as data
            if "PER-CENT" in err_units:
                err_data *= xs / 100
            elif err_units != xs_units:
                raise ValueError(
                    f"Attempted to extract error column {err[0]} with incompatible units"
                    f"{err_units} for data column {data_column[0]} with units {xs_units}"
                )

            err_col_match.append(err_data)

    if not err_col_match:
        xs_err = np.zeros_like(xs)
    elif err_treatment == "independent":
        # sum errors in quadrature
        xs_err = np.sqrt(np.sum(np.array(err_col_match) ** 2, axis=0))
    elif err_treatment == "cumulative":
        # add errors
        xs_err = np.sum(np.array(err_col_match), axis=0)
    elif err_treatment == "difference":
        # subtract second error column from first
        if len(err_col_match) > 2:
            raise ValueError(
                f"Cannot only take difference of 2 error columns, but {len(err_col_match)} were found!"
            )
        xs_err = err_col_match[0] - err_col_match[1]

    return xs, xs_err


# TODO handle Q-value and level number
def parse_ex_energy(subentry):
    E_inc_cm = np.array(
        reduce(condenseColumn, [c.getValue(subentry) for c in energyExParserList])[2:],
        dtype=np.float64,
    )
    E_inc_cm_err = np.array(
        reduce(condenseColumn, [c.getError(subentry) for c in energyExParserList])[2:],
        dtype=np.float64,
    )
    return E_inc_cm, E_inc_cm_err


def parse_angle(subentry):
    angle = reduce(condenseColumn, [c.getValue(subentry) for c in angleParserList])
    if angle[1] != "degrees":
        raise ValueError(f"Cannot parse angle in units of {angle[1]}")
    angle = np.array(angle[2:])

    angle_err = reduce(condenseColumn, [c.getError(subentry) for c in angleParserList])
    if angle_err[1] != "degrees":
        raise ValueError(f"Cannot parse angle error in units of {angle_err[1]}")
    angle_err = np.array(angle_err[2:])
    return angle, angle_err


def parse_inc_energy(subentry):
    if subentry.referenceFrame != "Center of mass":
        raise NotImplementedError("Conversions from lab frame not implemented")

    E_inc_cm = np.array(
        reduce(
            condenseColumn, [c.getValue(subentry) for c in incidentEnergyParserList]
        )[2:],
        dtype=np.float64,
    )
    E_inc_cm_err = np.array(
        reduce(
            condenseColumn, [c.getError(subentry) for c in incidentEnergyParserList]
        )[2:],
        dtype=np.float64,
    )
    return E_inc_cm, E_inc_cm_err


def parse_angular_distribution(
    data_set,
    data_error_columns=None,
    err_treatment="independent",
):
    r"""
    Extracts angular differential cross sections, returning incident and product excitation
    energy in MeV, angles and error in angle in degrees, and differential cross section and its
    error in mb/Sr.
    """

    if data_error_columns is None:
        data_error_columns = [b + "-ERR" for b in baseDataKeys] + dataTotalErrorKeys

    # parse angle
    angle, angle_err = parse_angle(data_set)

    # parse energy if it's present
    E_inc_cm, E_inc_cm_err = parse_inc_energy(data_set)

    # parse excitation energy if it's present
    E_ex, E_ex_err = parse_ex_energy(data_set)

    # parse diff xs
    xs, xs_err = parse_differential_data(
        data_set, data_error_columns=data_error_columns, err_treatment=err_treatment
    )

    return E_inc_cm, E_inc_cm_err, E_ex, E_ex_err, angle, angle_err, xs, xs_err


def get_measurements_from_subentry(
    subentry,
    data_set,
    Einc_range=(0, np.inf),
    Ex_range=(0, np.inf),
    elastic_only=False,
    suppress_numbered_errs=True,
):
    r"""unrolls subentry into individual arrays for each energy"""

    err_labels = [label for label in data_set.labels if "ERR" in label]
    if "ANG-ERR" in err_labels:
        err_labels.remove("ANG-ERR")

    print(f"Error labels for {subentry}: ")
    print(err_labels)

    if suppress_numbered_errs:
        err_labels = [
            l for l in err_labels if not (l[-1].isdigit() and l[:-1] == "ERR-")
        ]

    err_labels_set = set(err_labels)
    standard_labels = set(["DATA-ERR", "ERR-T"])
    asymmetric_labels = set(["-DATA-ERR", "+DATA-ERR"])

    if err_labels == ["DATA-ERR"]:
        err_treatment = "independent"
    elif err_labels == ["ERR-T"]:
        err_treatment = "independent"
    elif err_labels_set.union(standard_labels) == err_labels_set.intersection(
        standard_labels
    ):
        err_treatment = "cumulative"
    elif err_labels_set.union(asymmetric_labels) == err_labels_set.intersection(
        asymmetric_labels
    ):
        err_treatment = "cumulative"
    else:
        raise NotImplementedError(
            "Ambiguous set of error labels:\n" + [f"{l}\n" for l in err_labels].join()
        )

    E_inc_cm, E_inc_cm_err, E_ex, E_ex_err, angle, angle_err, xs, xs_err = (
        parse_angular_distribution(
            data_set, data_error_columns=err_labels, err_treatment=err_treatment
        )
    )

    N = data_set.numrows()
    data = np.zeros((6, N))

    data[:, :] = [
        E_inc_cm,
        np.nan_to_num(E_inc_cm_err),
        np.nan_to_num(E_ex),
        np.nan_to_num(E_ex_err),
        angle,
        np.nan_to_num(angle_err),
        xs,
        np.nan_to_num(xs_err),
    ]

    Einc_mask = np.logical_and(data[0, :] >= Einc_range[0], data[0, :] < Einc_range[1])
    data = data[:, Einc_mask]

    Ex_mask = np.logical_and(data[2, :] >= Ex_range[0], data[2, :] < Ex_range[1])
    data = data[:, Ex_mask]

    # AngularDistribution objects sorted by incident energy, then excitation energy
    # or just incident enrgy if elastic_only is True
    measurements = []

    # find set of unique incident energies
    unique_Einc = np.unique(data[0, :])

    # sort and fragment data by unique incident energy
    for Einc in np.sort(unique_Einc):
        mask = np.isclose(data[0, :], Einc)
        Einc_err = data[1, mask][0]

        if elastic_only:
            measurements.append(
                (Einc, Einc_err),
                AngularDistribution(subentry, data[4:, mask], Einc, Einc_err, 0, 0),
            )
        else:
            measurements.append((Einc, Einc_err), [])
            subset = data[2:, mask]

            # find set of unique residual excitation energies
            unique_Ex = np.unique(subset[0, :])

            # sort and fragment data by unique excitation energy
            for Ex in np.sort(unique_Ex):
                mask = np.isclose(subset[0, :], Ex)
                Ex_err = subset[1, mask][0]
                measurement = AngularDistribution(
                    subentry, subset[2:, mask], Einc, Einc_err, Ex, Ex_err
                )
                measurements[-1][1].append(((Ex, Ex_err), measurement))

    return measurements


class AngularDistribution:
    def __init__(
        self,
        subentry: str,
        data: np.array,
        Einc: float,
        Einc_err: float,
        Ex: float,
        Ex_err: float,
    ):
        self.subentry = subentry
        self.data = data
        self.Einc = Einc
        self.Einc_err = Einc_err
        self.Ex = Ex
        self.Ex_err = Ex_err


def get_symbol(A, Z, Ex=None):
    if (A, Z) == (1, 0):
        return "n"
    elif (A, Z) == (1, 1):
        return "p"
    elif (A, Z) == (2, 1):
        return "d"
    elif (A, Z) == (3, 1):
        return "t"
    elif (A, Z) == (4, 2):
        return r"$\alpha$"
    else:
        ex = f"({float(Ex):1.3f})"
        return f"$^{{{A}}}${str(periodictable.elements[Z])}{ex}"


class ExforEntryAngularDistribution:
    r"""2-body reaction"""

    def __init__(
        self,
        entry: str,
        target: tuple,
        projectile: tuple,
        quantity: str,
        residual: tuple = None,
        product: tuple = None,
        special_rxn_type="",
        Einc_range: tuple = None,
        Ex_range: tuple = None,
        elastic_only=False,
        suppress_numbered_errs=True,
    ):
        self.entry = entry
        entry_datasets = __EXFOR_DB__.retrieve(ENTRY=entry)[entry].getDataSets()

        self.target = target
        self.projectile = projectile

        if product is None:
            self.product = projectile
        if residual is None:
            self.residual = target

        if len(self.residual) == 3:
            self.Ex_prime = self.residual[2]
            ex_fudge = 0.01
            Ex_range = (self.Ex_prime - ex_fudge, self.Ex_prime + ex_fudge)

        Apre = self.target[0] + self.projectile[0]
        Apost = self.residual[0] + self.product[0]
        Zpre = self.target[1] + self.projectile[1]
        Zpost = self.residual[1] + self.product[1]

        if Apre != Apost and Zpre != Zpost:
            raise ValueError("Isospin not conserved in this reaction")

        self.symbol_target = get_symbol(*self.target)
        self.symbol_residual = get_symbol(*self.residual)
        self.symbol_projectile = get_symbol(*self.projectile)
        self.symbol_product = get_symbol(*self.product)

        if self.residual == self.target:
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},{self.symbol_product})_{{{special_rxn_type}}}$"
        else:
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},{self.symbol_product})_{{{special_rxn_type}}}${self.symbol_residual}"

        self.quantity = quantity
        self.exfor_quantities = quantity_matches[quantity]
        self.data_symbol = quantity_symbols[tuple(self.exfor_quantities[0])]

        if Einc_range is None:
            Einc_range = (0, np.inf)
        self.Einc_range = Einc_range
        if Ex_range is None:
            Ex_range = (0, np.inf)
        self.Ex_range = Ex_range

        self.subentries = [key[1] for key in entry_datasets.keys()]
        self.measurements = []

        for key, data_set in entry_datasets.items():

            if isinstance(data_set.reaction[0], X4Reaction):
                # TODO need to do like EL for product for elastic or something stupid like that?
                isotope = (
                    data_set.reaction[0].targ.getA(),
                    data_set.reaction[0].targ.getZ(),
                )
                projectile = (
                    data_set.reaction[0].proj.getA(),
                    data_set.reaction[0].proj.getZ(),
                )
                product = (
                    data_set.reaction[0].products[0].getA(),
                    data_set.reaction[0].products[0].getZ(),
                )
                residual = (
                    data_set.reaction[0].residual.getA(),
                    data_set.reaction[0].residual.getZ(),
                )
                quantity = data_set.reaction[0].quantity
                if quantity[-1] == "EXP":
                    quantity = quantity[:-1]
                if (
                    isotope == self.isotope
                    and projectile == self.projectile
                    and product == self.product
                    and residual == self.residual
                ):
                    # matched reaction
                    if quantity in self.exfor_quantities:
                        # matched reaction and quantity
                        # should be the same for every subentry
                        self.meta = {
                            "author": data_set.author,
                            "title": data_set.title,
                            "year": data_set.year,
                            "institute": data_set.institute,
                        }
                        self.measurements = get_measurements_from_subentry(
                            key[1],
                            data_set,
                            self.Einc_range,
                            self.Ex_range,
                            elastic_only,
                            suppress_numbered_errs,
                        )


def plot_angular_distributions(
    ax,
    measurements,
    offsets=None,
    data_symbol="",
    rxn_label="",
    label_xloc_deg=None,
    label_offset_factor=2,
    log=True,
    add_baseline=False,
    xlim=[0, 180],
    label_energy_err=True,
    label_offset=True,
    fontsize=10,
):
    r"""
    Given measurements, a list where each entry is a tuple of ((E, E_err), AngularDistribution)
    , plots them all on the same ax
    """
    # if offsets is not a sequence, figure it out
    if isinstance(offsets, float) or isinstance(offsets, int) or offsets is None:
        if offsets is None:
            constant_factor = 1 if log else 0
        else:
            constant_factor = offsets
        if log:
            offsets = constant_factor ** np.arange(0, len(measurements))
        else:
            offsets = constant_factor * np.arange(0, len(measurements))

    units_x = "deg"
    units_y = "mb/Sr"

    # plot each measurement and add a label
    for offset, ((E, E_err), m) in zip(offsets, measurements):

        x = np.copy(m.data[0, :])
        dx = np.copy(m.data[1, :])
        y = np.copy(m.data[2, :])
        dy = np.copy(m.data[3, :])
        if log:
            y *= offset
            dy *= offset
        else:
            y += offset

        p = ax.errorbar(
            x,
            y,
            yerr=dy,
            xerr=dx,
            marker="s",
            markersize=2,
            alpha=0.6,
            linestyle="none",
            linewidth=4,
        )

        if add_baseline:
            ax.plot([0, 180], [offset, offset], "k--", alpha=0.5)

        if label_xloc_deg is not None:
            if x[0] > 15 and x[-1] > 170:
                label_xloc_deg = -18
            elif x[-1] < 140:
                label_xloc_deg = 150
            else:
                label_xloc_deg = -18

        label_yloc_deg = np.mean(y)
        if log:
            label_yloc_deg *= label_offset_factor
        else:
            label_yloc_deg += label_offset_factor

        label_location = (label_xloc_deg, label_yloc_deg)

        if log:
            offset_text = f"\n($\\times$ {offset:0.0e})"
        else:
            offset_text = f"\n($+$ {offset:1.0f})"
        label = f"{E}"
        if label_energy_err:
            label += f" $\pm$ {E_err}"
        label += " MeV"
        if label_offset:
            label += offset_text

        ax.text(*label_location, label, fontsize=fontsize, color=p.lines[0].get_color())

    ax.set_xlabel(r"$\theta$ [{}]".format(units_x))
    ax.set_ylabel(r"{} [{}]".format(data_symbol, units_y))
    if log:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(f"{rxn_label}")

    return offsets
