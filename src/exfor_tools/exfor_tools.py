r"""
Library tools for parsing EXFOR entries
"""

import numpy as np
import periodictable
from functools import reduce
import jitr.utils.mass as mass

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

from .db import __EXFOR_DB__


# these are the supported quantities at the moment
quantity_matches = {
    "dXS/dA": [["DA"]],
    "dXS/dRuth": [["DA", "RTH"], ["DA", "RTH/REL"]],
    "Ay": [["POL/DA", "ANA"]],
}
quantities = list(quantity_matches.keys())

quantity_symbols = {
    ("DA",): r"$\frac{d\sigma}{d\Omega}$",
    ("DA", "RTH"): r"$\sigma / \sigma_{Rutherford}$",
    ("DA", "RTH/REL"): r"$\sigma / \sigma_{Rutherford}$",
    ("POL/DA", "ANA"): r"$A_y$",
}

unit_symbols = {"no-dim": "unitless", "barns/ster": "b/Sr"}


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


class EnergyDistribution:
    pass  # TODO


class AngularDistribution:
    """
    Stores angular distribution with x and y errors for a given incident and
    residual excitation energy. Allows for multiple y_errs with different
    labels.

    Attributes:
        subentry (str): Subentry identifier.
        Einc (float): Incident energy.
        Einc_err (float): Error in incident energy.
        Einc_units (str): Units of incident energy.
        Ex (float): Residual excitation energy.
        Ex_err (float): Error in residual excitation energy.
        Ex_units (str): Units of residual excitation energy.
        x_units (str): Units of x (angle).
        y_units (str): Units of y.
        x (np.ndarray): Sorted angles in degrees.
        x_err (np.ndarray): Errors in angles.
        y (np.ndarray): Sorted y values.
        y_errs (list): List of y errors.
        y_err_labels (list): Labels for y errors.
        rows (int): Number of data points.
    """

    def __init__(
        self,
        subentry: str,
        x: np.ndarray,
        x_err: np.ndarray,
        y: np.ndarray,
        y_errs: list,
        y_err_labels: str,
        Einc: float,
        Einc_err: float,
        Einc_units: str,
        Ex: float,
        Ex_err: float,
        Ex_units: str,
        x_units: str,
        y_units: str,
        quantity: str,
    ):
        self.subentry = subentry
        self.Einc = Einc
        self.Einc_err = Einc_err
        self.Einc_units = Einc_units
        self.Ex = Ex
        self.Ex_err = Ex_err
        self.Ex_units = Ex_units
        self.x_units = x_units
        self.y_units = y_units

        sort_by_angle = x.argsort()
        self.x = x[sort_by_angle]
        self.x_err = x_err[sort_by_angle]
        self.y = y[sort_by_angle]
        self.y_errs = [y_err[sort_by_angle] for y_err in y_errs]
        self.y_err_labels = y_err_labels
        self.rows = self.x.shape[0]
        self.quantity = quantity

        assert (
            np.all(self.x[1:] - self.x[:-1] >= 0)
            and self.x[0] >= 0
            and self.x[-1] <= 180
        )


def extract_syserr_labels(
    labels,
    allowed_sys_errs=frozenset(["ERR-SYS"]),
    allowed_stat_errs=frozenset(["DATA-ERR", "ERR-T", "ERR-S"]),
    expected_stat_errs=frozenset([]),
):
    """
    Extracts systematic error labels from a list of labels.

    Parameters:
    labels (list): A list of error labels.
    allowed_sys_errs (set): A set of allowed systematic error labels.
    allowed_stat_errs (set): A set of allowed statistical error labels.

    Returns:
    tuple: A tuple containing the systematic error labels and a string specifying
    the treatment

    Raises:
    ValueError: If the statistical error labels are ambiguous
    """
    allowed_sys_err_combos = frozenset([frozenset([l]) for l in allowed_sys_errs])
    sys_err_labels = (
        frozenset(labels)
        - allowed_stat_errs
        - frozenset(["ERR-DIG"])
        - expected_stat_errs
    )
    if len(sys_err_labels) == 0:
        return [], "independent"
    if sys_err_labels in allowed_sys_err_combos:
        return list(sys_err_labels), "independent"
    else:
        raise ValueError("Ambiguous systematic error labels: " + ", ".join(labels))


def extract_staterr_labels(
    labels,
    allowed_sys_errs=frozenset(["ERR-SYS"]),
    allowed_stat_errs=frozenset(["DATA-ERR", "ERR-T", "ERR-S"]),
    expected_sys_errs=frozenset([]),
):
    """
    Extracts statistical error labels from a list of labels.

    Parameters:
    labels (list): A list of error labels.
    allowed_sys_errs (set): A set of allowed systematic error labels.
    allowed_stat_errs (set): A set of allowed statistical error labels.

    Returns:
    tuple: A tuple containing the statistical error labels and a string specifying
    the treatment

    Raises:
    ValueError: If the statistical error labels are ambiguous
    """
    allowed_stat_err_combos = set(
        [frozenset([l, "ERR-DIG"]) for l in allowed_stat_errs]
        + [frozenset([l]) for l in allowed_stat_errs | frozenset(["ERR-DIG"])]
    )
    stat_err_labels = frozenset(labels) - allowed_sys_errs - expected_sys_errs
    if len(stat_err_labels) == 0:
        return [], "independent"
    if stat_err_labels in allowed_stat_err_combos:
        return list(stat_err_labels), "independent"
    else:
        raise ValueError("Ambiguous statistical error labels:\n" + ", ".join(labels))


class AngularDistributionStatErr(AngularDistribution):
    """
    AngularDistribution with one total statistical uncertainty.

    Attributes:
        statistical_err (np.ndarray): Total statistical error.
    """

    def __init__(
        self,
        *args,
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
    ):
        """
        Initialize an AngularDistributionStatErr instance.

        Args:
            *args: Variable length argument list for the base class.
            statistical_err_labels (list of str, optional): Labels for
                statistical errors. Defaults to [].
            statistical_err_treatment (str, optional): Method to treat
                statistical errors. Options are "independent" or "difference".
                Defaults to "independent".
            statistical_err_labels (list of str, optional): Labels for
                systematic errors. Defaults to []. Only relevant if
                statistical_err_labels is empty, meaning automatic parsing
                will be attempted.

        Raises:
            ValueError: If a column label in statistical_err_labels is not
            found in the subentry or if an unknown statistical_err_treatment
            is provided.
        """
        super().__init__(*args)
        if statistical_err_labels is None:
            statistical_err_labels, statistical_err_treatment = extract_staterr_labels(
                self.y_err_labels,
                expected_sys_errs=frozenset(
                    systematic_err_labels if systematic_err_labels is not None else []
                ),
            )

        self.statistical_err = np.zeros(
            (len(statistical_err_labels), self.rows), dtype=np.float64
        )

        for i, label in enumerate(statistical_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                self.statistical_err[i, :] = self.y_errs[index]

        if statistical_err_treatment == "independent":
            self.statistical_err = np.sqrt(np.sum(self.statistical_err**2, axis=0))
        elif statistical_err_treatment == "difference":
            self.statistical_err = -np.diff(self.statistical_err, axis=0)
        else:
            raise ValueError(
                f"Unknown statistical_err_treatment option: {statistical_err_treatment}"
            )

        self.y_err = self.statistical_err


class AngularDistributionSysStatErr(AngularDistributionStatErr):
    """
    AngularDistribution with a statistical uncertainty, and one or both
    of a systematic normalization or offset uncertainty

    Attributes:
        systematic_err (np.ndarray): Total systematic error.
    """

    def __init__(
        self,
        *args,
        statistical_err_labels=None,
        statistical_err_treatment="independent",
        systematic_err_labels=None,
        systematic_err_treatment="independent",
    ):
        """
        Initialize an AngularDistributionSysStatErr instance.

        Args:
            *args: Variable length argument list for the base class.
            statistical_err_labels (list of str, optional): Labels for
                statistical errors. Defaults to [].
            statistical_err_treatment (str, optional): Method to treat
                statistical errors. Options are "independent" or "difference".
                Defaults to "independent".
            systematic_err_labels (list of str, optional): Labels for
                systematic errors. Defaults to [].
            systematic_err_treatment (str, optional): Method to treat
                systematic errors. Options are "independent" or "difference".
                Defaults to "independent".

        Raises:
            ValueError: If a column label in systematic_err_labels is not
                found in the subentry, or if an unknown
                systematic_err_treatment is provided, or the systematic error
                column is non-uniform across angle.
        """
        super().__init__(
            *args,
            statistical_err_labels=statistical_err_labels,
            statistical_err_treatment=statistical_err_treatment,
            systematic_err_labels=systematic_err_labels,
        )

        if systematic_err_labels is None:
            systematic_err_labels, systematic_err_treatment = extract_syserr_labels(
                self.y_err_labels,
                expected_stat_errs=frozenset(
                    statistical_err_labels if statistical_err_labels is not None else []
                ),
            )

        self.systematic_offset_err = []
        self.systematic_norm_err = []
        self.general_systematic_err = []

        for i, label in enumerate(systematic_err_labels):
            if label not in self.y_err_labels:
                raise ValueError(
                    f"Did not find error column label {label} in subentry {self.subentry}"
                )
            else:
                index = self.y_err_labels.index(label)
                err = self.y_errs[index]
                ratio = err / self.y
                if np.allclose(ratio, ratio[0]):
                    self.systematic_norm_err.append(ratio[0])
                elif np.allclose(err, err[0]):
                    self.systematic_offset_err.append(err[0])
                else:
                    self.general_systematic_err.append(err)

        self.general_systematic_err = np.array(self.general_systematic_err)
        self.systematic_norm_err = np.array(self.systematic_norm_err)
        self.systematic_offset_err = np.array(self.systematic_offset_err)

        if systematic_err_treatment == "independent":
            self.systematic_offset_err = np.sqrt(
                np.sum(self.systematic_offset_err**2, axis=0)
            )
            self.systematic_norm_err = np.sqrt(
                np.sum(self.systematic_norm_err**2, axis=0)
            )
            self.general_systematic_err = np.sqrt(
                np.sum(self.general_systematic_err**2, axis=0)
            )
        else:
            raise ValueError(
                "Unknown systematic_err_treatment option:"
                f" {systematic_err_treatment}"
            )


def parse_differential_data(
    data_set,
    data_error_columns=["DATA-ERR"],
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
    match_idxs = data_parser.allMatches(data_set)
    if len(match_idxs) != 1:
        raise ValueError(f"Expected only one DATA column, found {len(match_idxs)}")
    iy = match_idxs[0]
    data_column = data_parser.getColumn(iy, data_set)
    xs_units = data_column[1]
    xs = np.array(data_column[2:], dtype=np.float64)

    # parse errors
    xs_err = []
    for label in data_error_columns:

        # parse error column
        err_parser = X4ColumnParser(
            match_labels=reduce(
                lambda x, y: x + y,
                [label],
            ),
            match_units=angDistUnits + percentUnits + noUnits,
        )

        if label not in data_set.labels:
            raise ValueError(f"Subentry does not have a column called {label}")
        else:
            iyerr = [idx for idx, value in enumerate(data_set.labels) if value == label]
            if len(iyerr) > 1:
                raise ValueError(
                    f"Expected only one {label} column, found {len(iyerr)}"
                )

            err = err_parser.getColumn(iyerr[0], data_set)
            err_units = err[1]
            err_data = np.nan_to_num(np.array(err[2:], dtype=np.float64))
            # convert to same units as data
            if "PER-CENT" in err_units:
                err_data *= xs / 100
            elif err_units != xs_units:
                raise ValueError(
                    f"Attempted to extract error column {err[0]} with incompatible units"
                    f"{err_units} for data column {data_column[0]} with units {xs_units}"
                )

            xs_err.append(err_data)

    return xs, xs_err, xs_units


def parse_ex_energy(data_set):
    Ex = reduce(condenseColumn, [c.getValue(data_set) for c in energyExParserList])

    missing_Ex = np.all([a is None for a in Ex[2:]])
    Ex_units = Ex[1]

    Ex = np.array(
        Ex[2:],
        dtype=np.float64,
    )
    if missing_Ex:
        return Ex, Ex, None

    if Ex[0][-3:] == "-CM":
        raise NotImplementedError("Incident energy in CM frame!")

    Ex_err = reduce(condenseColumn, [c.getError(data_set) for c in energyExParserList])
    if Ex_err[1] != Ex_units:
        raise ValueError(
            f"Inconsistent units for Ex and Ex error: {Ex_units} and {Ex_err[1]}"
        )
    Ex_err = np.array(
        Ex_err[2:],
        dtype=np.float64,
    )

    return Ex, Ex_err, Ex_units


def parse_angle(data_set):
    # TODO handle cosine or momentum transfer
    # TODO how does this handle multiple matched entries
    angle = reduce(condenseColumn, [c.getValue(data_set) for c in angleParserList])
    if angle[1] != "degrees":
        raise ValueError(f"Cannot parse angle in units of {angle[1]}")
    if angle[0][-3:] != "-CM":
        raise NotImplementedError("Angle in lab frame!")
    angle = np.array(
        angle[2:],
        dtype=np.float64,
    )
    angle_err = reduce(condenseColumn, [c.getError(data_set) for c in angleParserList])
    missing_err = np.all([a is None for a in angle_err[2:]])
    if not missing_err:
        if angle_err[1] != "degrees":
            raise ValueError(f"Cannot parse angle error in units of {angle_err[1]}")
    angle_err = np.array(
        angle_err[2:],
        dtype=np.float64,
    )
    return angle, angle_err, "degrees"


def parse_inc_energy(data_set):
    Einc_lab = reduce(
        condenseColumn, [c.getValue(data_set) for c in incidentEnergyParserList]
    )

    Einc_units = Einc_lab[1]
    if Einc_lab[0][-3:] == "-CM":
        raise NotImplementedError("Incident energy in CM frame!")

    Einc_lab = np.array(
        Einc_lab[2:],
        dtype=np.float64,
    )

    Einc_lab_err = reduce(
        condenseColumn, [c.getError(data_set) for c in incidentEnergyParserList]
    )
    missing_err = np.all([a is None for a in Einc_lab_err[2:]])
    if not missing_err:
        if Einc_lab_err[1] != Einc_units:
            raise ValueError(
                f"Inconsistent units for Einc and Einc error: {Einc_units} and {Einc_lab_err[1]}"
            )
    Einc_lab_err = np.array(
        Einc_lab_err[2:],
        dtype=np.float64,
    )

    return Einc_lab, Einc_lab_err, Einc_units


def parse_angular_distribution(
    subentry,
    data_set,
    data_error_columns=None,
    vocal=False,
):
    r"""
    Extracts angular differential cross sections, returning incident and product excitation
    energy in MeV, angles and error in angle in degrees, and differential cross section and its
    error in mb/Sr all in a numpy array.
    """
    if vocal:
        print(
            f"Found subentry {subentry} with the following columns:\n{data_set.labels}"
        )

    if data_error_columns is None:
        data_error_columns = [b + "-ERR" for b in baseDataKeys] + dataTotalErrorKeys

    try:
        # parse angle
        angle, angle_err, angle_units = parse_angle(data_set)

        # parse energy if it's present
        Einc_lab, Einc_lab_err, Einc_units = parse_inc_energy(data_set)

        # parse excitation energy if it's present
        Ex, Ex_err, Ex_units = parse_ex_energy(data_set)

        # parse diff xs
        xs, xs_err, xs_units = parse_differential_data(
            data_set, data_error_columns=data_error_columns
        )
    except Exception as e:
        new_exception = type(e)(f"Error while parsing {subentry}: {e}")
        raise new_exception from e

    N = data_set.numrows()
    n_err_cols = len(xs_err)
    data_err = np.zeros((n_err_cols, N))
    data = np.zeros((7, N))

    data[:, :] = [
        Einc_lab,
        np.nan_to_num(Einc_lab_err),
        np.nan_to_num(Ex),
        np.nan_to_num(Ex_err),
        angle,
        np.nan_to_num(angle_err),
        xs,
    ]

    if vocal:
        if len(xs_err) == 0 or np.all([np.allclose(d, 0) for d in xs_err]):
            print(f"Warning: subentry {subentry} has no reported data errors")
            xs_err = np.zeros((n_err_cols, N))

    data_err[:, :] = np.nan_to_num(xs_err)

    return (
        data,
        data_err,
        data_error_columns,
        (angle_units, Einc_units, Ex_units, xs_units),
    )


def attempt_parse_subentry(*args, **kwargs):
    failed_parses = {}
    measurements = []
    try:
        measurements = get_measurements_from_subentry(*args, **kwargs)
    except Exception as e:
        subentry = args[0]
        if kwargs.get("vocal", False):
            print(f"Failed to parse subentry {subentry}:\n\t{e}")
        failed_parses[subentry] = e

    return measurements, dict(failed_parses)


def get_measurements_from_subentry(
    subentry,
    data_set,
    quantity: str,
    Einc_range=(0, np.inf),
    Ex_range=(0, np.inf),
    elastic_only=False,
    vocal=False,
    MeasurementClass=AngularDistribution,
    parsing_kwargs={},
):
    r"""unrolls subentry into individual arrays for each energy"""

    Einc = parse_inc_energy(data_set)[0]
    Ex = np.nan_to_num(parse_ex_energy(data_set)[0])
    if not np.any(
        np.logical_and(
            np.logical_and(Einc >= Einc_range[0], Einc <= Einc_range[1]),
            np.logical_and(Ex >= Ex_range[0], Ex <= Ex_range[1]),
        )
    ):
        return []

    lbl_frags_to_skip = ["ANG", "EN", "E-LVL", "E-EXC"]
    err_labels = [
        label
        for label in data_set.labels
        if "ERR" in label and np.all([frag not in label for frag in lbl_frags_to_skip])
    ]

    data, data_err, error_columns, units = parse_angular_distribution(
        subentry,
        data_set,
        data_error_columns=err_labels,
        vocal=vocal,
    )

    measurement_data = sort_subentry_data_by_energy(
        subentry,
        data,
        data_err,
        error_columns,
        Einc_range,
        Ex_range,
        elastic_only,
        units,
        quantity,
    )
    measurements = [MeasurementClass(*m, **parsing_kwargs) for m in measurement_data]
    return measurements


def sort_subentry_data_by_energy(
    subentry,
    data,
    data_err,
    error_columns,
    Einc_range,
    Ex_range,
    elastic_only,
    units,
    quantity,
):
    angle_units, Einc_units, Ex_units, xs_units = units
    Einc_mask = np.logical_and(
        data[0, :] >= Einc_range[0],
        data[0, :] <= Einc_range[1],
    )
    data = data[:, Einc_mask]
    data_err = data_err[:, Einc_mask]

    if not elastic_only:
        Ex_mask = np.logical_and(
            data[2, :] >= Ex_range[0],
            data[2, :] <= Ex_range[1],
        )
        data = data[:, Ex_mask]
        data_err = data_err[:, Ex_mask]

    # AngularDistribution objects sorted by incident energy,
    # then excitation energy or just incident enrgy if
    # elastic_only is True
    measurements = []

    # find set of unique incident energies
    unique_Einc = np.unique(data[0, :])

    # sort and fragment data by unique incident energy
    for Einc in np.sort(unique_Einc):
        mask = np.isclose(data[0, :], Einc)
        Einc_err = data[1, mask][0]

        if elastic_only:
            measurements.append(
                (
                    subentry,
                    data[4, mask],
                    data[5, mask],
                    data[6, mask],
                    [data_err[i, mask] for i in range(data_err.shape[0])],
                    error_columns,
                    Einc,
                    Einc_err,
                    Einc_units,
                    0,
                    0,
                    Ex_units,
                    angle_units,
                    xs_units,
                    quantity,
                )
            )
        else:
            subset = data[2:, mask]
            subset_err = data_err[:, mask]

            # find set of unique residual excitation energies
            unique_Ex = np.unique(subset[0, :])

            # sort and fragment data by unique excitation energy
            for Ex in np.sort(unique_Ex):
                mask = np.isclose(subset[0, :], Ex)
                Ex_err = subset[1, mask][0]
                measurements.append(
                    (
                        subentry,
                        subset[2, mask],
                        subset[3, mask],
                        subset[4, mask],
                        [subset_err[i, mask] for i in range(data_err.shape[0])],
                        error_columns,
                        Einc,
                        Einc_err,
                        Einc_units,
                        Ex,
                        Ex_err,
                        Ex_units,
                        angle_units,
                        xs_units,
                        quantity,
                    )
                )
    return measurements


def get_exfor_particle_symbol(A, Z):
    return {
        (1, 0): "N",
        (1, 1): "P",
        (2, 1): "D",
        (3, 1): "T",
        (4, 2): "A",
    }.get(
        (A, Z),
        f"{str(periodictable.elements[Z])}-{A}",
    )


class Reaction:
    """Represents a simple A + a -> b + B reaction"""

    # TODO specify if elastic only
    # TODO generalize to multiple products or specific excited residual states
    # TODO put in jitR along with ripl levels/LDs, make nucleus class
    def __init__(self, target, projectile, residual=None, product=None, mass_kwargs={}):
        if product is None:
            product = projectile
        if residual is None:
            residual = target

        self.target = target
        self.projectile = projectile
        self.product = product
        self.residual = residual

        Apre = self.target[0] + self.projectile[0]
        Apost = self.residual[0] + self.product[0]
        Zpre = self.target[1] + self.projectile[1]
        Zpost = self.residual[1] + self.product[1]

        self.mass_target = mass.mass(*self.target, **mass_kwargs)[0]
        self.mass_projectile = mass.mass(*self.projectile, **mass_kwargs)[0]
        self.mass_residual = mass.mass(*self.residual, **mass_kwargs)[0]
        self.mass_product = mass.mass(*self.product, **mass_kwargs)[0]
        self.Q = (
            self.mass_projectile
            + self.mass_target
            - self.mass_residual
            - self.mass_product
        )

        if Apre != Apost and Zpre != Zpost:
            raise ValueError("Isospin not conserved in this reaction")

        self.exfor_symbol_target = get_exfor_particle_symbol(*self.target)
        self.exfor_symbol_residual = get_exfor_particle_symbol(*self.residual)
        self.exfor_symbol_projectile = get_exfor_particle_symbol(*self.projectile)
        self.exfor_symbol_product = get_exfor_particle_symbol(*self.product)

        self.symbol_target = get_symbol(*self.target)
        self.symbol_residual = get_symbol(*self.residual)
        self.symbol_projectile = get_symbol(*self.projectile)
        self.symbol_product = get_symbol(*self.product)

        if self.residual == self.target:
            self.pretty_string = (
                f"{self.symbol_target}$({self.symbol_projectile},"
                f"{self.symbol_product})$"
            )
        else:
            self.pretty_string = (
                f"{self.symbol_target}$({self.symbol_projectile},"
                f"{self.symbol_product})${self.symbol_residual}"
            )

    def is_match(self, subentry, vocal=False):
        target = (
            subentry.reaction[0].targ.getA(),
            subentry.reaction[0].targ.getZ(),
        )
        projectile = (
            subentry.reaction[0].proj.getA(),
            subentry.reaction[0].proj.getZ(),
        )
        if target != self.target or projectile != self.projectile:
            return False

        # parse product and residual
        product = subentry.reaction[0].products[0]
        if product == "EL":
            product = projectile
        elif isinstance(product, str):
            # TODO use inheritance in Reaction class to handle more general
            # case in which there may be multiple or no residuals or products
            return False  # not an A + a -> B + b reaction

        else:
            product = (
                subentry.reaction[0].products[0].getA(),
                subentry.reaction[0].products[0].getZ(),
            )
        if subentry.reaction[0].residual is None:
            return False
        else:
            residual = (
                subentry.reaction[0].residual.getA(),
                subentry.reaction[0].residual.getZ(),
            )
        return product == self.product and residual == self.residual

    def __eq__(self, other):
        if not isinstance(other, Reaction):
            return False
        return (self.target, self.projectile, self.product, self.residual) == (
            other.target,
            other.projectile,
            other.product,
            other.residual,
        )

    def __hash__(self):
        return hash((self.target, self.projectile, self.product, self.residual))


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
        if Ex is None:
            return f"$^{{{A}}}${str(periodictable.elements[Z])}"
        else:
            ex = f"({float(Ex):1.3f})"
            return f"$^{{{A}}}${str(periodictable.elements[Z])}{ex}"


def filter_subentries(data_set, filter_lab_angle=True, min_num_pts=4):
    angle_labels = [
        l
        for l in data_set.labels
        if (
            "ANG" in l
            and "-NRM" not in l
            and np.all(
                [
                    f not in l
                    for f in errorSuffix + resolutionFWSuffix + resolutionHWSuffix
                ]
            )
        )
    ]

    if len(angle_labels) == 0:
        return False
    if min_num_pts is not None:
        if data_set.numrows() < min_num_pts:
            return False
    if filter_lab_angle:
        if "-CM" not in angle_labels[0]:
            return False
    return True


def extract_err_analysis(common_subent):
    sections = common_subent.__repr__().split("\n")
    ea_sections = []
    start = False
    for section in sections:
        if start:
            if section[0] == " ":
                start = True
            else:
                break
            ea_sections.append(section)

        if section.find("ERR-ANAL") >= 0:
            ea_sections.append(section)
            start = True
    return "\n".join(ea_sections)


class ExforEntry:

    def __init__(
        self,
        entry: str,
        reaction: Reaction,
        quantity: str,
        Einc_range: tuple = None,
        Ex_range: tuple = None,
        vocal=False,
        MeasurementClass=AngularDistributionSysStatErr,
        parsing_kwargs={},
        filter_kwargs={},
    ):
        r""" """
        if "min_num_pts" not in filter_kwargs:
            filter_kwargs["min_num_pts"] = 4

        self.vocal = vocal
        self.entry = entry
        self.reaction = reaction
        if Einc_range is None:
            Einc_range = (0, np.inf)
        self.Einc_range = Einc_range

        elastic_only = False
        if (
            Ex_range is None
            and self.reaction.product == self.reaction.projectile
            and self.reaction.residual == self.reaction.target
        ):
            Ex_range = (0, 0)
            elastic_only = True
        elif Ex_range is None:
            Ex_range = (0, np.inf)

        self.Ex_range = Ex_range

        self.quantity = quantity
        self.exfor_quantities = quantity_matches[quantity]
        self.data_symbol = quantity_symbols[tuple(self.exfor_quantities[0])]

        # parsing
        entry_data = __EXFOR_DB__.retrieve(ENTRY=entry)[entry]
        subentry_ids = entry_data.keys()

        # parse common
        self.meta = None
        self.err_analysis = None
        self.common_labels = []
        self.normalization_uncertainty = 0

        if entry + "001" not in subentry_ids:
            raise ValueError(f"Missing first subentry in entry {entry}")
        elif entry_data[entry + "001"] is not None:
            common_subentry = entry_data[entry + "001"]
            self.meta = common_subentry["BIB"].meta(entry + "001")

            # parse any common errors
            self.err_analysis = extract_err_analysis(common_subentry)
            if "COMMON" in common_subentry.keys():
                common = common_subentry["COMMON"]
                self.common_labels = common.labels

        self.subentry_err_analysis = {}
        for subentry in subentry_ids:
            self.subentry_err_analysis[subentry] = extract_err_analysis(entry_data[subentry])

        entry_datasets = entry_data.getDataSets()
        self.subentries = [key[1] for key in entry_datasets.keys()]
        self.measurements = []
        self.failed_parses = {}

        for key, data_set in entry_datasets.items():

            if not isinstance(data_set.reaction[0], X4Reaction):
                # TODO handle ReactionCombinations
                continue

            quantity = data_set.reaction[0].quantity

            if quantity[-1] == "EXP":
                quantity = quantity[:-1]

            # matched reaction
            if (
                quantity in self.exfor_quantities
                and filter_subentries(data_set, **filter_kwargs)
                and self.reaction.is_match(data_set, self.vocal)
            ):

                measurements, failed_parses = attempt_parse_subentry(
                    key[1],
                    data_set,
                    self.quantity,
                    Einc_range=self.Einc_range,
                    Ex_range=self.Ex_range,
                    elastic_only=elastic_only,
                    vocal=vocal,
                    MeasurementClass=MeasurementClass,
                    parsing_kwargs=parsing_kwargs,
                )
                for m in measurements:
                    if m.x.size < filter_kwargs["min_num_pts"]:
                        continue
                    self.measurements.append(m)
                for subentry, e in failed_parses.items():
                    self.failed_parses[key[0]] = (subentry, e)

    def plot(
        self,
        ax,
        offsets=None,
        log=True,
        draw_baseline=False,
        baseline_offset=None,
        xlim=[0, 180],
        fontsize=10,
        label_kwargs={
            "label_offset_factor": 2,
            "label_energy_err": False,
            "label_offset": True,
        },
    ):
        plot_angular_distributions(
            self.measurements,
            ax,
            offsets,
            self.data_symbol,
            self.reaction.pretty_string,
            log,
            draw_baseline,
            baseline_offset,
            xlim,
            fontsize,
            label_kwargs,
        )


def set_label(
    ax,
    measurements: list,
    colors: list,
    offset,
    x,
    y,
    log,
    fontsize=10,
    label_xloc_deg=None,
    label_offset_factor=2,
    label_energy_err=False,
    label_offset=True,
    label_incident_energy=True,
    label_excitation_energy=False,
    label_exfor=False,
):

    # TODO when there is more than one measurement, make each subentry label correspond to its
    # corresponding color: https://matplotlib.org/1.5.0/examples/text_labels_and_annotations/rainbow_text.html
    if label_xloc_deg is None:
        if x[-1] < 60:
            label_xloc_deg = 65
        elif x[-1] < 90:
            label_xloc_deg = 95
        elif x[-1] < 120:
            label_xloc_deg = 125
        elif x[0] > 30 and x[-1] > 150:
            label_xloc_deg = 1
        elif x[0] > 20 and x[-1] > 150:
            label_xloc_deg = -18
        elif x[-1] < 150:
            label_xloc_deg = 155
        else:
            label_xloc_deg = 175

    label_yloc = offset
    if log:
        label_yloc *= label_offset_factor
    else:
        label_yloc += label_offset_factor

    label_location = (label_xloc_deg, label_yloc)

    if log:
        offset_text = f"\n($\\times$ {offset:1.0e})"
    else:
        offset_text = f"\n($+$ {offset:1.0f})"

    m = measurements[0]
    label = ""
    if label_incident_energy:
        label += f"\n{m.Einc:1.2f}"
        if label_energy_err:
            label += f" $\pm$ {m.Einc_err:1.2f}"
        label += f" {m.Einc_units}"
    if label_excitation_energy:
        label += f"\n$E_{{x}} = ${m.Ex:1.2f}"
        if label_energy_err:
            label += f" $\pm$ {m.Ex_err:1.2f}"
        label += f" {m.Ex_units}"
    if label_exfor:
        label += "\n"
        for i, m in enumerate(measurements):
            if i == len(measurements) - 1:
                label += f"{m.subentry}"
            else:
                label += f"{m.subentry},\n"
    if label_offset:
        label += offset_text

    t = ax.text(*label_location, label, fontsize=fontsize, color=colors[-1])


def plot_errorbar(ax, x, x_err, y, y_err, offset, log):
    if log:
        y *= offset
        y_err *= offset
    else:
        y += offset

    p = ax.errorbar(
        x,
        y,
        yerr=y_err,
        xerr=x_err,
        marker="s",
        markersize=2,
        alpha=0.75,
        linestyle="none",
        elinewidth=3,
        # capthick=2,
        # capsize=1,
    )
    return p.lines[0].get_color()


def plot_angular_distributions(
    measurements,
    ax,
    offsets=None,
    data_symbol="",
    rxn_label="",
    log=True,
    draw_baseline=False,
    baseline_offset=None,
    xlim=[0, 180],
    fontsize=10,
    label_kwargs={
        "label_offset_factor": 2,
        "label_energy_err": False,
        "label_offset": True,
    },
):
    r"""
    Given a collection of measurements, plots them on the same axis with offsets
    """
    # if offsets is not a sequence, figure it out
    # TODO do the same for label_offset_factor
    if isinstance(offsets, float) or isinstance(offsets, int) or offsets is None:
        if offsets is None:
            constant_factor = 1 if log else 0
        else:
            constant_factor = offsets
        if log:
            offsets = constant_factor ** np.arange(0, len(measurements))
        else:
            offsets = constant_factor * np.arange(0, len(measurements))

    # plot each measurement and add a label
    for offset, m in zip(offsets, measurements):

        if not isinstance(m, list):
            m = [m]

        c = []
        for measurement in m:
            x = np.copy(measurement.x)
            x_err = np.copy(measurement.x_err)
            y = np.copy(measurement.y)
            y_err = np.copy(measurement.y_err)
            color = plot_errorbar(
                ax,
                np.copy(measurement.x),
                np.copy(measurement.x_err),
                np.copy(measurement.y),
                np.copy(measurement.y_err),
                offset,
                log,
            )
            c.append(color)

        if draw_baseline:
            if log:
                baseline_offset = baseline_offset if baseline_offset is not None else 1
                baseline_height = offset * baseline_offset
            else:
                baseline_offset = baseline_offset if baseline_offset is not None else 0
                baseline_height = offset + baseline_offset
            ax.plot([0, 180], [baseline_height, baseline_height], "k--", alpha=0.25)

        if label_kwargs is not None:
            set_label(ax, m, c, offset, x, y, log, fontsize, **label_kwargs)

    if isinstance(measurements[0], list):
        x_units = unit_symbols.get(
            measurements[0][0].x_units, measurements[0][0].x_units
        )
        y_units = unit_symbols.get(
            measurements[0][0].y_units, measurements[0][0].y_units
        )
    else:
        x_units = unit_symbols.get(measurements[0].x_units, measurements[0].x_units)
        y_units = unit_symbols.get(measurements[0].y_units, measurements[0].y_units)

    ax.set_xlabel(r"$\theta$ [{}]".format(x_units))
    ax.set_ylabel(r"{} [{}]".format(data_symbol, y_units))
    ax.set_xticks(np.arange(0, 180.01, 30))
    if log:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(f"{rxn_label}")

    if log:
        ax.set_yscale("log")

    return offsets
