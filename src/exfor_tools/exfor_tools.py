import numpy as np
import periodictable

from x4i3 import exfor_manager
from x4i3.exfor_reactions import X4Reaction


__EXFOR_DB__ = None


def init_exfor_db():
    global __EXFOR_DB__
    if __EXFOR_DB__ is None:
        __EXFOR_DB__ = exfor_manager.X4DBManagerDefault()


def get_exfor_differential_data(
    target, projectile, quantity, energy_range=None, product=None
):
    r"""query EXFOR for all entries satisfying search criteria, and return them as a dictionary of entry number to ExforDifferentialData"""
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
            data_set = ExforDifferentialData(
                entry=entry,
                target=target,
                projectile=projectile,
                quantity=quantity,
                products=[product],
                energy_range=energy_range,
            )
        except ValueError as e:
            print(f"There was an error reading entry {entry}, it will be skipped:")
            print(e)
        if len(data_set.measurements) > 0 and entry not in data_sets:
            data_sets[entry] = data_set

    return data_sets


def sort_measurements_by_energy(all_entries, min_num_pts=5):
    r"""given a dictionary form EXFOR entry number to ExforDifferentialData, grabs all the ExforDifferentialDataSet's and sorts them by energy, concatenating ones that are at the same energy"""
    measurements = []
    energies = []
    for entry, data in all_entries.items():
        for measurement in data.measurements:
            if measurement.data.shape[1] > min_num_pts:
                energies.append(measurement.Ecm)
                measurements.append(measurement)

    energies = np.array(energies)
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

        # re-sort data by anglwe
        data = data[:, data[0, :].argsort()]

        measurements_condensed.append(
            ExforDifferentialDataSet(
                m.Ecm, m.dEcm, m.energy_units, m.units, m.labels, data
            )
        )
    return measurements_condensed


# these are the supported quantities at the moment
# XS = cross section, A = angle, Ruth = Rutherford cross section, Ay = analyzing power
# TODO add DE (differential with energy)
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

unit_conversions = dict(
    zip(
        ["MEV", "KEV", "ADEG", "NO-DIM", "barns/ster", "MB/SR", "no-dim"],
        [
            (1.0, "MeV"),
            (1e-3, "MeV"),
            (1.0, "degrees"),
            (1.0, "dimensionless"),
            (1.0e3, "mb/Sr"),
            (1.0, "mb/Sr"),
            (1.0, "dimensionless"),
        ],
    )
)


class ExforDifferentialDataSet:
    def __init__(self, Ecm, dEcm, energy_units, units, labels, data):
        self.Ecm = Ecm
        self.dEcm = dEcm
        self.energy_units = energy_units
        self.units = units
        self.labels = labels
        self.data = data


class ExforDifferentialData:
    r"""Assumes data is given as [.., angle, quantity, quantity err, ...]"""

    def __init__(
        self,
        entry,
        target,
        projectile,
        quantity,
        products,
        energy_range=None,
        subentry=None,
    ):
        self.entry = entry
        entry_data = __EXFOR_DB__.retrieve(ENTRY=entry)[entry]
        self.isotope = target
        self.projectile = projectile
        self.quantity = quantity
        self.exfor_quantities = quantity_matches[quantity]
        self.data_symbol = quantity_symbols[tuple(self.exfor_quantities[0])]
        self.products = products
        if energy_range is None:
            energy_range = (0, np.inf)
        self.energy_range = energy_range

        data_sets = []
        self.subentry = [key[1] for key in entry_data.getSimplifiedDataSets().keys()]
        self.measurements = []

        for key, data_set in entry_data.getSimplifiedDataSets().items():
            subentry = key[1]

            # handle common data
            common_energy = None
            common_energy_uncertainty = None

            if "COMMON" in entry_data[subentry]:
                common = entry_data[subentry]["COMMON"]
                if "EN" in common.labels:
                    # COMMON energy
                    i = np.argmax([l == "EN" for l in common.labels])
                    common_energy = common.data[0][i]
                    common_energy_unit = common.units[i]
                    if common_energy_unit in unit_conversions:
                        conversion, common_energy_unit = unit_conversions[
                            common_energy_unit
                        ]
                        common_energy *= conversion
                    if common_energy_unit != "MeV":
                        raise ValueError(f"Unknown energy unit: {common_energy_unit}")
                    if "EN-RSL" in common.labels:
                        # COMMON energy uncertainty
                        i = np.argmax([l == "EN-RSL" for l in common.labels])
                        common_energy_uncertainty = common.data[0][i]
                        unit = common.units[i]
                        if unit in unit_conversions:
                            conversion, unit = unit_conversions[unit]
                            common_energy_uncertainty *= conversion
                        if unit != "MeV":
                            raise ValueError(f"Unknown energy resolution unit: {unit}")

            if isinstance(data_set.reaction[0], X4Reaction):
                isotope = (
                    data_set.reaction[0].targ.getA(),
                    data_set.reaction[0].targ.getZ(),
                )
                projectile = (
                    data_set.reaction[0].proj.getA(),
                    data_set.reaction[0].proj.getZ(),
                )
                quantity = data_set.reaction[0].quantity
                products = data_set.reaction[0].products
                if quantity[-1] == "EXP":
                    quantity = quantity[:-1]
                if (
                    isotope == self.isotope
                    and projectile == self.projectile
                    and quantity in self.exfor_quantities
                    and products == self.products
                ):
                    data_sets.append(data_set)
                    measurements = self.get_measurements_from_subentry(
                        data_set,
                        common_energy=common_energy,
                        common_energy_uncertainty=common_energy_uncertainty,
                    )
                    for m in measurements:
                        self.measurements.append(m)

        if len(data_sets) > 0:
            # assume all subentries have same meta info
            self.symbol = data_sets[0].reaction[0].targ.sym
            proj = data_sets[0].reaction[0].proj.prettyStyle()
            res = data_sets[0].reaction[0].products[0].lower()
            rxn = r"$({},{})$".format(proj, res)
            self.fancy_label = r"$^{%d}$%s " % (self.isotope[0], self.symbol) + rxn
            self.meta = {
                "author": data_sets[0].author,
                "title": data_sets[0].title,
                "year": data_sets[0].year,
                "institute": data_sets[0].institute,
            }

    def get_measurements_from_subentry(
        self,
        data_set,
        common_energy=None,
        common_energy_uncertainty=None,
    ):
        r"""unrolls subentry into individual arrays for each energy"""
        data_array = np.array(data_set.data)

        # TODO in the case that number of unique energies > number of angles,
        # then store as single-angle data sets for a range of energies

        # sanitize labels and convert units
        for i in range(len(data_set.labels)):
            if data_set.labels[i] in label_matches:
                data_set.labels[i] = label_matches[data_set.labels[i]]
            if data_set.units[i] in unit_conversions:
                conversion, data_set.units[i] = unit_conversions[data_set.units[i]]
                data_array[:, i] *= conversion

        # these fields are mandatory
        if "Angle" not in data_set.labels:
            raise ValueError("Missing 'Angle' field!")
            # TODO allow for momentum transfer as well as angle
        if "Data" not in data_set.labels:
            raise ValueError("Missing 'Data' field!")

        xi = np.argmax(np.array(data_set.labels) == "Angle")
        yi = np.argmax(np.array(data_set.labels) == "Data")
        x = data_array[:, xi]
        y = data_array[:, yi]

        # these fields are optional
        if "d(Data)" in data_set.labels:
            dyi = np.argmax(np.array(data_set.labels) == "d(Data)")
            dy = data_array[:, dyi]
            if data_set.units[yi] != data_set.units[dyi]:
                raise ValueError(
                    "Inconsistent units between 'Data' and 'd(Data)' fields :"
                    + f"{data_set.units[yi]} and {data_set.units[dyi]}"
                )
        else:
            dy = np.zeros(len(x))

        if "d(Angle)" in data_set.labels:
            dxi = np.argmax(np.array(data_set.labels) == "d(Angle)")
            dx = data_array[:, dxi]
            if data_set.units[xi] != data_set.units[dxi]:
                raise ValueError(
                    "Inconsistent units between 'Angle' and 'd(Angle)' fields :"
                    + f"{data_set.units[xi]} and {data_set.units[dxi]}"
                )
        else:
            dx = np.zeros(len(x))

        # put em all together
        all_data = np.vstack([x, dx, y, dy])
        labels = [data_set.labels[xi], data_set.labels[yi]]
        units = [data_set.units[xi], data_set.units[yi]]

        # split up data by energy (if applicable)
        energy_errs = []
        energies = []
        measurements = []

        if "Energy" in data_set.labels:
            e_idx = np.argmax(np.array(data_set.labels) == "Energy")
            energy_units = data_set.units[e_idx]

            for energy in np.unique(data_array[:, e_idx]):
                mask = data_array[:, e_idx] == energy
                energies.append(energy)
                measurements.append(np.copy(all_data[:, mask]))

            # TODO x4i3 gives the wrong error in energy sometimes?
            if "d(Energy)" in data_set.labels:
                de_idx = np.argmax(np.array(data_set.labels) == "d(Energy)")
                for energy in np.unique(data_array[:, e_idx]):
                    mask = data_array[:, e_idx] == energy
                    energy_errs.append(data_array[mask, de_idx][0])

            else:
                if common_energy_uncertainty is None:
                    energy_errs = np.zeros(len(energies))
                else:
                    energy_errs = np.ones(len(energies)) * common_energy_uncertainty

        else:
            # no need to split into energies, it's all in one
            energies = [common_energy]
            energy_errs = [common_energy_uncertainty]
            measurements = [all_data]

        # sort by energy
        sorted_data = sorted(zip(energies, energy_errs, measurements))
        return [
            ExforDifferentialDataSet(
                energy, energy_err, energy_units, units, labels, measurement
            )
            for energy, energy_err, measurement in sorted_data
            if energy >= self.energy_range[0] and energy < self.energy_range[1]
        ]

    def plot_experiment(
        self,
        ax,
        measurements=None,
        offsets=None,
        label_hloc_deg=130,
        label_offset_factor=2,
        log=True,
        add_baseline=False,
        xlim=[0, 180],
        label_energy_err=True,
        label_offset=True,
    ):
        if measurements is None:
            measurements = self.measurements

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

        # assume all units are the same, which should be true if santization worked
        units_x = measurements[0].units[0]
        units_y = measurements[0].units[1]

        # plot each measurement and add a label
        for offset, m in zip(offsets, measurements):
            # plot the data
            x = np.copy(m.data[0, :])
            dx = np.copy(m.data[1, :])
            y = np.copy(m.data[2, :])
            dy = np.copy(m.data[3, :])
            if log:
                y *= offset
                dy *= offset
            else:
                y += offset

            ax.errorbar(x, y, yerr=dy, xerr=dx, color="k", linestyle="None", marker=".")

            if add_baseline:
                ax.plot([0, 180], [offset, offset], "k--", alpha=0.5)

            hloc_deg = label_hloc_deg
            yloc_deg = np.mean(y[np.argmin(np.fabs(x - hloc_deg)) :])
            if log:
                yloc_deg *= label_offset_factor
            else:
                yloc_deg += label_offset_factor
            label_location = (hloc_deg, yloc_deg)

            if log:
                offset_text = f"\n($\\times$ {offset:0.0e})"
            else:
                offset_text = f"\n($+$ {offset:1.0f})"
            label = f"{m.Ecm}"
            if label_energy_err:
                label += f" $\pm$ {m.dEcm}"
            label += f" {m.energy_units}"
            if label_offset:
                label += offset_text

            ax.text(
                *label_location,
                label,
                fontsize=8,
            )

        ax.set_xlabel(r"$\theta$ [{}]".format(units_x))
        ax.set_ylabel(r"{} [{}]".format(self.data_symbol, units_y))
        if log:
            ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(f"{self.fancy_label}")
