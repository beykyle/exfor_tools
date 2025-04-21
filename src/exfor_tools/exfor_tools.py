r"""
Library tools for parsing EXFOR entries
"""

import numpy as np

from x4i3.exfor_reactions import X4Reaction
from x4i3.exfor_column_parsing import (
    errorSuffix,
    resolutionFWSuffix,
    resolutionHWSuffix,
)
from .db import __EXFOR_DB__

from .reaction import Reaction, ElasticReaction
from .parsing import (
    parse_angular_distribution,
    parse_inc_energy,
    parse_ex_energy,
    quantity_matches,
    quantity_symbols,
    unit_symbols,
)
from .angular_distribution import AngularDistributionSysStatErr


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
    MeasurementClass=AngularDistributionSysStatErr,
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
        if isinstance(reaction, ElasticReaction):
            elastic_only = True
            Ex_range = (0, 0)
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
            raise ValueError(f"Missing first subentry filter_in entry {entry}")
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
            self.subentry_err_analysis[subentry] = extract_err_analysis(
                entry_data[subentry]
            )

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
            self.reaction.reaction_latex,
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

    ax.text(*label_location, label, fontsize=fontsize, color=colors[-1])


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
            y = np.copy(measurement.y)
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
