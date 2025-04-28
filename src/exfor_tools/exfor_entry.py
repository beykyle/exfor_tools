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

from .parsing import quantity_matches, quantity_symbols
from .distribution import AngularDistribution
from . import reaction as rxn


def attempt_parse_subentry(MeasurementClass, *args, **kwargs):
    failed_parses = {}
    measurements = []
    try:
        measurements = MeasurementClass.parse_subentry(*args, **kwargs)
    except Exception as e:
        subentry = args[0]
        if kwargs.get("vocal", False):
            print(f"Failed to parse subentry {subentry}:\n\t{e}")
        failed_parses[subentry] = e

    return measurements, dict(failed_parses)


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
        reaction: rxn.Reaction,
        quantity: str,
        Einc_range: tuple = None,
        Ex_range: tuple = None,
        vocal=False,
        MeasurementClass=None,
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
        if isinstance(reaction, rxn.ElasticReaction):
            elastic_only = True
            Ex_range = (0, 0)
        elif Ex_range is None:
            Ex_range = (0, np.inf)

        self.Ex_range = Ex_range

        self.quantity = quantity
        if MeasurementClass is None:
            if (
                self.quantity == "dXS/dA"
                or self.quantity == "dXS/dRuth"
                or self.quantity == "Ay"
            ):
                MeasurementClass = AngularDistribution
            else:
                raise NotImplementedError()
        self.MeasurementClass = MeasurementClass
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
                and rxn.is_match(self.reaction, data_set, self.vocal)
            ):

                measurements, failed_parses = attempt_parse_subentry(
                    MeasurementClass,
                    key[1],
                    data_set,
                    self.quantity,
                    parsing_kwargs=parsing_kwargs,
                    Einc_range=self.Einc_range,
                    Ex_range=self.Ex_range,
                    elastic_only=elastic_only,
                    vocal=vocal,
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
        self.MeasurementClass.plot(
            self.measurements,
            ax,
            offsets,
            self.data_symbol,
            f"${self.reaction.reaction_latex}$",
            log,
            draw_baseline,
            baseline_offset,
            xlim,
            fontsize,
            label_kwargs,
        )
