r"""
Library tools for interactively curating data sets from EXFOR
"""

import numpy as np

from matplotlib import pyplot as plt

from .db import __EXFOR_DB__
from .exfor_tools import (
    ExforEntry,
    plot_angular_distributions,
    Reaction,
    quantity_matches,
)


def query_for_entries(reaction: Reaction, quantity: str, **kwargs):
    """
    Query for entries in the EXFOR database based on projectile, target,
    and quantity.

    reaction: the reaction to query
    quantity: The quantity to query.
    kwargs: Additional keyword arguments for entry parsing.

    Returns: A tuple containing successfully parsed entries and failed entries.
    """

    exfor_quantity = quantity_matches[quantity][0][0]
    entries = __EXFOR_DB__.query(
        quantity=exfor_quantity,
        target=reaction.exfor_symbol_target,
        projectile=reaction.exfor_symbol_projectile,
        # TODO handle case of reacxtion = "projectile,EL"
        # reaction=f"{reaction.exfor_symbol_projectile},{reaction.exfor_symbol_product}",
    ).keys()

    successfully_parsed_entries = {}
    failed_entries = {}

    for entry in entries:
        parsed_entry = ExforEntry(
            entry,
            reaction,
            quantity,
            **kwargs,
        )
        if len(parsed_entry.failed_parses) == 0 and len(parsed_entry.measurements) > 0:
            successfully_parsed_entries[entry] = parsed_entry
        elif len(parsed_entry.failed_parses) > 0:
            failed_entries[entry] = parsed_entry

    return successfully_parsed_entries, failed_entries


def find_unique_elements_with_tolerance(arr, tolerance):
    """
    Identify unique elements in an array within a specified tolerance.

    Parameters:
    arr (list or array-like): The input array to process.
    tolerance (float): The tolerance within which elements are considered
        identical.

    Returns:
    unique_elements (list):
    idx_sets (list): a list of sets, each entry corresponding to the indices
        to array that are within tolerance of the corresponding entry in
        unique_elements
    """
    unique_elements = []
    idx_sets = []

    for idx, value in enumerate(arr):
        found = False
        for i, unique_value in enumerate(unique_elements):
            if abs(value - unique_value) <= tolerance:
                idx_sets[i].add(idx)
                found = True
                break

        if not found:
            unique_elements.append(value)
            idx_sets.append({idx})

    return unique_elements, idx_sets


def categorize_measurement_list(measurements, min_num_pts=5, Einc_tol=0.1):
    """
    Categorize a list of measurements by unique incident energy.

    Parameters:
    measurements (list): A list of `AngularDistribution`s
    min_num_pts (int, optional): Minimum number of points for a valid
        measurement group. Default is 5.
    Einc_tol (float, optional): Tolerance for considering energies
        as identical. Default is 0.1.

    Returns:
    sorted_measurements (list): A list of lists, where each sublist contains
        measurements with similar incident energy.
    """
    energies = np.array([m.Einc for m in measurements])
    unique_energies, idx_sets = find_unique_elements_with_tolerance(energies, Einc_tol)
    unique_energies, idx_sets = zip(*sorted(zip(unique_energies, idx_sets)))

    sorted_measurements = []
    for idx_set in idx_sets:
        group = [measurements[idx] for idx in idx_set]
        sorted_measurements.append(group)

    return sorted_measurements


def categorize_measurements_by_energy(all_entries, min_num_pts=5, Einc_tol=0.1):
    r"""
    Given a dictionary form EXFOR entry number to ExforEntry, grabs all
    and sorts them by energy, concatenating ones that are at the same energy
    """
    measurements = []
    unique_subentries = set()
    for entry, data in all_entries.items():
        for measurement in data.measurements:
            if measurement.subentry not in unique_subentries:
                unique_subentries.add(measurement.subentry)
                if measurement.x.shape[0] > min_num_pts:
                    measurements.append(measurement)
    return categorize_measurement_list(
        measurements, min_num_pts=min_num_pts, Einc_tol=Einc_tol
    )


class ReactionEntries:
    r"""
    Collects all entries for a given reaction and quantity over
    a range of incident energies
    """

    def __init__(
        self,
        reaction: Reaction,
        quantity: str,
        vocal=False,
        **kwargs,
    ):
        self.reaction = reaction
        self.quantity = quantity
        self.settings = kwargs
        self.vocal = vocal
        kwargs["vocal"] = self.vocal

        self.entries, self.failed_parses = self.query(**kwargs)

    def query(self, **kwargs):
        if self.vocal:
            print("\n========================================================")
            print(f"Now parsing {self.quantity} for {self.reaction.pretty_string}")
            print("\n========================================================")
        entries, failed_parses = query_for_entries(
            reaction=self.reaction,
            quantity=self.quantity,
            **kwargs,
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Succesfully parsed {len(entries.keys())} entries")
            print(f"Failed to parse {len(failed_parses.keys())} entries:")
            print_failed_parses(failed_parses)
            print("\n========================================================")

        return entries, failed_parses

    def reattempt_parse(self, entry, parsing_kwargs):
        r"""
        Tries to re-parse a specific entry from failed_parses with specific
        parsing_kwargs. If it works, inserts it into self.entries and removes
        from self.failed_parses
        """
        failed_parse = self.failed_parses[entry]
        new_entry = ExforEntry(
            entry=failed_parse.entry,
            target=failed_parse.target,
            projectile=failed_parse.projectile,
            quantity=failed_parse.quantity,
            parsing_kwargs=parsing_kwargs,
            **self.settings,
        )
        if len(new_entry.failed_parses) == 0 and len(new_entry.measurements) > 0:
            self.entries[entry] = new_entry
            del self.failed_parses[entry]
        elif self.vocal:
            print("Reattempt parse failed")

    def plot(
        self,
        label_kwargs={
            "label_energy_err": False,
            "label_offset": False,
            "label_incident_energy": True,
            "label_excitation_energy": False,
            "label_exfor": True,
        },
        plot_kwargs={},
        n_per_plot=10,
        y_size=10,
    ):
        measurements_categorized = categorize_measurements_by_energy(self.entries)
        N = len(measurements_categorized)
        num_plots = N // n_per_plot
        left_over = N % n_per_plot
        if left_over > 0:
            num_plots += 1

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, y_size))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for i in range(num_plots):
            idx0 = i * n_per_plot
            if i == num_plots - 1:
                idxf = N
            else:
                idxf = (i + 1) * n_per_plot

            plot_angular_distributions(
                measurements_categorized[idx0:idxf],
                axes[i],
                data_symbol=list(self.entries.values())[0].data_symbol,
                rxn_label=list(self.entries.values())[0].reaction.pretty_string,
                label_kwargs=label_kwargs,
                **plot_kwargs,
            )
        return axes


class MulltiQuantityReactionData:
    r"""
    Given a single `Reaction` and a list of quantities, creates a corresponding
    list of `ReactionEntries` objects holding all the ExforEntry objects for
    that`Reaction` and the quantity of interest
    """

    def __init__(
        self,
        reaction: Reaction,
        quantities: list[str],
        settings: dict,
        vocal=False,
    ):
        self.reaction = reaction
        self.quantities = quantities
        self.settings = settings
        self.vocal = vocal
        self.data = {}

        for quantity in quantities:
            self.data[quantity] = ReactionEntries(
                self.reaction,
                quantity,
                vocal=self.vocal,
                **settings,
            )
        self.post_process_entries()

    def post_process_entries(self):
        r"""
        Handles duplicate entries, cross referencing and metadata.
        Should be called again after any failed parses are handled.
        """
        # handle duplicates between absolute and ratio to Rutherford
        # keeping only ratio
        if set(["dXS/dA", "dXS/dRuth"]).issubset(set(self.quantities)):
            remove_duplicates(
                *self.reaction.target,
                self.data["dXS/dRuth"].entries,
                self.data["dXS/dA"].entries,
                vocal=self.vocal,
            )

        self.data_by_entry = self.cross_reference_entries()
        self.num_data_pts, self.num_measurements = self.number_of_data_pts()

    def to_json(self):
        # TODO
        pass

    def cross_reference_entries(self):
        r"""Builds a dictionary from entry ID to ExforEntry from all of self.data"""
        unique_entries = {}
        for quantity, entries in self.data.items():
            for k, v in entries.entries.items():
                if k in unique_entries:
                    unique_entries[k].append(v)
                else:
                    unique_entries[k] = [v]
        return unique_entries

    def number_of_data_pts(self):
        """return a nested dict of the same structure as self.data but
        with the total number of of data points instead"""
        n_data_pts = {}
        n_measurements = {}
        for quantity, entries in self.data.items():
            n_measurements[quantity] = np.sum(
                [len(entry.measurements) for entry_id, entry in entries.entries.items()]
            )
            n_data_pts[quantity] = np.sum(
                [
                    np.sum([m.rows for m in entry.measurements])
                    for entry_id, entry in entries.entries.items()
                ]
            )

        return n_data_pts, n_measurements


def remove_duplicates(A, Z, entries_ppr, entries_pp, vocal=False):
    """remove subentries for (p,p) absolute cross sections if the Rutherford ratio is also reported"""
    all_dup = []
    for kr, vr in entries_ppr.items():
        for k, v in entries_pp.items():
            if kr == k:
                Eratio = [er.Einc for er in vr.measurements]
                nm = [e for e in v.measurements if e.Einc not in Eratio]
                if nm != v.measurements:
                    if vocal:
                        print(
                            f"({A},{Z}): found duplicates between (p,p) absolute and ratio to Rutherford in {k}"
                        )
                        print([x.Einc for x in v.measurements])
                        print([x.Einc for x in vr.measurements])

                    # replace (p,p) absolute with only non-duplicate measurements
                    v.measurements = nm
                    if nm == []:
                        all_dup.append(k)

    # in cases where all (p,p) absolute measurements are duplicates, remove whole entry
    for k in all_dup:
        del entries_pp[k]

    return entries_ppr, entries_pp


def print_failed_parses(failed_parses):
    for k, v in failed_parses.items():
        print(f"Entry: {k}")
        print(v.failed_parses[k][0], " : ", v.failed_parses[k][1])
