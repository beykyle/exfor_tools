r"""
Library tools for interactively curating data sets from EXFOR
"""

import numpy as np

from matplotlib import pyplot as plt

from .exfor_entry import ExforEntry
from . import reaction as rxn
from .distribution import AngularDistribution
from .parsing import quantity_matches, quantity_symbols


# 11848 has an issue https://github.com/afedynitch/x4i3/issues/11
def query_for_entries(
    reaction: rxn.Reaction, quantity: str, disclude=["11848"], **kwargs
):
    """
    Query for entries in the EXFOR database based on projectile, target,
    and quantity.

    reaction: the reaction to query
    quantity: The quantity to query.
    kwargs: Additional keyword arguments for entry parsing.

    Returns: A tuple containing successfully parsed entries and failed entries.
    """

    entries = rxn.query_for_reaction(reaction, quantity)
    successfully_parsed_entries = {}
    failed_entries = {}

    for entry in entries:
        if entry not in disclude:
            try:
                parsed_entry = ExforEntry(
                    entry,
                    reaction,
                    quantity,
                    **kwargs,
                )
            except Exception as e:
                new_exception = type(e)(f"Error while parsing Entry: {entry}")
                raise new_exception from e
            if (
                len(parsed_entry.failed_parses) == 0
                and len(parsed_entry.measurements) > 0
            ):
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
    unique_energies, idx_sets = zip(
        *sorted(zip(unique_energies, idx_sets), reverse=True)
    )

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
    for entry, data in all_entries.items():
        for measurement in data.measurements:
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
        reaction: rxn.Reaction,
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
            print(f"Now parsing {self.quantity} for {self.reaction.reaction_string}")
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
            # print_failed_parses(failed_parses)
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
            reaction=failed_parse.reaction,
            quantity=failed_parse.quantity,
            parsing_kwargs=parsing_kwargs,
            **self.settings,
        )
        if len(new_entry.failed_parses) == 0 and len(new_entry.measurements) > 0:
            self.entries[entry] = new_entry
            del self.failed_parses[entry]
        elif self.vocal:
            print("Reattempt parse failed")

    def print_failed_parses(self):
        print_failed_parses(self.failed_parses)

    def plot(self, **kwargs):
        measurements_categorized = categorize_measurements_by_energy(self.entries)
        return plot_measurements(
            self.quantity,
            self.reaction,
            measurements_categorized,
            **kwargs,
        )


class MultiQuantityReactionData:
    r"""
    Given a single `Reaction` and a list of quantities, creates a corresponding
    list of `ReactionEntries` objects holding all the ExforEntry objects for
    that`Reaction` and the quantity of interest
    """

    def __init__(
        self,
        reaction: rxn.Reaction,
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
            n_measurements[quantity] = int(
                np.sum(
                    [
                        len(entry.measurements)
                        for entry_id, entry in entries.entries.items()
                    ]
                )
            )
            n_data_pts[quantity] = int(
                np.sum(
                    [
                        np.sum([m.rows for m in entry.measurements])
                        for entry_id, entry in entries.entries.items()
                    ]
                )
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


def build_measurement_list(
    quantity: str,
    data: dict[tuple[int, int], MultiQuantityReactionData],
    allowed_measurement_quantities=None,
):
    r"""
    Builds a list of measurements from the data dictionary.

    Parameters:
    quantity: str
        The quantity of interest.
    data: dict
        A dictionary mapping target tuples to MultiQuantityReactionData objects.
    allowed_measurement_quantities: list, optional
        A list of allowed measurement
    """
    if allowed_measurement_quantities is None:
        allowed_measurement_quantities = [quantity]

    measurements = []
    for target, data_set in data.items():
        for q in allowed_measurement_quantities:
            for entry_id, entry in data_set.data[q].entries.items():
                for measurement in entry.measurements:
                    measurements.append((entry.reaction, measurement))

    return measurements


def build_measurement_dict(
    quantity: str,
    data: dict[tuple[int, int], MultiQuantityReactionData],
    allowed_measurement_quantities=None,
):
    r"""
    Builds a dictionary of measurements categorized by entry ID.
    The keys are entry IDs, and the values are lists of tuples
    containing the reaction and measurement.

    Parameters:
    quantity: str
        The quantity of interest.
    data: dict
        A dictionary mapping target tuples to MultiQuantityReactionData objects.
    allowed_measurement_quantities: list, optional
        A list of allowed measurement
    """
    if allowed_measurement_quantities is None:
        allowed_measurement_quantities = [quantity]

    measurements = {}
    for target, data_set in data.items():
        for q in allowed_measurement_quantities:
            for entry_id, entry in data_set.data[q].entries.items():
                if entry_id not in measurements:
                    measurements[entry_id] = []
                for measurement in entry.measurements:
                    measurements[entry_id].append((entry.reaction, measurement))

    return measurements


def cross_reference_entry_systematic_err(
    all_data: list[MultiQuantityReactionData],
):
    all_data_by_entry = {}
    for data in all_data:
        for entry_id, entries in data.data_by_entry.items():
            if entry_id in all_data_by_entry:
                all_data_by_entry[entry_id].extend(entries)
            else:
                all_data_by_entry[entry_id] = entries

    sys_norm_err = {}
    for entry_id, data_sets in all_data_by_entry.items():
        norm_errs = []
        for data_set in data_sets:
            for m in data_set.measurements:
                norm_errs.append(m.systematic_norm_err)
            sys_norm_err[entry_id] = norm_errs
    return all_data_by_entry, sys_norm_err


def print_failed_parses(failed_parses):
    for k, v in failed_parses.items():
        print(f"Entry: {k}")
        print(v.failed_parses[k][0], " : ", v.failed_parses[k][1])
        print(v.err_analysis)
        print(v.subentry_err_analysis[v.failed_parses[k][0]])


def plot_measurements(
    quantity,
    reaction,
    measurements_categorized,
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
    latex_title = reaction.reaction_latex
    exfor_quantity = tuple(quantity_matches[quantity][0])
    quantity_symbol = quantity_symbols[exfor_quantity]

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

        AngularDistribution.plot(
            measurements_categorized[idx0:idxf],
            axes[i],
            data_symbol=quantity_symbol,
            rxn_label=f"${latex_title}$",
            label_kwargs=label_kwargs,
            **plot_kwargs,
        )
    return axes
