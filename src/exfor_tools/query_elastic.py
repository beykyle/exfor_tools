from pathlib import Path
import pickle
import numpy as np

from matplotlib import pyplot as plt
from periodictable import elements

from .exfor_tools import (
    ExforEntryAngularDistribution,
    query_for_entries,
    plot_angular_distributions,
    categorize_measurements_by_energy,
)


class DifferentialData:
    def __init__(
        self, target, projectile, quantity, energy_range, min_num_pts, vocal=False
    ):
        self.target = target
        self.projectile = projectile
        self.quantity = quantity
        self.energy_range = energy_range
        self.min_num_pts = min_num_pts
        self.vocal = vocal

        self.entries, self.failed_parses = self.query()

    def query(self):
        entries, failed_parses = query_for_entries(
            target=self.target,
            projectile=self.projectile,
            quantity=self.quantity,
            Einc_range=self.energy_range,
            vocal=self.vocal,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
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
        new_entry = ExforEntryAngularDistribution(
            entry=failed_parse.entry,
            target=failed_parse.target,
            projectile=failed_parse.projectile,
            quantity=failed_parse.quantity,
            Einc_range=self.energy_range,
            vocal=self.vocal,
            parsing_kwargs=parsing_kwargs,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
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
                rxn_label=list(self.entries.values())[0].rxn,
                label_kwargs=label_kwargs,
                **plot_kwargs,
            )
        return axes


class TargetData:
    r"""
    Queries EXFOR for all of the (n,n) and (p,p) (both absolute and ratio to Rutherford)
    subentries for a given target, within energy_range and including at least min_num_pts
    angular points, storing them as a dictionary from entry ID to ExforEntryAngularDistribution
    """

    # TODO add analyzing powers
    def __init__(self, target, energy_range, min_num_pts, vocal=False):
        self.target = target
        self.energy_range = energy_range
        self.min_num_pts = min_num_pts
        self.vocal = vocal

        A, Z = target
        # query and parse EXFOR
        if self.vocal:
            print("\n========================================================")
            print(f"Parsing: {A}-{elements[Z]}(n,n)")
            print("\n========================================================")

        self.nn = DifferentialData(
            self.target,
            (1, 0),
            "dXS/dA",
            self.energy_range,
            self.min_num_pts,
            self.vocal,
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Parsing: {A}-{elements[Z]}(p,p) (absolute)")
            print("\n========================================================")
        self.pp_abs = DifferentialData(
            self.target,
            (1, 1),
            "dXS/dA",
            self.energy_range,
            self.min_num_pts,
            self.vocal,
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Parsing: {A}-{elements[Z]}(p,p) (ratio to Rutherford)")
            print("\n========================================================")
        self.pp_ratio = DifferentialData(
            self.target,
            (1, 1),
            "dXS/dRuth",
            self.energy_range,
            self.min_num_pts,
            self.vocal,
        )

        # remove (p,p) absolute data sets that are duplicate to (p,p) ratio data sets
        remove_duplicates(*self.target, self.pp_ratio.entries, self.pp_abs.entries)
        remove_duplicates(
            *self.target, self.pp_ratio.failed_parses, self.pp_abs.failed_parses
        )

    def write(self, fpath: Path):
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read(fpath: Path):
        with open(fpath, "rb") as f:
            return pickle.load(f)


def cross_reference_entries(data_by_target: list[TargetData]):
    entries = {}
    for target, data in data_by_target.items():
        for k, v in data.nn.entries.items():
            if k in entries:
                entries[k].append(v)
            else:
                entries[k] = [v]

        for k, v in data.pp_ratio.entries.items():
            if k in entries:
                entries[k].append(v)
            else:
                entries[k] = [v]

        for k, v in data.pp_abs.entries.items():
            if k in entries:
                entries[k].append(v)
            else:
                entries[k] = [v]
    return entries


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
