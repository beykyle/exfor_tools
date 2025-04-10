from pathlib import Path
import pickle

from matplotlib import pyplot as plt

from .exfor_tools import (
    ExforEntryAngularDistribution,
    query_for_entries,
    plot_angular_distributions,
    categorize_measurements_by_energy,
)


class ElasticDifferentialData:
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

        # query and parse EXFOR
        self.nn, self.failed_parses_nn = self.query_nn()
        self.pp_abs, self.failed_parses_pp_abs = self.query_nn()
        self.pp_ratio, self.failed_parses_pp_ratio = self.query_nn()

        # remove (p,p) absolute data sets that are duplicate to (p,p) ratio data sets
        remove_duplicates(*self.target, self.pp_ratio, self.pp_abs)
        remove_duplicates(
            *self.target, self.failed_parses_pp_ratio, self.failed_parses_pp_abs
        )

    def query_pp_absolute(self):
        if self.vocal:
            print("\n========================================================")
            print("Parsing (p,p) ...")
            print("========================================================")
        entries, failed_parses = query_for_entries(
            target=self.target,
            projectile=(1, 1),
            quantity="dXS/dA",
            Einc_range=self.energy_range,
            vocal=self.vocal,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Succesfully parsed {len(entries.keys())} entries for (p,p)")
            print(f"Failed to parse {len(failed_parses.keys())} entries")
            print("========================================================\n\n")
            print_failed_parses(failed_parses)

        return entries, failed_parses

    def query_pp_ratio(self):
        if self.vocal:
            print("\n========================================================")
            print("Parsing (p,p) ratio ...")
            print("========================================================")
        entries, failed_parses = query_for_entries(
            target=self.target,
            projectile=(1, 1),
            quantity="dXS/dRuth",
            Einc_range=self.energy_range,
            vocal=self.vocal,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Succesfully parsed {len(entries.keys())} entries for (p,p) ratio")
            print(f"Failed to parse {len(failed_parses.keys())} entries")
            print("========================================================\n\n")
            print_failed_parses(failed_parses)

        return entries, failed_parses

    def query_nn(self):
        if self.vocal:
            print("\n========================================================")
            print("Parsing (n,n)...")
            print("========================================================")
        entries, failed_parses = query_for_entries(
            target=self.target,
            projectile=(1, 0),
            quantity="dXS/dA",
            Einc_range=self.energy_range,
            vocal=True,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
        )
        if self.vocal:
            print("\n========================================================")
            print(f"Succesfully parsed {len(entries.keys())} entries for (n,n)")
            print(f"Failed to parse {len(failed_parses.keys())} entries")
            print("========================================================\n\n")
            print_failed_parses(failed_parses)
        return entries, failed_parses

    def reattempt_parse(self, failed_parse, parsing_kwargs):
        r"""Try to re-parse a specific entry with specific parsing options"""
        return ExforEntryAngularDistribution(
            entry=failed_parse.entry,
            target=failed_parse.target,
            projectile=failed_parse.projectile,
            quantity=failed_parse.quantity,
            Einc_range=self.energy_range,
            vocal=self.vocal,
            parsing_kwargs=parsing_kwargs,
            filter_kwargs={"filter_lab_angle": True, "min_num_pts": self.min_num_pts},
        )

    def plot(
        self,
        entries,
        n_per_plot=10,
        y_size=10,
        log=False,
        draw_baseline=False,
        **label_kwargs,
    ):
        measurements_categorized = categorize_measurements_by_energy(entries)
        N = len(measurements_categorized)
        num_plots = N // n_per_plot
        left_over = N % n_per_plot
        if left_over > 0:
            num_plots += 1

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, y_size))
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
                offsets=100,
                data_symbol=list(entries.values())[0].data_symbol,
                rxn_label=list(entries.values())[0].rxn,
                label_kwargs=label_kwargs,
                log=log,
                draw_baseline=draw_baseline,
            )
        return axes

    def write(self, fpath: Path):
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def read(fpath: Path):
        with open(fpath, "rb") as f:
            return pickle.load(f)


def cross_reference_entries(data_by_target: list[ElasticDifferentialData]):
    entries = {}
    for data in data_by_target:
        for k, v in data.nn_elastic.items():
            if k in entries:
                entries[k].append(v)
            else:
                entries[k] = [v]

        for k, v in data.pp_elastic_ratio.items():
            if k in entries:
                entries[k].append(v)
            else:
                entries[k] = [v]

        for k, v in data.pp_elastic_abs.items():
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
