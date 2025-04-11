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
    get_symbol,
)


class ReactionAngularData:
    r"""
    Collects all entries for angular data (SDX, Ay, etc) for a given reaction
    over a range of energies
    """

    def __init__(
        self,
        target,
        projectile,
        quantity,
        **kwargs,
    ):
        self.target = target
        self.projectile = projectile
        self.quantity = quantity
        self.settings = kwargs
        self.residual = kwargs.get("residual", self.target)
        self.product = kwargs.get("product", self.projectile)
        self.special_rxn_type = kwargs.get("special_rxn_type", "")
        self.vocal = kwargs.get("vocal", False)

        self.symbol_target = get_symbol(*self.target)
        self.symbol_residual = get_symbol(*self.residual)
        self.symbol_projectile = get_symbol(*self.projectile)
        self.symbol_product = get_symbol(*self.product)

        if self.residual == self.target:
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},{self.symbol_product})_{{{self.special_rxn_type}}}$"
        else:
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},{self.symbol_product})_{{{self.special_rxn_type}}}${self.symbol_residual}"

        self.entries, self.failed_parses = self.query(**kwargs)

    def query(self, **kwargs):
        if self.vocal:
            print("\n========================================================")
            print(f"Now parsing {self.quantity} for {self.rxn}")
            print("\n========================================================")
        entries, failed_parses = query_for_entries(
            target=self.target,
            projectile=self.projectile,
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
        new_entry = ExforEntryAngularDistribution(
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
                rxn_label=list(self.entries.values())[0].rxn,
                label_kwargs=label_kwargs,
                **plot_kwargs,
            )
        return axes


def cross_reference_entries(data_by_target: list):
    # TODO make this a member function of DataCorpus
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
