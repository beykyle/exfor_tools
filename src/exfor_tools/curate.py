import numpy as np

from matplotlib import pyplot as plt

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
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},"
            f"{self.symbol_product})_{{{self.special_rxn_type}}}$"
        else:
            self.rxn = f"{self.symbol_target}$({self.symbol_projectile},"
            f"{self.symbol_product})_{{{self.special_rxn_type}}}${self.symbol_residual}"

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


class AngularDataCorpus:
    r"""Queries, parses and stores differential cross sections and
    analyzing powers for multiple reactions from EXFOR, storing them
    as nested dicts from target -> projectile -> quantity
    """

    def __init__(
        self,
        targets: list[tuple],
        projectiles: list[tuple],
        quantities: list[str],
        settings: list[dict],
        vocal=False,
    ):
        self.targets = targets
        self.projectiles = projectiles
        self.quantities = quantities
        self.settings = settings
        self.vocal = vocal
        self.data = {}

        for target, projectile, quantity, kwargs in zip(
            targets, projectiles, quantities, settings
        ):
            kwargs["vocal"] = vocal
            entries = ReactionAngularData(
                target,
                projectile,
                quantity,
                **kwargs,
            )

            if target in self.data:
                if projectile in self.data[target]:
                    self.data[target][projectile][quantity] = entries
                else:
                    self.data[target][projectile] = {quantity: entries}
            else:
                self.data[target] = {projectile: {quantity: entries}}

        # handle duplicates between absolute angular cross sections and ratio to Rutherford
        for target in self.data.keys():
            for projectile in self.data[target].keys():
                if projectile[1] > 0:
                    quantities = self.data[target][projectile]
                    if "dXS/dA" in quantities and "dXS/dRuth" in quantities:
                        if (
                            quantities["dXS/dA"].settings
                            == quantities["dXS/dRuth"].settings
                        ):
                            remove_duplicates(
                                *target,
                                quantities["dXS/dRuth"].entries,
                                quantities["dXS/dA"].entries,
                            )
                            remove_duplicates(
                                *target,
                                quantities["dXS/dRuth"].failed_parses,
                                quantities["dXS/dA"].failed_parses,
                            )

        self.data_by_entry = self.cross_reference_entries()
        self.num_data_pts, self.num_measurements = self.number_of_data_pts()

    def to_json(self):
        # TODO
        pass

    def cross_reference_entries(self):
        r"""Builds a dictionary from entry ID to ExforEntryAngularDistribution from all of self.data"""
        entries = {}
        for target in self.data.keys():
            for projectile in self.data[target].keys():
                for quantity, data in self.data[target][projectile].items():
                    for k, v in data.entries.items():
                        if k in entries:
                            entries[k].append(v)
                        else:
                            entries[k] = [v]

        return entries

    def number_of_data_pts(self):
        """return a nested dict of the same structure as self.data but with the total number of of data points instead"""
        n_data_pts = {}
        n_measurements = {}
        for target in self.data.keys():
            n_data_pts[target] = {}
            n_measurements[target] = {}
            for projectile in self.data[target].keys():
                n_data_pts[target][projectile] = {}
                n_measurements[target][projectile] = {}
                for quantity, data in self.data[target][projectile].items():
                    n_measurements[target][projectile][quantity] = np.sum(
                        [len(entry.measurements) for entry_id, entry in data.entries.items()]
                    )
                    n_data_pts[target][projectile][quantity] = np.sum(
                        [np.sum([m.rows for m in entry.measurements]) for entry_id, entry in data.entries.items()]
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
