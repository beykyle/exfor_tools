from .exfor_tools import ExforEntryAngularDistribution, query_for_entries


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


def reattempt_parse(
    failed_parse,
    parsing_kwargs,
    energy_range,
    min_num_pts,
    vocal=False,
):
    r"""Try to re-parse a specific entry with specific parsing options"""
    return ExforEntryAngularDistribution(
        entry=failed_parse.entry,
        target=failed_parse.target,
        projectile=failed_parse.projectile,
        quantity=failed_parse.quantity,
        Einc_range=energy_range,
        vocal=vocal,
        parsing_kwargs=parsing_kwargs,
        filter_kwargs={"filter_lab_angle": True, "min_num_pts": min_num_pts},
    )


def print_failed_parses(failed_parses):
    for k, v in failed_parses.items():
        print(f"Entry: {k}")
        print(v.failed_parses[k][0], " : ", v.failed_parses[k][1])


def query_elastic_data(target, energy_range, min_num_pts, vocal=False):
    """
    For a given target, queries exfor for all differential elastic cross sections for (n,n) and
    (p,p), including (p,p) cross sections as a ratio to Rutherford. If an entry reports both (p,p)
    absolute and ratio, keeps only the ratio.
    """

    if vocal:
        print("\n========================================================")
        print("Parsing (p,p) ...")
        print("========================================================")
    entries_pp, failed_parses_pp = query_for_entries(
        target=target,
        projectile=(1, 1),
        quantity="dXS/dA",
        Einc_range=energy_range,
        vocal=True,
        filter_kwargs={"filter_lab_angle": True, "min_num_pts": min_num_pts},
    )
    if vocal:
        print("\n========================================================")
        print(f"Succesfully parsed {len(entries_pp.keys())} entries for (p,p)")
        print(f"Failed to parse {len(failed_parses_pp.keys())} entries")
        print("========================================================\n\n")

    if vocal:
        print("\n========================================================")
        print("Parsing (p,p) ratio ...")
        print("========================================================")
    entries_ppr, failed_parses_ppr = query_for_entries(
        target=target,
        projectile=(1, 1),
        quantity="dXS/dRuth",
        Einc_range=energy_range,
        vocal=True,
        filter_kwargs={"filter_lab_angle": True, "min_num_pts": min_num_pts},
    )
    if vocal:
        print("\n========================================================")
        print(f"Succesfully parsed {len(entries_ppr.keys())} entries for (p,p) ratio")
        print(f"Failed to parse {len(failed_parses_ppr.keys())} entries")
        print("========================================================\n\n")

    if vocal:
        print("\n========================================================")
        print("Parsing (n,n)...")
        print("========================================================")
    entries_nn, failed_parses_nn = query_for_entries(
        target=target,
        projectile=(1, 0),
        quantity="dXS/dA",
        Einc_range=energy_range,
        vocal=True,
        filter_kwargs={"filter_lab_angle": True, "min_num_pts": min_num_pts},
    )
    if vocal:
        print("\n========================================================")
        print(f"Succesfully parsed {len(entries_nn.keys())} entries for (n,n)")
        print(f"Failed to parse {len(failed_parses_nn.keys())} entries")
        print("========================================================\n\n")

    remove_duplicates(*target, entries_ppr, entries_pp)
    remove_duplicates(*target, failed_parses_ppr, failed_parses_pp)

    return (
        (entries_pp, failed_parses_pp),
        (entries_ppr, failed_parses_ppr),
        (entries_nn, failed_parses_nn),
    )
