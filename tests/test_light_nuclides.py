"""Symbols for light nuclides and heavy-ion projectiles in EXFOR queries.

EXFOR's light-particle codes (N, P, D, T, HE3, A) are legal only in the incident
and outgoing particle slots of a reaction string. The target and residual slots
are always nuclides, so n+p elastic scattering is ``1-H-1(N,EL)1-H-1`` and the
x4i3 index stores its target as ``H-1``, never ``P``. Heavy-ion projectiles are
stored as ``Z-SYM-A`` (``6-C-12``), and 3He as ``HE3``.

See https://github.com/beykyle/exfor_tools/issues/17.
"""

import pytest

from exfor_tools import ExforEntry
from exfor_tools.reaction import (
    Reaction,
    get_exfor_nuclide_symbol,
    get_exfor_particle_symbol,
    query_for_reaction,
)


@pytest.mark.parametrize(
    "A,Z,nuclide,particle",
    [
        (1, 1, "H-1", "P"),
        (2, 1, "H-2", "D"),
        (3, 1, "H-3", "T"),
        (3, 2, "He-3", "HE3"),
        (4, 2, "He-4", "A"),
        (12, 6, "C-12", "6-C-12"),
        (208, 82, "Pb-208", "82-Pb-208"),
    ],
)
def test_symbol_depends_on_slot(A, Z, nuclide, particle):
    assert get_exfor_nuclide_symbol(A, Z) == nuclide
    assert get_exfor_particle_symbol(A, Z) == particle


def test_natural_target():
    assert get_exfor_nuclide_symbol(0, 26) == "Fe-0"


def test_neutron_symbols():
    assert get_exfor_particle_symbol(1, 0) == "N"
    with pytest.raises(ValueError):
        get_exfor_nuclide_symbol(1, 0)


@pytest.mark.parametrize(
    "reaction,string",
    [
        (Reaction(target=(1, 1), projectile=(1, 0), process="EL"), "H-1(N,el)"),
        (Reaction(target=(4, 2), projectile=(4, 2), process="EL"), "He-4(A,el)"),
        (
            Reaction(target=(3, 2), projectile=(1, 0), product=(1, 1), residual=(3, 1)),
            "He-3(N,P)H-3",
        ),
        (Reaction(target=(58, 28), projectile=(3, 2), process="EL"), "Ni-58(HE3,el)"),
    ],
)
def test_reaction_string(reaction, string):
    assert reaction.reaction_string == string


# Entries carrying both DATA-ERR and ERR-1 need their error attribution passed
# explicitly, read from ERR-ANALYSIS:
#   A0192003: DATA-ERR rms statistical; ERR-1 normalization (2%)
#   A1003005: DATA-ERR graph bars and digitizing; ERR-1 systematic (10%)
#   A1146003: DATA-ERR digitizing; ERR-1 all sources but counting statistics (4%)
STAT_DATA_ERR_SYS_ERR_1 = {
    "statistical_err_labels": ["DATA-ERR"],
    "systematic_err_labels": ["ERR-1"],
}

# (target, projectile, quantity, an entry that must be returned, parsing kwargs).
# Entries are pinned against x4i3_X4-2023-04-29; we assert membership rather than
# counts so that database updates do not break the tests.
QUERY_CASES = [
    pytest.param((1, 1), (1, 0), "dXS/dA", "10036", {}, id="n+p"),
    pytest.param((1, 1), (1, 1), "dXS/dA", "A0192", STAT_DATA_ERR_SYS_ERR_1, id="p+p"),
    pytest.param(
        (2, 1), (2, 1), "dXS/dRuth", "A1003", STAT_DATA_ERR_SYS_ERR_1, id="d+d"
    ),
    pytest.param(
        (4, 2), (4, 2), "dXS/dA", "A1146", STAT_DATA_ERR_SYS_ERR_1, id="a+4He"
    ),
    pytest.param((3, 2), (1, 1), "dXS/dA", "A1022", {}, id="p+3He"),
    pytest.param((4, 2), (1, 1), "dXS/dA", "A0925", {}, id="p+4He"),
    pytest.param((58, 28), (3, 2), "dXS/dRuth", "C1976", {}, id="3He+58Ni"),
    pytest.param((12, 6), (12, 6), "dXS/dA", "A0842", {}, id="12C+12C"),
]


@pytest.mark.parametrize(
    "target,projectile,quantity,entry_id,parsing_kwargs", QUERY_CASES
)
def test_query_finds_light_and_heavy_ion_systems(
    target, projectile, quantity, entry_id, parsing_kwargs
):
    reaction = Reaction(target=target, projectile=projectile, process="EL")
    assert entry_id in query_for_reaction(reaction, quantity)

    entry = ExforEntry(
        entry_id, reaction, quantity=quantity, parsing_kwargs=parsing_kwargs
    )
    assert entry.failed_parses == {}
    assert len(entry.measurements) > 0


# Heavy-target queries already worked before the symbol split; entries they
# returned before it must still be returned.
HEAVY_TARGET_ENTRIES = {
    ((208, 82), (1, 1), "dXS/dA"): ["C0071", "C0081", "C0260", "T0101"],
    ((56, 26), (1, 0), "dXS/dA"): ["10037", "10958", "11276", "40417"],
    ((48, 20), (1, 1), "Ay"): ["C0154", "C0156", "C0586", "O0239"],
}


@pytest.mark.parametrize("key", HEAVY_TARGET_ENTRIES, ids=str)
def test_heavy_target_queries_unchanged(key):
    target, projectile, quantity = key
    reaction = Reaction(target=target, projectile=projectile, process="EL")
    entries = query_for_reaction(reaction, quantity)
    assert set(HEAVY_TARGET_ENTRIES[key]) <= set(entries)
