"""Charge-exchange data resolved to a level of the residual.

EXFOR entry O0669 is Gosset, Mayer and Escudie, Phys. Rev. C 14, 878 (1976):
the analyzing power of the quasi-elastic (p,n) reaction to the isobaric analog
state at 22.8 MeV on nine targets. Its subentries are written as, e.g.,
``49-TI-49(P,N)23-V-49,PAR,POL/DA,,ANA`` -- a polarization quantity carrying
the ``PAR`` qualifier, with a named residual.
"""

import pytest

from exfor_tools import ExforEntry
from exfor_tools.reaction import Reaction, is_match

# target, residual, analog-state excitation energy (Table I of the paper)
GOSSET_TARGETS = [
    ((49, 22), (49, 23), 5.36),
    ((56, 26), (56, 27), 3.51),
    ((64, 28), (64, 29), 6.70),
    ((70, 30), (70, 31), 8.12),
    ((90, 40), (90, 41), 5.03),
    ((96, 40), (96, 41), 11.07),
    ((117, 50), (117, 51), 11.18),
    ((165, 67), (165, 68), 15.49),
    ((208, 82), (208, 83), 15.33),
]


@pytest.mark.parametrize("target,residual,excitation", GOSSET_TARGETS)
def test_analyzing_power_to_analog_state(target, residual, excitation):
    """Every O0669 subentry parses as an Ay angular distribution."""
    reaction = Reaction(
        target=target, projectile=(1, 1), product=(1, 0), residual=residual
    )
    entry = ExforEntry("O0669", reaction, quantity="Ay")

    assert entry.failed_parses == {}
    assert len(entry.measurements) == 1

    measurement = entry.measurements[0]
    assert measurement.Einc == pytest.approx(22.8)
    assert measurement.Ex == pytest.approx(excitation)
    assert measurement.x_units == "CM-degrees"
    assert measurement.y_units == "no-dim"
    assert measurement.rows == len(measurement.y)
    assert measurement.rows >= 4


def test_process_form_accepts_a_residual():
    """(p,n) leaving a named residual is expressible in the process form."""
    reaction = Reaction(
        target=(56, 26), projectile=(1, 1), process="n", residual=(56, 27)
    )
    assert reaction.reaction_string == "Fe-56(P,n)Co-56"
    assert str(reaction) == "Fe-56(P,n)Co-56"


def test_unnamed_residual_does_not_match_a_named_one():
    """A product-form reaction with no residual is a mismatch, not an error."""
    import exfor_tools.db as db

    data_sets = db.__EXFOR_DB__.retrieve(ENTRY="O0669")["O0669"].getDataSets()
    subentry = next(
        data_set
        for key, data_set in data_sets.items()
        if key[1] == "O0669003"  # 56Fe(p,n)56Co
    )

    reaction = Reaction(target=(56, 26), projectile=(1, 1), product=(1, 0))
    assert is_match(reaction, subentry) is False
