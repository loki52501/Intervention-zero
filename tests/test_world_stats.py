from iz.types import Outcome, OutcomeStatus
from iz.world.stats import tfmr, wilson


def test_wilson_edges():
    assert wilson(0, 10)[0] == 0.0
    assert wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_55_100():
    lo, hi = wilson(55, 100)
    assert abs(lo - 0.452) < 0.005 and abs(hi - 0.644) < 0.005


def _o(s):
    return Outcome(status=s, reason="x")


def test_tfmr():
    S, T = OutcomeStatus.SUCCESS, OutcomeStatus.TAKEOVER
    assert tfmr([]) == 0.0
    assert tfmr([[], [_o(S)], [_o(S), _o(T)], [_o(T)]]) == 0.5
