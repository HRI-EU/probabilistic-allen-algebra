#
#  SPDX-License-Identifier: MIT
#
#  MIT License
#
#  Copyright (c) 2024-2025 Honda Research Institute Europe GmbH
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
#
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Tests for the temporal-primitive decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from paa.intervals import IntervalGaussian
from paa.primitives import (
    CANONICAL_SIGNS,
    PRIMITIVES,
    contacts,
    decompose,
    primitive_probabilities,
)
from paa.relations import CONVERSE, RELATION_NAMES, classify_arrangement, relation_probabilities

from _util import _sample_duration, _sample_midpoint

_PROTOTYPE = {
    "before": ((0, 2), (3, 5)), "meets": ((0, 2), (2, 4)), "overlaps": ((0, 3), (2, 5)),
    "starts": ((0, 2), (0, 4)), "during": ((1, 3), (0, 5)), "finishes": ((2, 4), (0, 4)),
    "equals": ((0, 4), (0, 4)), "finished_by": ((0, 4), (2, 4)), "contains": ((0, 5), (1, 3)),
    "started_by": ((0, 4), (0, 2)), "overlapped_by": ((2, 5), (0, 3)), "met_by": ((2, 4), (0, 2)),
    "after": ((3, 5), (0, 2)),
}

CONFIGS = [
    (IntervalGaussian(0.0, 0.7, 2.0, 0.4), IntervalGaussian(1.2, 0.7, 2.0, 0.4), 0.3),
    (IntervalGaussian(0.0, 1.0, 2.0, 0.3), IntervalGaussian(3.0, 1.0, 1.5, 0.4), 0.25),
    (IntervalGaussian(0.0, 0.4, 3.0, 0.5), IntervalGaussian(0.5, 0.6, 3.0, 0.5), 0.5),
]


def test_canonical_signs_match_classifier():
    """Each prototype, classified with a small tau, recovers its own relation."""
    for name, (e_s, r_s) in _PROTOTYPE.items():
        a_x, b_x = e_s
        a_y, b_y = r_s
        A, B, G, H = a_y - a_x, b_y - b_x, a_y - b_x, a_x - b_y
        assert classify_arrangement(A, B, G, H, tau=0.1) == name


def test_contacts_structure():
    """Strict relations have no contact, touching relations one, equals two."""
    assert contacts("before") == ()
    assert contacts("overlaps") == ()
    assert contacts("during") == ()
    assert contacts("meets") == ("G",)
    assert contacts("met_by") == ("H",)
    assert contacts("starts") == ("A",)
    assert contacts("finishes") == ("B",)
    assert contacts("equals") == ("A", "B")
    counts = {n: len(contacts(n)) for n in RELATION_NAMES}
    assert sum(counts.values()) == 6 * 1 + 1 * 2  # six single contacts + equals


def test_meets_decomposition():
    """meets = leading-edge precedence with end(X) coinciding start(Y)."""
    d = decompose("meets")
    assert d["A"] == "start(X) precedes start(Y)"
    assert d["B"] == "end(X) precedes end(Y)"
    assert d["G"] == "end(X) coincides start(Y)"
    # before/meets/overlaps share A, B, H and differ only in the G state.
    for prim in ("A", "B", "H"):
        assert CANONICAL_SIGNS["before"][PRIMITIVES.index(prim)] == \
               CANONICAL_SIGNS["meets"][PRIMITIVES.index(prim)] == \
               CANONICAL_SIGNS["overlaps"][PRIMITIVES.index(prim)]
    g = PRIMITIVES.index("G")
    assert (CANONICAL_SIGNS["before"][g], CANONICAL_SIGNS["meets"][g], CANONICAL_SIGNS["overlaps"][g]) == (1, 0, -1)


@pytest.mark.parametrize("e,r,tau", CONFIGS)
def test_primitive_states_partition(e, r, tau):
    """Each primitive's three states form a proper distribution (sum to 1)."""
    prim = primitive_probabilities(e, r, tau=tau)
    for key in PRIMITIVES:
        total = sum(prim[key].values())
        assert total == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("e,r,tau", CONFIGS)
def test_points_close_is_meets(e, r, tau):
    """The 'coincides' state of G maps to meets alone (only meets has sgn(G)=0)."""
    assert [n for n in RELATION_NAMES if CANONICAL_SIGNS[n][PRIMITIVES.index("G")] == 0] == ["meets"]
    prim = primitive_probabilities(e, r, tau=tau)
    p = relation_probabilities(e, r, tau=tau)
    # Equal up to the MVN-CDF noise floor (the two engine calls are independent).
    assert prim["G"]["coincides"] == pytest.approx(p["meets"], abs=2e-3)


@pytest.mark.parametrize("e,r,tau", CONFIGS)
def test_primitive_matches_monte_carlo(e, r, tau):
    """Analytic primitive marginals match a direct Monte-Carlo of the 1-D comparison."""
    n = 600_000
    rng = np.random.default_rng(1)
    te, tr = _sample_midpoint(e, n, rng), _sample_midpoint(r, n, rng)
    de, dr = _sample_duration(e, n, rng), _sample_duration(r, n, rng)
    se, fe, sr, fr = te - de / 2, te + de / 2, tr - dr / 2, tr + dr / 2
    diffs = {"A": sr - se, "B": fr - fe, "G": sr - fe, "H": se - fr}
    prim = primitive_probabilities(e, r, tau=tau)
    for key, v in diffs.items():
        mc = {
            "precedes": float((v > tau).mean()),
            "coincides": float((np.abs(v) <= tau).mean()),
            "follows": float((v < -tau).mean()),
        }
        for state in ("precedes", "coincides", "follows"):
            assert prim[key][state] == pytest.approx(mc[state], abs=4e-3)


def test_converse_mirrors_primitives():
    """Swapping X and Y negates A,B and swaps G<->H in the sign pattern."""
    for name in RELATION_NAMES:
        sA, sB, sG, sH = CANONICAL_SIGNS[name]
        assert CANONICAL_SIGNS[CONVERSE[name]] == (-sA, -sB, sH, sG)
