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

"""Temporal-primitive decomposition (CIDOC CRM style).

CIDOC CRM replaces Allen's relations by *temporal primitives*: individual,
attestable statements about the order of two interval boundaries, e.g. "the start
of X is before the start of Y". An Allen relation is then a conjunction of such
primitives. This module exposes that decomposition for the probabilistic algebra.

Each primitive compares one boundary of ``X`` to one boundary of ``Y`` through one
of the four boundary-difference variables used by the partition,

    A = a_Y - a_X   (start_X vs start_Y)
    B = b_Y - b_X   (end_X   vs end_Y)
    G = a_Y - b_X   (end_X   vs start_Y)   -- the frontal gap
    H = a_X - b_Y   (end_Y   vs start_X)   -- the rear gap

and is *three-valued* via the thresholded sign ``sgn_tau``:

    v >  tau  ->  first boundary strictly precedes the second   ('precedes')
    |v| <= tau ->  the two boundaries coincide within tau         ('coincides')
    v < -tau  ->  first boundary strictly follows the second     ('follows')

The middle ('coincides') state is the "points close within tau" predicate; it is
what gives the touching/aligning relations (``meets``, ``starts``, ``equals``, ...)
positive probability. Each Allen relation corresponds to a fixed sign pattern of
``(A, B, G, H)`` (:data:`CANONICAL_SIGNS`); the contact relations are exactly those
with at least one coincident primitive (one for the six touching relations, two for
``equals``).

Because the three states of a single primitive partition that boundary-difference
axis, each primitive induces a proper 3-way distribution whose probability is a sum
of Allen-leaf probabilities (a coarse predicate). A relation, by contrast, is the
*joint* probability over its primitives -- the primitives share boundaries and are
correlated, so a relation is never the product of its primitive marginals.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .intervals import IntervalGaussian
from .relations import RELATION_NAMES, relation_probabilities

__all__ = [
    "PRIMITIVES",
    "PRIMITIVE_POINTS",
    "CANONICAL_SIGNS",
    "STATE_NAMES",
    "primitive_probabilities",
    "decompose",
    "contacts",
    "format_decomposition",
]

#: The four boundary-difference variables, in order.
PRIMITIVES: Tuple[str, str, str, str] = ("A", "B", "G", "H")

#: For each primitive, the ordered pair of boundaries (first, second) it compares,
#: such that the primitive's ``+1`` state means ``first`` strictly precedes ``second``.
PRIMITIVE_POINTS: Dict[str, Tuple[str, str]] = {
    "A": ("start(X)", "start(Y)"),
    "B": ("end(X)", "end(Y)"),
    "G": ("end(X)", "start(Y)"),
    "H": ("end(Y)", "start(X)"),
}

#: Names of the three primitive states, keyed by sign.
STATE_NAMES: Dict[int, str] = {1: "precedes", 0: "coincides", -1: "follows"}

# Prototype mean arrangements (e_span, r_span) for each relation; the source of
# the canonical sign pattern. Cross-checked against ``classify_arrangement`` in the
# test suite so the two cannot drift.
_PROTOTYPE: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "before": ((0, 2), (3, 5)),
    "meets": ((0, 2), (2, 4)),
    "overlaps": ((0, 3), (2, 5)),
    "starts": ((0, 2), (0, 4)),
    "during": ((1, 3), (0, 5)),
    "finishes": ((2, 4), (0, 4)),
    "equals": ((0, 4), (0, 4)),
    "finished_by": ((0, 4), (2, 4)),
    "contains": ((0, 5), (1, 3)),
    "started_by": ((0, 4), (0, 2)),
    "overlapped_by": ((2, 5), (0, 3)),
    "met_by": ((2, 4), (0, 2)),
    "after": ((3, 5), (0, 2)),
}


def _sgn(v: float) -> int:
    return 0 if v == 0 else (1 if v > 0 else -1)


def _signs(e_span: Tuple[float, float], r_span: Tuple[float, float]) -> Tuple[int, int, int, int]:
    a_x, b_x = e_span
    a_y, b_y = r_span
    A, B, G, H = a_y - a_x, b_y - b_x, a_y - b_x, a_x - b_y
    return _sgn(A), _sgn(B), _sgn(G), _sgn(H)


#: Relation -> its fixed ``(A, B, G, H)`` sign pattern (``+1`` precedes, ``0``
#: coincides, ``-1`` follows).
CANONICAL_SIGNS: Dict[str, Tuple[int, int, int, int]] = {
    name: _signs(*_PROTOTYPE[name]) for name in RELATION_NAMES
}


def primitive_probabilities(
    X: IntervalGaussian, Y: IntervalGaussian, tau: float = 0.0
) -> Dict[str, Dict[str, float]]:
    """The four soft temporal primitives for ``X`` relative to ``Y``.

    Returns ``{primitive: {"precedes": p, "coincides": p, "follows": p}}`` for each
    of ``A, B, G, H``. Each inner distribution sums to 1: the probability of a
    primitive state is the sum of the probabilities of the Allen relations sharing
    that primitive sign (a coarse predicate over the taxonomy).
    """
    probs = relation_probabilities(X, Y, tau=tau)
    out: Dict[str, Dict[str, float]] = {}
    for k, prim in enumerate(PRIMITIVES):
        acc = {1: 0.0, 0: 0.0, -1: 0.0}
        for name in RELATION_NAMES:
            acc[CANONICAL_SIGNS[name][k]] += probs[name]
        out[prim] = {STATE_NAMES[s]: float(acc[s]) for s in (1, 0, -1)}
    return out


def decompose(relation: str) -> Dict[str, str]:
    """The primitive predicate for each of ``A, B, G, H`` defining ``relation``.

    Returns ``{primitive: predicate}`` where ``predicate`` reads like
    ``"end(X) coincides start(Y)"``.
    """
    if relation not in CANONICAL_SIGNS:
        raise KeyError(f"Unknown relation: {relation!r}")
    signs = CANONICAL_SIGNS[relation]
    out: Dict[str, str] = {}
    for k, prim in enumerate(PRIMITIVES):
        first, second = PRIMITIVE_POINTS[prim]
        out[prim] = f"{first} {STATE_NAMES[signs[k]]} {second}"
    return out


def contacts(relation: str) -> Tuple[str, ...]:
    """The primitives in the 'coincides' (points-close) state for ``relation``.

    Empty for the six strict relations, one element for the six touching relations
    (``meets``, ``starts``, ``finishes``, ``finished_by``, ``started_by``,
    ``met_by``), and two for ``equals``.
    """
    signs = CANONICAL_SIGNS[relation]
    return tuple(PRIMITIVES[k] for k in range(4) if signs[k] == 0)


def format_decomposition(relation: str) -> str:
    """A human-readable conjunction of the primitives defining ``relation``."""
    parts = decompose(relation)
    body = "  and  ".join(parts[p] for p in PRIMITIVES)
    return f"{relation}(X,Y)  <=>  {body}"
