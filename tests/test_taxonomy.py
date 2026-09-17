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

"""The taxonomy is a strict partition tree: children sum to their parent.

These checks are what keep ``paa.taxonomy.TREE`` (and the decoder's view of it)
from drifting -- the historical bug being that ``equals`` was shared between both
containment directions, making the structure a DAG so sibling masses did not sum
to the parent.
"""

from __future__ import annotations

from pytest import approx

from paa import IntervalGaussian, relation_probabilities
from paa.relations import RELATION_NAMES
from paa.taxonomy import TREE, VIEWS, coarse_predicates, leaves_of, refine

# A spread of configurations so the partition identities are exercised on
# non-degenerate leaf distributions, not just one arrangement.
_CONFIGS = [
    (IntervalGaussian(0.0, 1.0, 2.0, 0.3), IntervalGaussian(3.0, 1.0, 1.5, 0.4)),
    (IntervalGaussian(0.0, 0.2, 1.0, 0.1), IntervalGaussian(0.0, 0.2, 5.0, 0.1)),
    (IntervalGaussian(0.0, 0.5, 2.0, 0.2), IntervalGaussian(2.0, 0.5, 2.0, 0.2)),
    (IntervalGaussian(1.0, 0.8, 3.0, 0.5), IntervalGaussian(0.5, 0.8, 3.0, 0.5)),
]
_TAUS = [0.0, 0.1, 0.5]


def test_leaves_cover_the_thirteen_exactly() -> None:
    leaves = leaves_of("relation")
    assert sorted(leaves) == sorted(RELATION_NAMES)
    assert len(leaves) == len(set(leaves)) == 13  # no duplication -> a tree, not a DAG


def test_children_are_disjoint() -> None:
    """Siblings share no leaf (the property `equals`-sharing used to violate)."""
    for node, children in TREE.items():
        seen: set = set()
        for child in children:
            cl = set(leaves_of(child))
            assert seen.isdisjoint(cl), (node, child, seen & cl)
            seen |= cl


def test_every_node_children_partition_the_parent() -> None:
    """For each TREE node, child masses sum to the node's mass, at every config."""
    for e, r in _CONFIGS:
        for tau in _TAUS:
            probs = relation_probabilities(e, r, tau=tau)
            cp = coarse_predicates(probs)
            for node, children in TREE.items():
                # A child is either a TREE node (in cp) or a bare leaf (in probs).
                child_mass = sum(cp[c] if c in cp else probs[c] for c in children)
                assert child_mass == approx(cp[node]), (node, tau)


def test_root_sums_to_one() -> None:
    for e, r in _CONFIGS:
        for tau in _TAUS:
            probs = relation_probabilities(e, r, tau=tau)
            assert coarse_predicates(probs)["relation"] == approx(1.0, abs=2e-3)


def test_refine_conditional_sums_to_one() -> None:
    e, r = _CONFIGS[1]
    probs = relation_probabilities(e, r, tau=0.25)
    for node in list(TREE) + list(VIEWS):
        dist = refine(probs, node)
        if dist:  # parent has positive mass
            assert sum(dist.values()) == approx(1.0)


def test_views_are_genuinely_overlapping() -> None:
    # The two containment views share exactly `equals`; that is why they are
    # views, not tree nodes.
    shared = set(VIEWS["contained_in_r"]) & set(VIEWS["contains_r"])
    assert shared == {"equals"}
