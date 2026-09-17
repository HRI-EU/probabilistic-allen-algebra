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

"""The Allen relation taxonomy as a probability calculus.

Coarse temporal predicates are unions of Allen leaves; because the leaves
partition the sample space (see ``latex_main/notes/partition_theorem.md``), each
coarse predicate's probability is the sum of its leaf probabilities.

This module encodes the taxonomy as a **strict partition tree**: the children of
every node are mutually exclusive and exhaustive within the parent, so their
probabilities sum to the parent's.  ``equals`` is its **own** leaf cell -- shared
by no other node -- which is exactly what makes the structure a tree rather than
a DAG.

Taxonomy::

    relation
    |-- separated
    |   |-- precede          : before, meets
    |   `-- follow           : met_by, after
    `-- non_separated
        |-- partial_overlap  : overlaps, overlapped_by
        |-- e_inside_r       : starts, during, finishes          (e subset r, strict)
        |-- r_inside_e       : started_by, contains, finished_by (r subset e, strict)
        `-- equals           : equals

The non-strict containment predicates ``e subseteq r`` and ``e supseteq r`` both
include ``equals`` and therefore overlap each other; they are **not** tree nodes.
They remain available as overlapping *views* (:data:`VIEWS`) computed by
:func:`coarse_predicates`, but they do not form a partition.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

__all__ = [
    "TREE",
    "VIEWS",
    "LEAVES",
    "leaves_of",
    "coarse_predicates",
    "refine",
]

#: The strict partition tree.  Each internal node maps to its ordered children
#: (sub-node names or leaf relation names).  The children of every node are
#: mutually exclusive and exhaustive within the parent.  A name is a *leaf*
#: (an Allen relation) iff it is not a key of this dict; note ``equals`` is a
#: bare leaf child of ``non_separated``, not a node.
TREE: Dict[str, List[str]] = {
    "relation": ["separated", "non_separated"],
    "separated": ["precede", "follow"],
    "non_separated": ["partial_overlap", "e_inside_r", "r_inside_e", "equals"],
    "precede": ["before", "meets"],
    "follow": ["met_by", "after"],
    "partial_overlap": ["overlaps", "overlapped_by"],
    "e_inside_r": ["starts", "during", "finishes"],
    "r_inside_e": ["started_by", "contains", "finished_by"],
}

#: Overlapping *views*: unions of leaves that cross the partition (they share
#: ``equals`` or span the separated / non_separated split).  Useful as coarse
#: queries, but -- unlike :data:`TREE` nodes -- they do **not** partition.
VIEWS: Dict[str, Tuple[str, ...]] = {
    "contained_in_r": ("starts", "during", "finishes", "equals"),  # e subseteq r (non-strict)
    "contains_r": ("started_by", "contains", "finished_by", "equals"),  # e supseteq r (non-strict)
    "outer_contact": ("meets", "met_by"),
    "inner_contact": ("starts", "started_by", "finishes", "finished_by", "equals"),
}


def leaves_of(node: str) -> List[str]:
    """The Allen leaves beneath ``node`` (``node`` itself if it is a leaf)."""
    children = TREE.get(node)
    if children is None:  # a leaf relation name
        return [node]
    out: List[str] = []
    for child in children:
        out.extend(leaves_of(child))
    return out


#: Every tree node mapped to the tuple of Allen leaves beneath it.
LEAVES: Dict[str, tuple] = {node: tuple(leaves_of(node)) for node in TREE}


def coarse_predicates(probs: Mapping[str, float]) -> Dict[str, float]:
    """Coarse-predicate probabilities from the thirteen leaf probabilities.

    Every :data:`TREE` node is reported (the children of any node sum to it, so
    e.g. ``separated + non_separated == 1``).  The overlapping :data:`VIEWS` are
    also reported for convenience; because ``contained_in_r`` and ``contains_r``
    share ``equals``, their union (containment in either direction) needs
    inclusion--exclusion and is reported separately as ``containment_either``.
    """
    out = {node: float(sum(probs[leaf] for leaf in leaves)) for node, leaves in LEAVES.items()}
    for name, leaves in VIEWS.items():
        out[name] = float(sum(probs[leaf] for leaf in leaves))
    out["containment_either"] = float(
        out["contained_in_r"] + out["contains_r"] - probs["equals"]
    )
    return out


def refine(probs: Mapping[str, float], parent: str) -> Dict[str, float]:
    """Conditional leaf distribution within a node, ``P(leaf | parent)``.

    ``parent`` may be any :data:`TREE` node or :data:`VIEWS` predicate.  Returns
    an empty dict if the parent has (numerically) zero probability.
    """
    if parent in TREE:
        leaves: tuple = LEAVES[parent]
    elif parent in VIEWS:
        leaves = VIEWS[parent]
    else:
        raise KeyError(f"Unknown taxonomy node: {parent!r}")
    total = sum(probs[leaf] for leaf in leaves)
    if total <= 0.0:
        return {}
    return {leaf: float(probs[leaf] / total) for leaf in leaves}
