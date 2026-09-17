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

"""Selecting a "best fitting" relation for a configuration.

Three distinct, well-defined notions are provided, because they answer different
questions (see the module docstring of :mod:`paa.taxonomy` for the taxonomy):

- :func:`most_probable_relation` -- flat ``argmax P(R)``.  Simple, but **biased by
  relation measure**: contact relations occupy thin tolerance bands (width ``~tau``)
  and so rarely win even when appropriate.  Best used with ``restrict`` to compare
  siblings of comparable measure (e.g. the six thick relations).
- :func:`hierarchical_decode` -- top-down MAP over the relation taxonomy: choose the
  most probable coarse family first (robust, large-measure), then refine.  This is the
  principled way to "argmax", comparing comparable alternatives at each level.
- :func:`map_relation` -- the relation of the most probable single arrangement (the mode
  of the latent Gaussian).  Unbiased by band width; the natural single descriptor.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union

from .intervals import IntervalGaussian
from .relations import RELATION_NAMES, classify_arrangement
from .taxonomy import TREE as _TAXONOMY

__all__ = ["most_probable_relation", "hierarchical_decode", "map_relation"]

# Decoding tree: (name, children) where children is either a list of leaf relation
# names or a list of sub-nodes.  The children of every node partition their parent.
# Built from the single source of truth in :mod:`paa.taxonomy` so the two can never
# drift.  Leaf children that sit among sub-nodes are wrapped as ``(leaf, [leaf])``
# to keep every node uniform.
_Node = Tuple[str, Union[List[str], List["_Node"]]]


def _build(name: str) -> _Node:
    children = _TAXONOMY[name]
    if all(child not in _TAXONOMY for child in children):  # leaf level
        return (name, list(children))
    return (name, [_build(c) if c in _TAXONOMY else (c, [c]) for c in children])


_TREE: _Node = _build("relation")


def most_probable_relation(
    probs: Mapping[str, float], restrict: Optional[Sequence[str]] = None
) -> Tuple[str, float]:
    """Flat ``argmax P(R)`` over the relations (optionally a restricted subset).

    Note the measure bias: prefer ``restrict`` (e.g. ``THICK_RELATIONS`` or a single
    taxonomy level) when comparing relations that should be weighed on equal footing.
    """
    names = list(restrict) if restrict is not None else list(RELATION_NAMES)
    best = max(names, key=lambda n: probs[n])
    return best, float(probs[best])


def _is_leaf_level(children: Sequence) -> bool:
    return bool(children) and isinstance(children[0], str)


def _leaves(node: _Node) -> List[str]:
    _, children = node
    if _is_leaf_level(children):
        return list(children)  # type: ignore[arg-type]
    out: List[str] = []
    for ch in children:  # type: ignore[assignment]
        out.extend(_leaves(ch))
    return out


def _mass(node: _Node, probs: Mapping[str, float]) -> float:
    return float(sum(probs[leaf] for leaf in _leaves(node)))


def hierarchical_decode(probs: Mapping[str, float]) -> List[Tuple[str, float]]:
    """Top-down MAP path through the taxonomy, from coarse family to leaf.

    Returns a list ``[(family, P(family)), ..., (leaf, P(leaf))]``.  At each level the
    most probable child (by summed leaf mass) is chosen, so coarse decisions weigh
    bulk against bulk rather than bulk against a thin contact band.
    """
    path: List[Tuple[str, float]] = []
    node = _TREE
    while True:
        _, children = node
        if _is_leaf_level(children):
            best = max(children, key=lambda leaf: probs[leaf])  # type: ignore[arg-type]
            path.append((best, float(probs[best])))
            return path
        best_child = max(children, key=lambda ch: _mass(ch, probs))  # type: ignore[arg-type]
        path.append((best_child[0], _mass(best_child, probs)))
        node = best_child


def map_relation(e: IntervalGaussian, r: IntervalGaussian, tau: float = 0.0) -> str:
    """Relation of the most probable arrangement (the mode of the latent distribution).

    Uses the modal midpoints (the means) and modal durations (``max(mu_d, 0)``), then
    classifies the resulting definite arrangement.  Unbiased by tolerance-band width.
    """
    de = max(e.mu_d, 0.0)
    dr = max(r.mu_d, 0.0)
    se, fe = e.mu_t - de / 2, e.mu_t + de / 2
    sr, fr = r.mu_t - dr / 2, r.mu_t + dr / 2
    return classify_arrangement(sr - se, fr - fe, sr - fe, se - fr, tau)
