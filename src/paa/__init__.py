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

"""``paa`` -- the probabilistic Allen algebra.

Allen's thirteen interval relations between uncertain temporal objects, derived
as probabilities from Gaussian midpoints and truncated-Gaussian durations.  Each
relation is a conditional multivariate-Gaussian orthant probability; with a
single tolerance ``tau`` the thirteen relations form a true partition (sum to 1)
and recover crisp Allen as ``tau -> 0``.

Quickstart
----------
>>> from paa import IntervalGaussian, relation_probabilities
>>> e = IntervalGaussian(mu_t=0.0, sigma_t=1.0, mu_d=2.0, sigma_d=0.3)
>>> r = IntervalGaussian(mu_t=3.0, sigma_t=1.0, mu_d=1.5, sigma_d=0.4)
>>> p = relation_probabilities(e, r, tau=0.25)
>>> abs(sum(p.values()) - 1.0) < 1e-3   # a true partition, up to MVN-CDF noise
True
"""

from __future__ import annotations

from .decode import hierarchical_decode, map_relation, most_probable_relation
from .intervals import IntervalGaussian, point
from .relations import (
    CONVERSE,
    RELATION_NAMES,
    THICK_RELATIONS,
    ProbabilisticAllenRelations,
    classify_arrangement,
    relation_probabilities,
)
from .primitives import (
    CANONICAL_SIGNS,
    contacts,
    decompose,
    format_decomposition,
    primitive_probabilities,
)
from .taxonomy import LEAVES, TREE, VIEWS, coarse_predicates, leaves_of, refine

__all__ = [
    "IntervalGaussian",
    "point",
    "ProbabilisticAllenRelations",
    "relation_probabilities",
    "RELATION_NAMES",
    "CONVERSE",
    "THICK_RELATIONS",
    "classify_arrangement",
    "coarse_predicates",
    "refine",
    "leaves_of",
    "LEAVES",
    "TREE",
    "VIEWS",
    "most_probable_relation",
    "hierarchical_decode",
    "map_relation",
    "primitive_probabilities",
    "decompose",
    "contacts",
    "format_decomposition",
    "CANONICAL_SIGNS",
    "main",
]

__version__ = "0.1.0"


def main() -> None:
    """Console-script entry point: print a short demonstration."""
    e = IntervalGaussian(mu_t=0.0, sigma_t=1.0, mu_d=2.0, sigma_d=0.3)
    r = IntervalGaussian(mu_t=3.0, sigma_t=1.0, mu_d=1.5, sigma_d=0.4)
    probs = relation_probabilities(e, r, tau=0.25)
    print("Probabilistic Allen relations  (e vs r),  tau = 0.25")
    for name in RELATION_NAMES:
        print(f"  {name:14s} {probs[name]:.5f}")
    print(f"  {'sum':14s} {sum(probs.values()):.5f}")
