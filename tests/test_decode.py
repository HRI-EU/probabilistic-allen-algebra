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

"""Best-fitting-relation selection: flat, hierarchical, and mode-arrangement."""

from __future__ import annotations

from paa import (
    IntervalGaussian,
    THICK_RELATIONS,
    hierarchical_decode,
    map_relation,
    most_probable_relation,
    relation_probabilities,
)

# A clearly "during" configuration: short e well inside long r, low uncertainty.
E_DURING = IntervalGaussian(mu_t=0.0, sigma_t=0.2, mu_d=1.0, sigma_d=0.1)
R_DURING = IntervalGaussian(mu_t=0.0, sigma_t=0.2, mu_d=5.0, sigma_d=0.1)

# A clearly "before" configuration: e well before r.
E_BEFORE = IntervalGaussian(mu_t=0.0, sigma_t=0.2, mu_d=1.0, sigma_d=0.1)
R_BEFORE = IntervalGaussian(mu_t=10.0, sigma_t=0.2, mu_d=1.0, sigma_d=0.1)


def test_most_probable_relation_basic() -> None:
    probs = relation_probabilities(E_DURING, R_DURING, tau=0.25)
    name, p = most_probable_relation(probs)
    assert name == "during"
    assert p == probs["during"]


def test_most_probable_relation_restrict() -> None:
    probs = relation_probabilities(E_BEFORE, R_BEFORE, tau=0.25)
    name, _ = most_probable_relation(probs, restrict=THICK_RELATIONS)
    assert name == "before"


def test_hierarchical_decode_path() -> None:
    probs = relation_probabilities(E_DURING, R_DURING, tau=0.25)
    path = hierarchical_decode(probs)
    families = [name for name, _ in path]
    assert families[0] == "non_separated"
    assert "e_inside_r" in families
    assert path[-1][0] == "during"
    # probabilities along the path are non-increasing (nested subsets).
    masses = [p for _, p in path]
    assert all(a + 1e-9 >= b for a, b in zip(masses, masses[1:]))


def test_map_relation_meets_unbiased() -> None:
    # Mode arrangement is an exact meeting; flat argmax would prefer a thick neighbour.
    e = IntervalGaussian(mu_t=0.0, sigma_t=0.5, mu_d=2.0, sigma_d=0.2)  # mode e = [-1, 1]
    r = IntervalGaussian(mu_t=2.0, sigma_t=0.5, mu_d=2.0, sigma_d=0.2)  # mode r = [1, 3]
    assert map_relation(e, r, tau=0.1) == "meets"


def test_map_relation_before() -> None:
    assert map_relation(E_BEFORE, R_BEFORE, tau=0.1) == "before"
