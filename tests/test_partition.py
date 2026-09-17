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

"""The thirteen relations form a true partition: they sum to 1 for any tau."""

from __future__ import annotations

import numpy as np
import pytest

from paa import IntervalGaussian, relation_probabilities


def _random_interval(rng: np.random.Generator) -> IntervalGaussian:
    return IntervalGaussian(
        mu_t=float(rng.uniform(-5, 5)),
        sigma_t=float(rng.uniform(0.1, 2.0)),
        mu_d=float(rng.uniform(0.2, 4.0)),
        sigma_d=float(rng.uniform(0.05, 1.0)),
    )


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_partition_sums_to_one_random(tau: float) -> None:
    rng = np.random.default_rng(12345)
    for _ in range(25):
        e = _random_interval(rng)
        r = _random_interval(rng)
        probs = relation_probabilities(e, r, tau=tau)
        assert abs(sum(probs.values()) - 1.0) < 5e-3
        assert all(-1e-9 <= p <= 1.0 + 1e-9 for p in probs.values())


def test_partition_holds_for_points_and_intervals() -> None:
    from paa import point

    cases = [
        (point(0.0, 1.0), point(0.4, 1.0)),  # point-point
        (point(0.0, 1.2), IntervalGaussian(0.0, 0.4, 3.0, 0.0)),  # point-interval
        (IntervalGaussian(0.0, 1.0, 2.0, 0.3), IntervalGaussian(3.0, 1.0, 1.5, 0.4)),
    ]
    for e, r in cases:
        for tau in (0.0, 0.3, 1.0):
            probs = relation_probabilities(e, r, tau=tau)
            assert abs(sum(probs.values()) - 1.0) < 5e-3
