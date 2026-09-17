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

"""Analytic Gaussian-orthant probabilities agree with Monte-Carlo frequencies."""

from __future__ import annotations

import pytest

from paa import IntervalGaussian, relation_probabilities
from paa.relations import RELATION_NAMES

from _util import mc_relations

CASES = [
    (IntervalGaussian(0.0, 1.0, 2.0, 0.3), IntervalGaussian(3.0, 1.0, 1.5, 0.4)),
    (IntervalGaussian(0.0, 0.8, 3.0, 0.5), IntervalGaussian(0.5, 1.2, 1.0, 0.2)),
    (IntervalGaussian(-1.0, 1.5, 1.0, 0.4), IntervalGaussian(1.0, 0.6, 4.0, 0.6)),
]


@pytest.mark.parametrize("e,r", CASES)
@pytest.mark.parametrize("tau", [0.0, 0.25])
def test_analytic_matches_monte_carlo(e: IntervalGaussian, r: IntervalGaussian, tau: float) -> None:
    analytic = relation_probabilities(e, r, tau=tau)
    mc = mc_relations(e, r, tau=tau, n=300_000, seed=1)
    for name in RELATION_NAMES:
        assert abs(analytic[name] - mc[name]) < 6e-3, name
