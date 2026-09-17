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

"""Limiting cases: deterministic configs and the 13 -> 5 -> 3 reduction."""

from __future__ import annotations

from paa import IntervalGaussian, point, relation_probabilities

POINT_INTERVAL_VANISH = (
    "meets",
    "overlaps",
    "equals",
    "finished_by",
    "contains",
    "started_by",
    "overlapped_by",
    "met_by",
)
POINT_INTERVAL_SURVIVE = ("before", "starts", "during", "finishes", "after")
POINT_POINT_SURVIVE = ("before", "equals", "after")


def test_deterministic_meets() -> None:
    # e = [-1, 1], r = [1, 3]: exact contact f_e = s_r.
    e = IntervalGaussian(mu_t=0.0, sigma_t=0.0, mu_d=2.0, sigma_d=0.0)
    r = IntervalGaussian(mu_t=2.0, sigma_t=0.0, mu_d=2.0, sigma_d=0.0)
    probs = relation_probabilities(e, r, tau=0.0)
    assert probs["meets"] == 1.0
    assert sum(v for k, v in probs.items() if k != "meets") == 0.0


def test_identical_points_equal() -> None:
    probs = relation_probabilities(point(0.0), point(0.0), tau=0.0)
    assert probs["equals"] == 1.0
    assert sum(v for k, v in probs.items() if k != "equals") == 0.0


def test_point_interval_reduces_to_five() -> None:
    e = point(0.0, 1.2)
    r = IntervalGaussian(mu_t=0.0, sigma_t=0.4, mu_d=3.0, sigma_d=0.0)
    probs = relation_probabilities(e, r, tau=0.3)
    for name in POINT_INTERVAL_VANISH:
        assert probs[name] < 1e-6, name
    assert abs(sum(probs[n] for n in POINT_INTERVAL_SURVIVE) - 1.0) < 1e-3


def test_point_point_reduces_to_three() -> None:
    probs = relation_probabilities(point(0.0, 1.0), point(0.4, 1.0), tau=0.3)
    vanish = [n for n in probs if n not in POINT_POINT_SURVIVE]
    for name in vanish:
        assert probs[name] < 1e-6, name
    assert abs(sum(probs[n] for n in POINT_POINT_SURVIVE) - 1.0) < 1e-3
