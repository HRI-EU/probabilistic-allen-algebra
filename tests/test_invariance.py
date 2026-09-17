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

"""Structural properties: scale invariance and converse symmetry."""

from __future__ import annotations

from dataclasses import replace

from paa import IntervalGaussian, relation_probabilities
from paa.relations import CONVERSE, RELATION_NAMES

E = IntervalGaussian(0.0, 1.0, 2.0, 0.3)
R = IntervalGaussian(3.0, 1.0, 1.5, 0.4)


def _scaled(iv: IntervalGaussian, c: float) -> IntervalGaussian:
    return replace(iv, mu_t=iv.mu_t * c, sigma_t=iv.sigma_t * c, mu_d=iv.mu_d * c, sigma_d=iv.sigma_d * c)


def test_scale_invariance() -> None:
    # Scale invariance is mathematically exact (identical standardized arguments);
    # the tolerance reflects the ~1e-5 numerical noise floor of scipy's MVN CDF.
    tau = 0.25
    base = relation_probabilities(E, R, tau=tau)
    for c in (0.5, 2.0, 10.0, 100.0):
        scaled = relation_probabilities(_scaled(E, c), _scaled(R, c), tau=tau * c)
        for name in RELATION_NAMES:
            assert abs(base[name] - scaled[name]) < 1e-4, (name, c)


def test_converse_symmetry() -> None:
    tau = 0.25
    p_er = relation_probabilities(E, R, tau=tau)
    p_re = relation_probabilities(R, E, tau=tau)
    for name in RELATION_NAMES:
        assert abs(p_er[name] - p_re[CONVERSE[name]]) < 5e-3, name
