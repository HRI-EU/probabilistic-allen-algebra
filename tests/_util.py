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

"""Shared test utilities: independent Monte-Carlo estimation of relations.

The Monte-Carlo classifier uses the *same* single-tolerance partition rules as
``paa.relations`` but evaluated on sampled endpoints, giving an independent
check of the analytic Gaussian-orthant probabilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import truncnorm

from paa.intervals import IntervalGaussian
from paa.relations import RELATION_NAMES


def _sample_midpoint(iv: IntervalGaussian, n: int, rng: np.random.Generator) -> np.ndarray:
    if iv.sigma_t == 0.0:
        return np.full(n, iv.mu_t)
    return rng.normal(iv.mu_t, iv.sigma_t, n)


def _sample_duration(iv: IntervalGaussian, n: int, rng: np.random.Generator) -> np.ndarray:
    if iv.sigma_d == 0.0:
        return np.full(n, iv.mu_d)
    a = (0.0 - iv.mu_d) / iv.sigma_d
    return truncnorm.rvs(a, np.inf, loc=iv.mu_d, scale=iv.sigma_d, size=n, random_state=rng)


def classify(A, B, G, H, tau: float) -> np.ndarray:
    """Assign each sample to exactly one relation via the partition rules."""
    n = len(A)
    rel = np.empty(n, dtype=object)
    sa = np.where(A > tau, 1, np.where(A < -tau, -1, 0))
    sb = np.where(B > tau, 1, np.where(B < -tau, -1, 0))
    rel[(sa == 1) & (sb == 0)] = "finished_by"
    rel[(sa == 1) & (sb == -1)] = "contains"
    rel[(sa == 0) & (sb == 1)] = "starts"
    rel[(sa == 0) & (sb == 0)] = "equals"
    rel[(sa == 0) & (sb == -1)] = "started_by"
    rel[(sa == -1) & (sb == 1)] = "during"
    rel[(sa == -1) & (sb == 0)] = "finishes"
    pp = (sa == 1) & (sb == 1)
    rel[pp & (G < -tau)] = "overlaps"
    rel[pp & (np.abs(G) <= tau)] = "meets"
    rel[pp & (G > tau)] = "before"
    mm = (sa == -1) & (sb == -1)
    rel[mm & (H < -tau)] = "overlapped_by"
    rel[mm & (np.abs(H) <= tau)] = "met_by"
    rel[mm & (H > tau)] = "after"
    return rel


def mc_relations(
    e: IntervalGaussian,
    r: IntervalGaussian,
    tau: float,
    n: int = 200_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Monte-Carlo frequencies of the thirteen relations for ``(e, r)``."""
    rng = np.random.default_rng(seed)
    te = _sample_midpoint(e, n, rng)
    tr = _sample_midpoint(r, n, rng)
    de = _sample_duration(e, n, rng)
    dr = _sample_duration(r, n, rng)
    se, fe = te - de / 2, te + de / 2
    sr, fr = tr - dr / 2, tr + dr / 2
    A, B, G, H = sr - se, fr - fe, sr - fe, se - fr
    cls = classify(A, B, G, H, tau)
    return {name: float((cls == name).mean()) for name in RELATION_NAMES}
