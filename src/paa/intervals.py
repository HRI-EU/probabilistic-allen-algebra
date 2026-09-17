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

"""Uncertain temporal objects: Gaussian time points and intervals.

An uncertain interval is

    I = [t - d/2, t + d/2]

with a Gaussian midpoint ``t ~ N(mu_t, sigma_t^2)`` and a non-negative duration
``d = D | D >= 0`` obtained by lower-truncating a latent Gaussian
``D ~ N(mu_d, sigma_d^2)``.  A time *point* is the degenerate interval with
``mu_d = sigma_d = 0``.

The midpoint and duration may be **correlated** through ``cov_td = Cov(t, D)``.
This is what lets the same machinery express the other interval
parameterizations (Appendix B of the paper): an interval given by an independent
*start* and duration has ``t = s + d/2`` and hence ``Cov(t, d) = +sigma_d^2/2``;
one given by an independent *end* and duration has ``t = e - d/2`` and
``Cov(t, d) = -sigma_d^2/2``.  An interval given by independent *start* and
*end* has ``t = (s + e)/2`` and ``D = e - s``, hence
``Cov(t, D) = (sigma_e^2 - sigma_s^2)/2`` -- zero when the two ends are equally
uncertain, in which case it *is* the midpoint form.  Use
:meth:`IntervalGaussian.from_start`, :meth:`IntervalGaussian.from_end` and
:meth:`IntervalGaussian.from_endpoints` to build these without computing the
covariance by hand.

Deterministic limits are allowed: ``sigma_t = 0`` fixes the midpoint and
``sigma_d = 0`` fixes the duration to ``mu_d`` (both then force ``cov_td = 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["IntervalGaussian", "point"]


@dataclass(frozen=True)
class IntervalGaussian:
    """A probabilistic interval parameterised by Gaussian midpoint and duration.

    Parameters
    ----------
    mu_t, sigma_t:
        Mean and standard deviation of the midpoint ``t``.
    mu_d, sigma_d:
        Mean and standard deviation of the latent duration ``D`` (truncated to
        ``D >= 0``).  ``sigma_d = 0`` makes the duration deterministic at
        ``mu_d``, which must then be non-negative.
    cov_td:
        Covariance ``Cov(t, D)`` between the midpoint and the latent duration
        (default ``0.0``, i.e. independent).  Must satisfy
        ``|cov_td| <= sigma_t * sigma_d`` so the ``(t, D)`` covariance block is
        positive semidefinite.  Prefer the :meth:`from_start` / :meth:`from_end`
        / :meth:`from_endpoints` constructors over setting this directly.
    """

    mu_t: float
    sigma_t: float
    mu_d: float
    sigma_d: float
    cov_td: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_t < 0:
            raise ValueError("sigma_t must be non-negative.")
        if self.sigma_d < 0:
            raise ValueError("sigma_d must be non-negative.")
        # A deterministic duration must be non-negative; a truncated Gaussian
        # duration may have a negative mean (mass below zero is discarded).
        if self.sigma_d == 0 and self.mu_d < 0:
            raise ValueError("A deterministic duration (sigma_d=0) must have mu_d >= 0.")
        # Keep the 2x2 (t, D) covariance block positive semidefinite.
        bound = self.sigma_t * self.sigma_d
        if abs(self.cov_td) > bound * (1.0 + 1e-9) + 1e-12:
            raise ValueError(
                "cov_td must satisfy |cov_td| <= sigma_t * sigma_d so that the "
                "midpoint-duration covariance matrix is positive semidefinite."
            )

    @classmethod
    def from_start(
        cls, mu_s: float, sigma_s: float, mu_d: float, sigma_d: float
    ) -> "IntervalGaussian":
        """Interval ``[s, s+d]`` from an **independent** Gaussian start and duration.

        With ``t = s + d/2`` this induces ``mu_t = mu_s + mu_d/2``,
        ``sigma_t^2 = sigma_s^2 + sigma_d^2/4`` and ``cov_td = +sigma_d^2/2``.
        """
        if sigma_s < 0:
            raise ValueError("sigma_s must be non-negative.")
        sigma_t = math.sqrt(sigma_s**2 + sigma_d**2 / 4.0)
        return cls(
            mu_t=mu_s + mu_d / 2.0,
            sigma_t=sigma_t,
            mu_d=mu_d,
            sigma_d=sigma_d,
            cov_td=sigma_d**2 / 2.0,
        )

    @classmethod
    def from_end(
        cls, mu_f: float, sigma_f: float, mu_d: float, sigma_d: float
    ) -> "IntervalGaussian":
        """Interval ``[f-d, f]`` from an **independent** Gaussian end and duration.

        With ``t = f - d/2`` this induces ``mu_t = mu_f - mu_d/2``,
        ``sigma_t^2 = sigma_f^2 + sigma_d^2/4`` and ``cov_td = -sigma_d^2/2``.
        """
        if sigma_f < 0:
            raise ValueError("sigma_f must be non-negative.")
        sigma_t = math.sqrt(sigma_f**2 + sigma_d**2 / 4.0)
        return cls(
            mu_t=mu_f - mu_d / 2.0,
            sigma_t=sigma_t,
            mu_d=mu_d,
            sigma_d=sigma_d,
            cov_td=-sigma_d**2 / 2.0,
        )

    @classmethod
    def from_endpoints(
        cls, mu_s: float, sigma_s: float, mu_e: float, sigma_e: float
    ) -> "IntervalGaussian":
        """Interval ``[s, e]`` from **independent** Gaussian start and end.

        The duration is the difference ``D = e - s``, so the truncation
        ``D >= 0`` reads as conditioning on the end not preceding the start.
        With ``t = (s + e)/2`` this induces ``mu_t = (mu_s + mu_e)/2``,
        ``sigma_t^2 = (sigma_s^2 + sigma_e^2)/4``, ``mu_d = mu_e - mu_s``,
        ``sigma_d^2 = sigma_s^2 + sigma_e^2`` and
        ``cov_td = (sigma_e^2 - sigma_s^2)/2``.  Equally uncertain ends give
        ``cov_td = 0``: the plain midpoint form.  A sharp start with an
        uncertain end is :meth:`from_start` with ``sigma_s = 0``, and the
        mirror case is :meth:`from_end` with ``sigma_f = 0``.
        """
        if sigma_s < 0:
            raise ValueError("sigma_s must be non-negative.")
        if sigma_e < 0:
            raise ValueError("sigma_e must be non-negative.")
        if sigma_s == 0 and sigma_e == 0 and mu_e < mu_s:
            raise ValueError("A deterministic end must not precede its start.")
        return cls(
            mu_t=(mu_s + mu_e) / 2.0,
            sigma_t=math.sqrt((sigma_s**2 + sigma_e**2) / 4.0),
            mu_d=mu_e - mu_s,
            sigma_d=math.sqrt(sigma_s**2 + sigma_e**2),
            cov_td=(sigma_e**2 - sigma_s**2) / 2.0,
        )

    @property
    def is_point(self) -> bool:
        """True if this object is a deterministic time point (zero duration)."""
        return self.mu_d == 0.0 and self.sigma_d == 0.0


def point(mu_t: float, sigma_t: float = 0.0) -> IntervalGaussian:
    """Construct a (possibly uncertain) time point as a degenerate interval."""
    return IntervalGaussian(mu_t=mu_t, sigma_t=sigma_t, mu_d=0.0, sigma_d=0.0)
