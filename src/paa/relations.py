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

"""Distribution-induced probabilistic Allen interval relations.

Every Allen relation between two uncertain intervals ``e`` (event) and ``r``
(reference) is a conjunction of linear inequalities in the latent Gaussian
vector ``(X, D_e, D_r)`` with ``X = t_r - t_e``.  Its probability is the
corresponding conditional multivariate-Gaussian orthant probability,
conditioned on the non-negative durations ``D_e >= 0`` and ``D_r >= 0``.

The thirteen relations are defined with a single tolerance ``tau >= 0`` so that
they form a **true partition**: for any ``tau`` the probabilities sum to one,
and crisp Allen is recovered as ``tau -> 0``.  See
``latex_main/notes/partition_theorem.md`` for the derivation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import multivariate_normal, norm

from .intervals import IntervalGaussian

__all__ = [
    "RELATION_NAMES",
    "CONVERSE",
    "THICK_RELATIONS",
    "ProbabilisticAllenRelations",
    "relation_probabilities",
    "classify_arrangement",
]

#: The six full-dimensional ("thick") relations defined by strict inequalities.
#: The other seven are contact relations occupying thin tolerance bands.
THICK_RELATIONS: Tuple[str, ...] = (
    "before",
    "overlaps",
    "during",
    "contains",
    "overlapped_by",
    "after",
)

#: The thirteen Allen base relations, in canonical order.
RELATION_NAMES: Tuple[str, ...] = (
    "before",
    "meets",
    "overlaps",
    "starts",
    "during",
    "finishes",
    "equals",
    "finished_by",
    "contains",
    "started_by",
    "overlapped_by",
    "met_by",
    "after",
)

#: Map each relation to its converse (the relation that holds for ``(r, e)``).
CONVERSE: Dict[str, str] = {
    "before": "after",
    "meets": "met_by",
    "overlaps": "overlapped_by",
    "starts": "started_by",
    "during": "contains",
    "finishes": "finished_by",
    "equals": "equals",
    "finished_by": "finishes",
    "contains": "during",
    "started_by": "starts",
    "overlapped_by": "overlaps",
    "met_by": "meets",
    "after": "before",
}

# An (expr, strict) inequality means ``expr <= 0`` (strict ``< 0`` if strict).
_Expr = Dict[str, float]
_Ineq = Tuple[_Expr, bool]


class ProbabilisticAllenRelations:
    """Compute probabilistic Allen relations between two uncertain intervals.

    Parameters
    ----------
    e, r:
        The event and reference intervals.
    eps:
        Threshold below which a variance is treated as deterministic; latent
        variables with ``sigma <= eps`` are removed from the Gaussian system to
        avoid singular CDF evaluations.
    """

    def __init__(self, e: IntervalGaussian, r: IntervalGaussian, eps: float = 1e-12):
        self.e = e
        self.r = r
        self.eps = eps

        # X = t_r - t_e
        self.mu_x = r.mu_t - e.mu_t
        self.sigma_x = float(np.sqrt(e.sigma_t**2 + r.sigma_t**2))

        # Active (non-deterministic) latent variables among X, D_e, D_r.
        self.var_names: List[str] = []
        means: List[float] = []
        if self.sigma_x > eps:
            self.var_names.append("X")
            means.append(self.mu_x)
        if e.sigma_d > eps:
            self.var_names.append("De")
            means.append(e.mu_d)
        if r.sigma_d > eps:
            self.var_names.append("Dr")
            means.append(r.mu_d)

        self.means = np.asarray(means, dtype=float)

        # Full latent covariance.  Diagonal entries are the marginal variances;
        # the off-diagonal entries carry each interval's midpoint--duration
        # covariance into X = t_r - t_e:
        #   Cov(X, D_e) = -Cov(t_e, D_e) = -e.cov_td,
        #   Cov(X, D_r) = +Cov(t_r, D_r) = +r.cov_td,
        # while D_e and D_r stay independent (distinct intervals).
        idx = {name: i for i, name in enumerate(self.var_names)}
        cov = np.zeros((len(self.var_names), len(self.var_names)), dtype=float)
        if "X" in idx:
            cov[idx["X"], idx["X"]] = self.sigma_x**2
        if "De" in idx:
            cov[idx["De"], idx["De"]] = e.sigma_d**2
        if "Dr" in idx:
            cov[idx["Dr"], idx["Dr"]] = r.sigma_d**2
        if "X" in idx and "De" in idx:
            cov[idx["X"], idx["De"]] = cov[idx["De"], idx["X"]] = -e.cov_td
        if "X" in idx and "Dr" in idx:
            cov[idx["X"], idx["Dr"]] = cov[idx["Dr"], idx["X"]] = r.cov_td
        self.cov = cov
        self.det = {
            "X": self.mu_x if self.sigma_x <= eps else None,
            "De": e.mu_d if e.sigma_d <= eps else None,
            "Dr": r.mu_d if r.sigma_d <= eps else None,
        }
        self.trunc_norm = self._truncation_normalizer()

    # ------------------------------------------------------------------
    # Gaussian probability core
    # ------------------------------------------------------------------

    def _truncation_normalizer(self) -> float:
        """Q = P(D_e >= 0) P(D_r >= 0)."""
        q = 1.0
        for iv in (self.e, self.r):
            if iv.sigma_d > self.eps:
                q *= float(norm.cdf(iv.mu_d / iv.sigma_d))
            else:
                q *= 1.0 if iv.mu_d >= -self.eps else 0.0
        if q <= 0:
            raise ValueError("Duration truncation normalization is zero.")
        return q

    def _row(self, expr: _Expr) -> Tuple[np.ndarray, float]:
        """Convert an expression dict to (active coefficients, deterministic constant)."""
        a = np.zeros(len(self.var_names), dtype=float)
        c = float(expr.get("_", 0.0))
        for name, coeff in expr.items():
            if name == "_":
                continue
            if name in self.var_names:
                a[self.var_names.index(name)] += coeff
            else:
                c += coeff * self.det[name]
        return a, c

    def _raw_prob(self, inequalities: List[_Ineq]) -> float:
        """Probability of a conjunction of ``expr <= 0`` inequalities."""
        rows: List[np.ndarray] = []
        consts: List[float] = []
        for expr, strict in inequalities:
            a, c = self._row(expr)
            if len(a) == 0 or not a.any():
                mean = float(a @ self.means + c) if len(a) else c
                # Deterministic inequality: decide it exactly.
                if strict:
                    if not (mean < -self.eps):
                        return 0.0
                else:
                    if not (mean <= self.eps):
                        return 0.0
                continue
            rows.append(a)
            consts.append(c)

        if not rows:
            return 1.0

        A = np.vstack(rows)
        c_vec = np.asarray(consts, dtype=float)
        mean_w = A @ self.means + c_vec
        cov_w = A @ self.cov @ A.T
        p = multivariate_normal(mean=mean_w, cov=cov_w, allow_singular=True).cdf(
            np.zeros(len(rows))
        )
        return float(np.clip(p, 0.0, 1.0))

    def _conditional_prob(self, event: List[_Ineq]) -> float:
        """P(event | D_e >= 0, D_r >= 0)."""
        ineqs: List[_Ineq] = []
        if self.e.sigma_d > self.eps:
            ineqs.append(({"De": -1.0}, False))
        if self.r.sigma_d > self.eps:
            ineqs.append(({"Dr": -1.0}, False))
        ineqs.extend(event)
        return self._raw_prob(ineqs) / self.trunc_norm

    # ------------------------------------------------------------------
    # Inequality helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_const(expr: _Expr, c: float) -> _Expr:
        out = dict(expr)
        out["_"] = out.get("_", 0.0) + c
        return out

    @staticmethod
    def _neg(expr: _Expr) -> _Expr:
        return {k: -v for k, v in expr.items()}

    def _le(self, expr: _Expr, bound: float, strict: bool = False) -> _Ineq:
        """expr <= bound."""
        return self._add_const(expr, -bound), strict

    def _ge(self, expr: _Expr, bound: float, strict: bool = False) -> _Ineq:
        """expr >= bound, i.e. -expr + bound <= 0."""
        return self._add_const(self._neg(expr), bound), strict

    def _abs_le(self, expr: _Expr, tau: float) -> List[_Ineq]:
        """|expr| <= tau."""
        return [self._le(expr, tau), self._ge(expr, -tau)]

    # ------------------------------------------------------------------
    # Boundary-difference expressions in (X, D_e, D_r)
    # ------------------------------------------------------------------

    @property
    def A(self) -> _Expr:
        """A = s_r - s_e = X + De/2 - Dr/2 (start comparison)."""
        return {"X": 1.0, "De": 0.5, "Dr": -0.5}

    @property
    def B(self) -> _Expr:
        """B = f_r - f_e = X - De/2 + Dr/2 (finish comparison)."""
        return {"X": 1.0, "De": -0.5, "Dr": 0.5}

    @property
    def G(self) -> _Expr:
        """G = s_r - f_e = X - De/2 - Dr/2 (frontal gap)."""
        return {"X": 1.0, "De": -0.5, "Dr": -0.5}

    @property
    def H(self) -> _Expr:
        """H = s_e - f_r = -X - De/2 - Dr/2 (rear gap)."""
        return {"X": -1.0, "De": -0.5, "Dr": -0.5}

    # ------------------------------------------------------------------
    # The thirteen relations (single-tolerance partition form)
    # ------------------------------------------------------------------

    def before(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob([self._ge(self.G, tau, strict=True)])

    def meets(self, tau: float = 0.0) -> float:
        # |G| <= tau within the (A>tau, B>tau) corner (symmetric contact band).
        _check_tau(tau)
        return self._conditional_prob(
            self._abs_le(self.G, tau)
            + [self._ge(self.A, tau, strict=True), self._ge(self.B, tau, strict=True)]
        )

    def overlaps(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [
                self._ge(self.A, tau, strict=True),
                self._le(self.G, -tau, strict=True),
                self._ge(self.B, tau, strict=True),
            ]
        )

    def starts(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            self._abs_le(self.A, tau) + [self._ge(self.B, tau, strict=True)]
        )

    def during(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [self._le(self.A, -tau, strict=True), self._ge(self.B, tau, strict=True)]
        )

    def finishes(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [self._le(self.A, -tau, strict=True)] + self._abs_le(self.B, tau)
        )

    def equals(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(self._abs_le(self.A, tau) + self._abs_le(self.B, tau))

    def finished_by(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [self._ge(self.A, tau, strict=True)] + self._abs_le(self.B, tau)
        )

    def contains(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [self._ge(self.A, tau, strict=True), self._le(self.B, -tau, strict=True)]
        )

    def started_by(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            self._abs_le(self.A, tau) + [self._le(self.B, -tau, strict=True)]
        )

    def overlapped_by(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob(
            [
                self._le(self.A, -tau, strict=True),
                self._le(self.H, -tau, strict=True),
                self._le(self.B, -tau, strict=True),
            ]
        )

    def met_by(self, tau: float = 0.0) -> float:
        # |H| <= tau within the (A<-tau, B<-tau) corner (symmetric contact band).
        _check_tau(tau)
        return self._conditional_prob(
            self._abs_le(self.H, tau)
            + [self._le(self.A, -tau, strict=True), self._le(self.B, -tau, strict=True)]
        )

    def after(self, tau: float = 0.0) -> float:
        _check_tau(tau)
        return self._conditional_prob([self._ge(self.H, tau, strict=True)])

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def all_relations(self, tau: float = 0.0) -> Dict[str, float]:
        """Return all thirteen base relation probabilities."""
        _check_tau(tau)
        return {name: float(np.clip(getattr(self, name)(tau), 0.0, 1.0)) for name in RELATION_NAMES}


def relation_probabilities(
    e: IntervalGaussian, r: IntervalGaussian, tau: float = 0.0, eps: float = 1e-12
) -> Dict[str, float]:
    """Convenience: all thirteen relation probabilities for ``(e, r)``."""
    return ProbabilisticAllenRelations(e, r, eps=eps).all_relations(tau)


def _check_tau(tau: float) -> None:
    if tau < 0:
        raise ValueError("tau must be non-negative.")


def classify_arrangement(A: float, B: float, G: float, H: float, tau: float = 0.0) -> str:
    """Return the single relation of a definite boundary arrangement (partition rule).

    ``A, B, G, H`` are the boundary differences ``s_r-s_e, f_r-f_e, s_r-f_e, s_e-f_r``.
    This is the crisp, single-configuration version of the relation partition; every
    arrangement maps to exactly one relation.
    """
    _check_tau(tau)
    sa = 1 if A > tau else (-1 if A < -tau else 0)
    sb = 1 if B > tau else (-1 if B < -tau else 0)
    if sa == 1 and sb == 1:
        if G < -tau:
            return "overlaps"
        return "meets" if abs(G) <= tau else "before"
    if sa == -1 and sb == -1:
        if H < -tau:
            return "overlapped_by"
        return "met_by" if abs(H) <= tau else "after"
    return {
        (1, 0): "finished_by",
        (1, -1): "contains",
        (0, 1): "starts",
        (0, 0): "equals",
        (0, -1): "started_by",
        (-1, 1): "during",
        (-1, 0): "finishes",
    }[(sa, sb)]
