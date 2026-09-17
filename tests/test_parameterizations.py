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

"""Alternative interval parameterizations: start+duration, end+duration, start+end.

``IntervalGaussian.from_start`` / ``from_end`` / ``from_endpoints`` build the
midpoint--duration form with the induced midpoint--duration covariance
``cov_td``, and the relation engine uses a full (non-diagonal) latent
covariance.  These tests check that

* the analytic probabilities match Monte-Carlo of the *actual* start/end or
  endpoint model (validating the correlated-covariance path end to end),
* an independent start+duration law genuinely differs from an independent
  midpoint+duration law with the same marginals (the correlation matters), and
* two equally uncertain endpoints are exactly the midpoint form.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx
from scipy.stats import truncnorm

from paa import IntervalGaussian, relation_probabilities
from paa.relations import RELATION_NAMES
from tests._util import classify

# (mu_loc, sigma_loc, mu_d, sigma_d) for the event and reference intervals.
_E = (0.0, 0.5, 2.0, 0.4)
_R = (1.5, 0.6, 1.5, 0.5)


def _sample_duration(mu_d: float, sigma_d: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if sigma_d == 0.0:
        return np.full(n, mu_d)
    a = (0.0 - mu_d) / sigma_d
    return truncnorm.rvs(a, np.inf, loc=mu_d, scale=sigma_d, size=n, random_state=rng)


def _sample_loc(mu: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.full(n, mu) if sigma == 0.0 else rng.normal(mu, sigma, n)


def _mc(kind: str, e_p, r_p, tau: float, n: int = 400_000, seed: int = 1):
    """Monte-Carlo relation frequencies for an independent location+duration model."""
    rng = np.random.default_rng(seed)

    def endpoints(mu_loc, sigma_loc, mu_d, sigma_d):
        loc = _sample_loc(mu_loc, sigma_loc, n, rng)
        d = _sample_duration(mu_d, sigma_d, n, rng)
        if kind == "start":
            return loc, loc + d  # [s, s+d]
        return loc - d, loc  # [f-d, f]

    se, fe = endpoints(*e_p)
    sr, fr = endpoints(*r_p)
    A, B, G, H = sr - se, fr - fe, sr - fe, se - fr
    cls = classify(A, B, G, H, tau)
    return {name: float((cls == name).mean()) for name in RELATION_NAMES}


def _mc_endpoints(e_p, r_p, tau: float, n: int = 600_000, seed: int = 2):
    """Monte-Carlo relation frequencies for independent start and end, kept where end >= start."""
    rng = np.random.default_rng(seed)

    def endpoints(mu_s, sigma_s, mu_e, sigma_e):
        return _sample_loc(mu_s, sigma_s, n, rng), _sample_loc(mu_e, sigma_e, n, rng)

    se, fe = endpoints(*e_p)
    sr, fr = endpoints(*r_p)
    keep = (fe >= se) & (fr >= sr)  # the truncation D >= 0, per interval
    se, fe, sr, fr = se[keep], fe[keep], sr[keep], fr[keep]
    A, B, G, H = sr - se, fr - fe, sr - fe, se - fr
    cls = classify(A, B, G, H, tau)
    return {name: float((cls == name).mean()) for name in RELATION_NAMES}


# (mu_s, sigma_s, mu_e, sigma_e): unequal end uncertainties, so cov_td != 0.
_E_ENDS = (0.0, 0.5, 2.0, 0.3)
_R_ENDS = (1.0, 0.2, 2.8, 0.6)


def test_from_start_matches_monte_carlo() -> None:
    e = IntervalGaussian.from_start(*_E)
    r = IntervalGaussian.from_start(*_R)
    analytic = relation_probabilities(e, r, tau=0.25)
    mc = _mc("start", _E, _R, tau=0.25)
    for name in RELATION_NAMES:
        assert analytic[name] == approx(mc[name], abs=5e-3), name


def test_from_end_matches_monte_carlo() -> None:
    e = IntervalGaussian.from_end(*_E)
    r = IntervalGaussian.from_end(*_R)
    analytic = relation_probabilities(e, r, tau=0.25)
    mc = _mc("end", _E, _R, tau=0.25)
    for name in RELATION_NAMES:
        assert analytic[name] == approx(mc[name], abs=5e-3), name


def test_from_endpoints_matches_monte_carlo() -> None:
    e = IntervalGaussian.from_endpoints(*_E_ENDS)
    r = IntervalGaussian.from_endpoints(*_R_ENDS)
    analytic = relation_probabilities(e, r, tau=0.25)
    mc = _mc_endpoints(_E_ENDS, _R_ENDS, tau=0.25)
    for name in RELATION_NAMES:
        assert analytic[name] == approx(mc[name], abs=5e-3), name


def test_from_endpoints_induced_moments() -> None:
    e = IntervalGaussian.from_endpoints(1.0, 0.3, 3.0, 0.4)
    assert e.mu_t == approx(2.0)
    assert e.sigma_t == approx(((0.3**2 + 0.4**2) / 4) ** 0.5)
    assert e.mu_d == approx(2.0)
    assert e.sigma_d == approx((0.3**2 + 0.4**2) ** 0.5)
    assert e.cov_td == approx((0.4**2 - 0.3**2) / 2)  # end noisier than start -> positive


def test_equal_endpoint_noise_is_the_midpoint_form() -> None:
    """Cov(t, D) = (sigma_e^2 - sigma_s^2)/2 vanishes when the ends are equally uncertain."""
    e = IntervalGaussian.from_endpoints(0.0, 0.5, 2.0, 0.5)
    mid = IntervalGaussian(1.0, 0.5 / 2**0.5, 2.0, 0.5 * 2**0.5)
    assert e.cov_td == approx(0.0)
    assert (e.mu_t, e.sigma_t, e.mu_d, e.sigma_d) == approx(
        (mid.mu_t, mid.sigma_t, mid.mu_d, mid.sigma_d)
    )
    # Same law, so the same probabilities -- up to the orthant integrator, whose quasi-random
    # path shifts with the last ulp of its inputs (scipy abseps is 1e-5 per call).
    r = IntervalGaussian(1.5, 0.4, 1.0, 0.3)
    p_ends = relation_probabilities(e, r, tau=0.2)
    p_mid = relation_probabilities(mid, r, tau=0.2)
    for name in RELATION_NAMES:
        assert p_ends[name] == approx(p_mid[name], abs=1e-4), name


def test_one_sharp_end_collapses_onto_the_single_location_forms() -> None:
    """A start known exactly is from_start with no start noise; mirror for the end."""

    def moments(iv):
        return (iv.mu_t, iv.sigma_t, iv.mu_d, iv.sigma_d, iv.cov_td)

    assert moments(IntervalGaussian.from_endpoints(0.0, 0.0, 2.0, 0.6)) == approx(
        moments(IntervalGaussian.from_start(0.0, 0.0, 2.0, 0.6))
    )
    assert moments(IntervalGaussian.from_endpoints(0.0, 0.6, 2.0, 0.0)) == approx(
        moments(IntervalGaussian.from_end(2.0, 0.0, 2.0, 0.6))
    )


def test_from_endpoints_refuses_a_deterministic_reversed_interval() -> None:
    with pytest.raises(ValueError):
        IntervalGaussian.from_endpoints(2.0, 0.0, 1.0, 0.0)
    # Uncertain ends may have mu_e < mu_s: the mass below D = 0 is what the truncation drops.
    IntervalGaussian.from_endpoints(2.0, 1.0, 1.0, 1.0)


def test_induced_covariance_values() -> None:
    e = IntervalGaussian.from_start(0.0, 0.3, 2.0, 0.6)
    assert e.mu_t == approx(0.0 + 2.0 / 2)  # mu_s + mu_d/2
    assert e.sigma_t == approx((0.3**2 + 0.6**2 / 4) ** 0.5)
    assert e.cov_td == approx(0.6**2 / 2)  # +sigma_d^2/2
    f = IntervalGaussian.from_end(0.0, 0.3, 2.0, 0.6)
    assert f.cov_td == approx(-(0.6**2) / 2)  # -sigma_d^2/2


def test_independent_parameterizations_differ() -> None:
    """Same marginals, different midpoint-duration correlation -> different law."""
    mu_d, sig_d = 2.0, 0.6
    sig_t = 0.5  # need sig_t >= sig_d/2 = 0.3 for a realizable start model
    sig_s = (sig_t**2 - sig_d**2 / 4) ** 0.5

    mid_e = IntervalGaussian(0.0, sig_t, mu_d, sig_d)  # cov_td = 0
    start_e = IntervalGaussian.from_start(-mu_d / 2, sig_s, mu_d, sig_d)  # cov_td > 0

    # The two share every marginal moment...
    assert start_e.mu_t == approx(mid_e.mu_t)
    assert start_e.sigma_t == approx(mid_e.sigma_t)
    assert (start_e.mu_d, start_e.sigma_d) == (mid_e.mu_d, mid_e.sigma_d)
    assert start_e.cov_td != approx(0.0)

    r = IntervalGaussian(1.0, 0.5, 1.5, 0.5)
    p_mid = relation_probabilities(mid_e, r, tau=0.2)
    p_start = relation_probabilities(start_e, r, tau=0.2)
    assert max(abs(p_mid[k] - p_start[k]) for k in RELATION_NAMES) > 1e-2


def test_partition_preserved_for_all_constructors() -> None:
    for ctor in (
        IntervalGaussian.from_start,
        IntervalGaussian.from_end,
        IntervalGaussian.from_endpoints,
    ):
        e = ctor(0.0, 0.5, 2.0, 0.4)
        r = ctor(1.0, 0.6, 1.5, 0.5)
        for tau in (0.0, 0.1, 0.5):
            p = relation_probabilities(e, r, tau=tau)
            assert sum(p.values()) == approx(1.0, abs=2e-3)


def test_cov_td_must_be_positive_semidefinite() -> None:
    with pytest.raises(ValueError):
        IntervalGaussian(0.0, 1.0, 2.0, 1.0, cov_td=2.0)  # |2| > 1*1
    IntervalGaussian(0.0, 1.0, 2.0, 1.0, cov_td=1.0)  # boundary is allowed


def test_default_is_independent_and_unchanged() -> None:
    """A plain IntervalGaussian still has zero correlation (backward compatible)."""
    e = IntervalGaussian(0.0, 1.0, 2.0, 0.3)
    assert e.cov_td == 0.0
