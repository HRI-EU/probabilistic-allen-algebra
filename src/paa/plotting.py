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

"""Figure generators for the paper, docs, and README.

Every figure is produced from the package itself, so the visuals stay in sync with the
algebra. ``matplotlib`` is imported lazily; install the ``viz`` extra to use this module
(``pip install -e '.[viz]'`` or ``uv run --extra viz ...``).

Each ``fig_*`` function returns a :class:`matplotlib.figure.Figure`; :func:`save` writes it
to a path (vector PDF/SVG recommended).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from .intervals import IntervalGaussian
from .relations import RELATION_NAMES, relation_probabilities

__all__ = [
    "save",
    "fig_robustness",
    "fig_inadequacy",
    "fig_case_study",
    "fig_relations_catalogue",
    "fig_probability_vs_separation",
    "fig_relation_cascade",
    "fig_partition_sum_vs_tau",
    "fig_tau_explainer",
    "fig_mc_parity",
    "fig_limiting_transition",
    "fig_scale_invariance",
    "fig_hourglass_heatmaps",
    "fig_phase_diagram",
    "generate_all",
]

# Intuitive label for each relation, used in annotated figures.
PRETTY = {
    "before": "really before",
    "meets": "just touching",
    "overlaps": "overlapping",
    "starts": "beginning with",
    "during": "fully inside",
    "finishes": "ending together",
    "equals": "coincident",
    "finished_by": "ends together (e longer)",
    "contains": "fully contains",
    "started_by": "starts together (e longer)",
    "overlapped_by": "overlapped from left",
    "met_by": "touched from left",
    "after": "really after",
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --- Print-friendly (greyscale) styling -------------------------------------
# Every figure must stay legible when printed in black-and-white, so colour is
# never the sole carrier of meaning: interval bars differ by fill, line
# ensembles differ by a luminance ramp *and* a cycling line style, and
# categorical maps use a monotone grey ramp with drawn boundaries.

_FILL = "0.78"   # filled interval bar  (e / X)
_OPEN = "white"  # outlined interval bar (r / Y)
_INK = "black"
_LINESTYLES = ("-", "--", "-.", ":")


def _ramp(n: int, lo: float = 0.0, hi: float = 0.72) -> List[str]:
    """``n`` grey shades from dark (``lo``) to light (``hi``)."""
    if n <= 1:
        return [str(lo)]
    return [f"{v:.3f}" for v in np.linspace(lo, hi, n)]


def _line_style(rank: int, total: int) -> Dict[str, object]:
    """A greyscale (colour, linestyle) pair, distinct by both shade and dash."""
    shade = 0.0 if total <= 1 else 0.66 * rank / (total - 1)
    return {"color": f"{shade:.3f}", "linestyle": _LINESTYLES[rank % len(_LINESTYLES)]}


def save(fig, path: str) -> str:
    """Save a figure to ``path`` (extension picks the format) and close it."""
    fig.savefig(path, bbox_inches="tight")
    _mpl().close(fig)
    return path


def _span(iv: IntervalGaussian) -> Tuple[float, float]:
    """Mean boundary span [s, f] of an interval."""
    d = max(iv.mu_d, 0.0)
    return iv.mu_t - d / 2, iv.mu_t + d / 2


def draw_interval_pair(
    ax,
    e_span: Tuple[float, float],
    r_span: Tuple[float, float],
    y_e: float = 0.62,
    y_r: float = 0.28,
    height: float = 0.20,
    label: bool = False,
) -> None:
    """Draw two horizontal interval bars (``e`` filled, ``r`` outlined) on ``ax``."""
    from matplotlib.patches import Rectangle

    se, fe = e_span
    sr, fr = r_span
    ax.add_patch(
        Rectangle((se, y_e - height / 2), fe - se, height, facecolor=_FILL, edgecolor=_INK, lw=0.8)
    )
    ax.add_patch(
        Rectangle((sr, y_r - height / 2), fr - sr, height, facecolor=_OPEN, edgecolor=_INK, lw=0.8)
    )
    if label:
        ax.text(se - 0.15, y_e, "e", ha="right", va="center", fontsize=8)
        ax.text(sr - 0.15, y_r, "r", ha="right", va="center", fontsize=8)


# Prototypical (e_span, r_span) arrangements for each relation.
_PROTOTYPE: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "before": ((0, 2), (3, 5)),
    "meets": ((0, 2), (2, 4)),
    "overlaps": ((0, 3), (2, 5)),
    "starts": ((0, 2), (0, 4)),
    "during": ((1, 3), (0, 5)),
    "finishes": ((2, 4), (0, 4)),
    "equals": ((0, 4), (0, 4)),
    "finished_by": ((0, 4), (2, 4)),
    "contains": ((0, 5), (1, 3)),
    "started_by": ((0, 4), (0, 2)),
    "overlapped_by": ((2, 5), (0, 3)),
    "met_by": ((2, 4), (0, 2)),
    "after": ((3, 5), (0, 2)),
}

_CONDITION = {
    "before": "b_e < a_r",
    "meets": "b_e = a_r",
    "overlaps": "a_e < a_r < b_e < b_r",
    "starts": "a_e = a_r, b_e < b_r",
    "during": "a_r < a_e, b_e < b_r",
    "finishes": "a_r < a_e, b_e = b_r",
    "equals": "a_e = a_r, b_e = b_r",
    "finished_by": "a_e < a_r, b_e = b_r",
    "contains": "a_e < a_r, b_r < b_e",
    "started_by": "a_e = a_r, b_r < b_e",
    "overlapped_by": "a_r < a_e < b_r < b_e",
    "met_by": "a_e = b_r",
    "after": "b_r < a_e",
}


def _iv(span: Tuple[float, float], sigma_t: float = 0.30, sigma_d: float = 0.15) -> IntervalGaussian:
    """An uncertain interval centred on a mean span [s, f]."""
    s, f = span
    return IntervalGaussian((s + f) / 2, sigma_t, f - s, sigma_d)


def _abgh(X: Tuple[float, float], Y: Tuple[float, float]) -> Tuple[float, float, float, float]:
    return Y[0] - X[0], Y[1] - X[1], Y[0] - X[1], X[0] - Y[1]


def _annotate_pair(ax, X, Y, tau, crisp_label=None, xlim=None):
    """Draw X (filled) over Y (outlined), the crisp label, and the top graded probabilities."""
    from .relations import classify_arrangement

    ax.set_xlim(*(xlim if xlim is not None else (-0.7, max(X[1], Y[1]) + 0.7)))
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_interval_pair(ax, X, Y, y_e=0.86, y_r=0.66, height=0.13)
    ax.text(X[0] - 0.25, 0.86, "X", ha="right", va="center", fontsize=8)
    ax.text(Y[0] - 0.25, 0.66, "Y", ha="right", va="center", fontsize=8)
    crisp = crisp_label or classify_arrangement(*_abgh(X, Y), 0.0)
    ax.set_title(f"crisp: {crisp}", fontsize=9.5, family="monospace")
    p = relation_probabilities(_iv(X), _iv(Y), tau=tau)
    top = [(k, v) for k, v in sorted(p.items(), key=lambda kv: -kv[1]) if v > 0.02][:3]
    ax.text(0.5, 0.40, "graded:", transform=ax.transAxes, ha="center", va="top", fontsize=7.5, style="italic")
    # List the top probabilities, bolding the winning (most probable) relation.
    line_h = 0.078
    for i, (k, v) in enumerate(top):
        winner = i == 0
        ax.text(
            0.5, 0.31 - i * line_h, f"P({k})={v:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, family="monospace",
            fontweight="bold" if winner else "normal",
        )


def fig_robustness(tau: float = 0.2):
    """F0a (à la Hourglass Fig. 1): a small boundary shift flips the crisp relation."""
    plt = _mpl()
    X = (0.0, 2.0)
    cfgs = [(2.4, 4.4), (2.0, 4.0), (1.6, 3.6)]  # Y shifted left by 0.4 each
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, Y in zip(axes, cfgs):
        _annotate_pair(ax, X, Y, tau, xlim=(-0.7, 5.1))
    fig.suptitle(
        "Lack of robustness: shifting Y by 0.4 flips the crisp relation, while the probabilities vary smoothly",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_inadequacy(tau: float = 0.2):
    """F0b (à la Hourglass Fig. 2): very different situations all crisply 'overlaps'."""
    plt = _mpl()
    X = (0.0, 3.0)
    cfgs = [(2.6, 5.6), (1.4, 4.4), (0.4, 3.4)]  # almost-before, half, almost-equal
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, Y in zip(axes, cfgs):
        _annotate_pair(ax, X, Y, tau, crisp_label="overlaps", xlim=(-0.7, 6.3))
    fig.suptitle(
        "Inadequacy: three very different configurations, all crisply 'overlaps', distinguished by the probabilities",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_relations_catalogue():
    """F0c: the 13 relations as name | boundary condition | two-bar schematic."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.set_xlim(-3.2, 6.2)
    ax.set_ylim(-0.5, len(RELATION_NAMES) - 0.5)
    ax.axis("off")
    for i, name in enumerate(RELATION_NAMES):
        y = len(RELATION_NAMES) - 1 - i
        ax.text(-3.1, y, name, ha="left", va="center", fontsize=10, fontweight="bold", family="monospace")
        ax.text(-1.2, y, _CONDITION[name], ha="left", va="center", fontsize=8.5, style="italic")
        sub = ax.inset_axes([0.66, (y + 0.15) / len(RELATION_NAMES), 0.32, 0.7 / len(RELATION_NAMES)])
        e_s, r_s = _PROTOTYPE[name]
        sub.set_xlim(-0.5, 5.5)
        sub.set_ylim(0, 1)
        sub.axis("off")
        draw_interval_pair(sub, e_s, r_s)
    ax.set_title("Allen's thirteen relations  (e filled, r outlined)", fontsize=11)
    return fig


def _sweep(
    e: IntervalGaussian, r: IntervalGaussian, offsets: np.ndarray, tau: float
) -> Dict[str, np.ndarray]:
    """Relation probabilities as r is shifted by each offset (r slides relative to e)."""
    out = {n: np.empty(len(offsets)) for n in RELATION_NAMES}
    base = r.mu_t
    for k, off in enumerate(offsets):
        p = relation_probabilities(e, replace(r, mu_t=base + off), tau=tau)
        for n in RELATION_NAMES:
            out[n][k] = p[n]
    return out


def fig_probability_vs_separation(tau: float = 0.3):
    """F3: all 13 relation probabilities as r slides across e; the curves sum to 1.

    Inset cartoons show the interval configuration at representative positions.
    """
    plt = _mpl()
    from .relations import classify_arrangement

    e = IntervalGaussian(0.0, 0.4, 2.0, 0.2)
    r = IntervalGaussian(0.0, 0.4, 5.0, 0.2)
    offsets = np.linspace(-9, 9, 241)
    curves = _sweep(e, r, offsets, tau)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    xs = -offsets  # plot vs mu_e - mu_r so 'before' is on the left and 'after' on the right
    # Order the visible relations by peak position so the grey ramp reads as a
    # left-to-right gradient (before -> after); the line style disambiguates neighbours.
    active = [n for n in RELATION_NAMES if curves[n].max() > 0.02]
    active.sort(key=lambda n: xs[int(np.argmax(curves[n]))])
    for rank, n in enumerate(active):
        ax.plot(xs, curves[n], lw=1.6, label=n, **_line_style(rank, len(active)))
    total = np.sum([curves[n] for n in RELATION_NAMES], axis=0)
    ax.plot(xs, total, color=_INK, lw=1.1, ls=(0, (1, 1)), label="sum (=1)")
    ax.set_xlabel(r"relative position  ($\mu_e - \mu_r$)")
    ax.set_ylabel("relation probability")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-0.03, 1.6)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    ax.set_title(rf"Relation probabilities vs. separation ($\tau={tau}$)")

    # Inset interval configurations at representative x positions.
    de, dr = e.mu_d, r.mu_d
    e_s = (e.mu_t - de / 2, e.mu_t + de / 2)
    for xval in (-6.5, -3.0, 0.0, 3.0, 6.5):
        off = -xval
        r_s = (r.mu_t + off - dr / 2, r.mu_t + off + dr / 2)
        fx = (xval + 9) / 18.0
        ax.axvline(xval, ymin=0.0, ymax=0.79, color="0.8", ls=":", lw=0.7, zorder=0)
        ins = ax.inset_axes([fx - 0.075, 0.80, 0.15, 0.15])
        lo, hi = min(e_s[0], r_s[0]), max(e_s[1], r_s[1])
        ins.set_xlim(lo - 0.6, hi + 0.6)
        ins.set_ylim(0, 1)
        ins.axis("off")
        draw_interval_pair(ins, e_s, r_s, y_e=0.66, y_r=0.34, height=0.26)
        rel = classify_arrangement(
            r_s[0] - e_s[0], r_s[1] - e_s[1], r_s[0] - e_s[1], e_s[0] - r_s[1], tau
        )
        ins.text(0.5, -0.12, rel, transform=ins.transAxes, ha="center", va="top",
                 fontsize=6.5, family="monospace")
    return fig


def fig_case_study(tau: float = 0.4):
    """Running example: an uncertain storm X and power outage Y, and the graded relations.

    Top: the two uncertain intervals (mean span as a bar, boundary uncertainty as
    +/-1 sigma whiskers). Bottom: the thirteen relation probabilities as greyscale
    bars, with the dominant crisp label highlighted.
    """
    plt = _mpl()
    X = IntervalGaussian(2.0, 0.5, 4.0, 0.5)   # storm  ~ [0, 4] h
    Y = IntervalGaussian(3.0, 0.6, 3.0, 0.6)   # outage ~ [1.5, 4.5] h
    probs = relation_probabilities(X, Y, tau=tau)

    fig, (axt, axb) = plt.subplots(2, 1, figsize=(7.6, 5.4), height_ratios=[1.0, 1.5])

    # --- top: the two uncertain intervals ---
    def boundaries(iv):
        a, b = iv.mu_t - iv.mu_d / 2, iv.mu_t + iv.mu_d / 2
        sig = float(np.hypot(iv.sigma_t, iv.sigma_d / 2))
        return a, b, sig

    for iv, y, fc, name in ((X, 1.0, _FILL, "storm  X"), (Y, 0.0, _OPEN, "outage  Y")):
        a, b, sig = boundaries(iv)
        from matplotlib.patches import Rectangle
        axt.add_patch(Rectangle((a, y - 0.16), b - a, 0.32, facecolor=fc, edgecolor=_INK, lw=1.0))
        for x in (a, b):
            axt.plot([x - sig, x + sig], [y, y], color=_INK, lw=1.0)
            for xx in (x - sig, x + sig):
                axt.plot([xx, xx], [y - 0.06, y + 0.06], color=_INK, lw=1.0)
        axt.text(a - sig - 0.25, y, name, ha="right", va="center", fontsize=9)
    axt.set_xlim(-1.4, 6.0)
    axt.set_ylim(-0.6, 1.6)
    axt.set_yticks([])
    axt.set_xlabel("time (h)")
    for s in ("top", "right", "left"):
        axt.spines[s].set_visible(False)
    axt.set_title("Uncertain intervals: mean span (bar) and $\\pm1\\sigma$ boundaries (whiskers)")

    # --- bottom: the thirteen relation probabilities ---
    x = np.arange(len(RELATION_NAMES))
    vals = [probs[n] for n in RELATION_NAMES]
    top = int(np.argmax(vals))
    colors = [_INK if i == top else "0.7" for i in range(len(RELATION_NAMES))]
    axb.bar(x, vals, color=colors, edgecolor=_INK, lw=0.4)
    axb.set_xticks(x)
    axb.set_xticklabels(RELATION_NAMES, rotation=60, ha="right", fontsize=7.5)
    axb.set_ylabel("probability")
    axb.set_title(rf"Relation probabilities ($\tau={tau}$ h); crisp label = "
                  f"{RELATION_NAMES[top]}")
    axb.margins(x=0.01)
    fig.tight_layout()
    return fig


def fig_relation_cascade(tau: float = 0.3):
    """F3b (hero): one row per relation, probability profile + inset arrangement + leader line."""
    plt = _mpl()
    e = IntervalGaussian(0.0, 0.35, 2.0, 0.15)
    r = IntervalGaussian(0.0, 0.35, 5.0, 0.15)
    offsets = np.linspace(-9, 9, 281)
    curves = _sweep(e, r, offsets, tau)
    # Keep relations with a visible bump, ordered by peak position.
    active = [n for n in RELATION_NAMES if curves[n].max() > 0.05]
    active.sort(key=lambda n: offsets[int(np.argmax(curves[n]))])
    k = len(active)
    fig, axes = plt.subplots(k, 1, figsize=(7.8, 1.05 * k), sharex=True)
    for row, (ax, n) in enumerate(zip(axes, active)):
        ax.fill_between(offsets, curves[n], color="0.55", edgecolor=_INK, lw=0.6)
        peak = int(np.argmax(curves[n]))
        xp, yp = offsets[peak], curves[n][peak]
        ax.plot([xp], [yp], "o", color=_INK, ms=3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=8, family="monospace")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        # Inset arrangement at this relation's peak, with a leader line to the peak.
        ins = ax.inset_axes([0.80, 0.18, 0.18, 0.75])
        ins.set_xlim(-0.5, 5.5)
        ins.set_ylim(0, 1)
        ins.axis("off")
        e_s, r_s = _PROTOTYPE[n]
        draw_interval_pair(ins, e_s, r_s)
        ax.annotate(
            PRETTY[n],
            xy=(xp, yp),
            xytext=(0.795, 0.5),
            textcoords=ax.transAxes,
            fontsize=7.5,
            va="center",
            ha="right",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"),
        )
    axes[-1].set_xlabel(r"relative position of r  ($\mu_r - \mu_e$)")
    axes[0].set_title(rf"Relation cascade: probability of each relation as r slides ($\tau={tau}$)")
    fig.tight_layout(h_pad=0.2)
    return fig


def fig_partition_sum_vs_tau():
    """F4: the 13 probabilities sum to 1 for every tau (true partition)."""
    plt = _mpl()
    e = IntervalGaussian(0.0, 1.0, 2.0, 0.3)
    r = IntervalGaussian(1.5, 1.0, 1.5, 0.4)
    taus = np.linspace(0.0, 1.5, 31)
    sums = np.array([sum(relation_probabilities(e, r, tau=float(t)).values()) for t in taus])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(taus, sums, "o-", color=_INK, ms=3, lw=1.4)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel(r"tolerance $\tau$")
    ax.set_ylabel("sum of the 13 probabilities")
    ax.set_ylim(0.98, 1.02)
    ax.set_title("Partition: the relations sum to 1 for any tolerance")
    return fig


def fig_tau_explainer(tau: float = 1.0):
    """Fτ: what the margin tau does, via the meets contact band on the gap axis G."""
    plt = _mpl()
    fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(7.2, 4.2), height_ratios=[1.1, 1])

    # Top: three sample arrangements (separated, touching, overlapping).
    ax_top.set_xlim(-0.5, 14)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    samples = [((0, 2), (3.0, 5.0), "G > tau"), ((4.6, 6.6), (6.9, 8.9), "|G| <= tau"), ((9.4, 11.6), (11.0, 13.0), "G < -tau")]
    for (e_s, r_s, lab) in samples:
        draw_interval_pair(ax_top, e_s, r_s)
        ax_top.text((e_s[0] + r_s[1]) / 2, 0.93, lab, ha="center", fontsize=8, family="monospace")

    # Bottom: the gap axis G with the three coloured regions and the meets band of half-width tau.
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)
    ax.axvspan(tau, 3, facecolor="0.9", label="before  (G > tau)")
    ax.axvspan(-tau, tau, facecolor="0.62", hatch="///", edgecolor="white", label=r"meets  ($|G|\leq\tau$)")
    ax.axvspan(-3, -tau, facecolor="0.78", label="overlaps  (G < -tau)")
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.annotate("", xy=(tau, 0.5), xytext=(-tau, 0.5), arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(0, 0.58, r"$2\tau$", ha="center", fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel(r"frontal gap  $G = a_r - b_e$")
    ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.25))
    ax.set_title(r"The margin $\tau$ turns the contact relation 'meets' into a band of half-width $\tau$")
    fig.tight_layout()
    return fig


def _truncnorm_durations(mu_d: float, sigma_d: float, n: int, rng) -> np.ndarray:
    if sigma_d == 0.0:
        return np.full(n, mu_d)
    from scipy.stats import truncnorm

    a = (0.0 - mu_d) / sigma_d
    return truncnorm.rvs(a, np.inf, loc=mu_d, scale=sigma_d, size=n, random_state=rng)


def fig_mc_parity(tau: float = 0.3, n: int = 400_000):
    """F8: analytic vs Monte-Carlo parity plot across the 13 relations."""
    plt = _mpl()
    e = IntervalGaussian(0.0, 1.0, 2.0, 0.3)
    r = IntervalGaussian(3.0, 1.0, 1.5, 0.4)
    analytic = relation_probabilities(e, r, tau=tau)
    rng = np.random.default_rng(0)
    te = rng.normal(e.mu_t, e.sigma_t, n)
    tr = rng.normal(r.mu_t, r.sigma_t, n)
    de = _truncnorm_durations(e.mu_d, e.sigma_d, n, rng)
    dr = _truncnorm_durations(r.mu_d, r.sigma_d, n, rng)
    A, B = (tr - dr / 2) - (te - de / 2), (tr + dr / 2) - (te + de / 2)
    G, H = (tr - dr / 2) - (te + de / 2), (te - de / 2) - (tr + dr / 2)

    mc = {nme: 0 for nme in RELATION_NAMES}
    # Vectorised classification via per-sample call would be slow; bucket with numpy.
    sa = np.where(A > tau, 1, np.where(A < -tau, -1, 0))
    sb = np.where(B > tau, 1, np.where(B < -tau, -1, 0))
    lbl = np.empty(n, dtype=object)
    lbl[(sa == 1) & (sb == 0)] = "finished_by"
    lbl[(sa == 1) & (sb == -1)] = "contains"
    lbl[(sa == 0) & (sb == 1)] = "starts"
    lbl[(sa == 0) & (sb == 0)] = "equals"
    lbl[(sa == 0) & (sb == -1)] = "started_by"
    lbl[(sa == -1) & (sb == 1)] = "during"
    lbl[(sa == -1) & (sb == 0)] = "finishes"
    pp = (sa == 1) & (sb == 1)
    lbl[pp & (G < -tau)] = "overlaps"
    lbl[pp & (np.abs(G) <= tau)] = "meets"
    lbl[pp & (G > tau)] = "before"
    mm = (sa == -1) & (sb == -1)
    lbl[mm & (H < -tau)] = "overlapped_by"
    lbl[mm & (np.abs(H) <= tau)] = "met_by"
    lbl[mm & (H > tau)] = "after"
    mc = {nme: float((lbl == nme).mean()) for nme in RELATION_NAMES}

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    xs = [analytic[nme] for nme in RELATION_NAMES]
    ys = [mc[nme] for nme in RELATION_NAMES]
    ax.plot([0, 1], [0, 1], "0.6", lw=1.0, ls="--")
    ax.scatter(xs, ys, s=24, facecolor="0.3", edgecolor=_INK, lw=0.6, zorder=3)
    lim = max(max(xs), max(ys)) * 1.1
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("analytic probability")
    ax.set_ylabel(f"Monte-Carlo frequency  (n={n:,})")
    ax.set_title("Analytic vs. Monte-Carlo")
    ax.set_aspect("equal")
    return fig


def fig_limiting_transition(tau: float = 0.3):
    """F5: interval -> point. As e's duration shrinks, the 13 relations reduce to 5."""
    plt = _mpl()
    mu_ds = np.linspace(3.0, 0.0, 31)
    survivors = ["before", "starts", "during", "finishes", "after"]
    # The reference carries duration uncertainty (sigma_d = 0.3): this smooths the
    # equals -> during handover, so the starts/finishes curves bend rather than kink.
    # The event keeps a sharp duration (sigma_d = 0), so it reduces to an exact point.
    r = IntervalGaussian(0.0, 0.4, 3.0, 0.3)
    # Place the event's centre one-third of the way along r (not at its midpoint),
    # so the before/after and starts/finishes curves separate instead of coinciding.
    mu_e = -r.mu_d / 6.0   # = r_start + dr/3 = -0.5 for dr = 3
    curves = {n: [] for n in RELATION_NAMES}
    for md in mu_ds:
        e = IntervalGaussian(mu_e, 1.2, float(md), 0.0)
        p = relation_probabilities(e, r, tau=tau)
        for n in RELATION_NAMES:
            curves[n].append(p[n])
    # The five surviving point-interval relations are drawn as bold SOLID curves,
    # distinguished by shade and marker; the eight vanishing relations are faint
    # dotted, so "solid vs dotted" reads as "survives vs vanishes".
    markers = ["o", "s", "^", "D", "v"]
    shades = _ramp(len(survivors), lo=0.0, hi=0.6)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(top=0.90, bottom=0.20)
    for n in RELATION_NAMES:
        arr = np.array(curves[n])
        if arr.max() > 0.02:
            if n in survivors:
                i = survivors.index(n)
                ax.plot(mu_ds, arr, lw=2.0, color=shades[i], linestyle="-",
                        marker=markers[i], markevery=5, markersize=5, label=n)
            else:
                ax.plot(mu_ds, arr, ":", lw=1.1, color="0.7", label=n)
    ax.set_xlabel(r"event duration  $\mu_{d_e}$  (e shrinks to a point $\to$ 0)")
    ax.set_ylabel("relation probability")
    ax.set_ylim(-0.03, 1.42)            # headroom for the constellation insets
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # Pin x-limits (no auto-margin) with a little padding, so each inset can be
    # centred exactly on its mu_d via the same linear map the data uses.
    x_left, x_right = 3.3, -0.3
    ax.set_xlim(x_left, x_right)
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Interval $\\to$ point: 13 relations reduce to 5 (bold solid); "
                 "the other 8 vanish (faint dotted)")

    # Constellation insets (cf. Figure~\\ref{fig:sweep}): the event e (filled, with
    # faint ghost copies for its positional uncertainty +/- sigma_e) shrinks against
    # the fixed reference r as mu_d -> 0.  Dotted guides tie each to the x-axis.
    from matplotlib.patches import Rectangle

    dr = r.mu_d
    r_span = (-dr / 2.0, dr / 2.0)
    sigma_e = e.sigma_t                 # positional uncertainty of the event
    for md in (3.0, 2.0, 1.0, 0.0):
        fx = (md - x_left) / (x_right - x_left)   # axes-fraction of this mu_d (matches the guide)
        ax.axvline(md, ymin=0.0, ymax=0.72, color="0.8", ls=":", lw=0.7, zorder=0)
        ins = ax.inset_axes([fx - 0.085, 0.76, 0.17, 0.15])
        ins.set_xlim(-3.5, 2.5)
        ins.set_ylim(0, 1)
        ins.axis("off")
        # faint ghost copies of e at mu_e +/- sigma_e show that its position is
        # uncertain, so the same duration can realise before / starts / during /
        # finishes / after
        for gx in (mu_e - sigma_e, mu_e + sigma_e):
            if md > 0.25:
                ins.add_patch(Rectangle((gx - md / 2, 0.49), md, 0.26,
                                        facecolor=_FILL, edgecolor="none", alpha=0.28))
            else:
                ins.plot([gx], [0.62], "s", color=_FILL, ms=4, alpha=0.3)
        if md > 0.25:
            draw_interval_pair(ins, (mu_e - md / 2, mu_e + md / 2), r_span,
                               y_e=0.62, y_r=0.30, height=0.26, label=(md == 3.0))
        else:                           # event has collapsed to a point
            ins.add_patch(Rectangle((r_span[0], 0.17), dr, 0.26,
                                    facecolor=_OPEN, edgecolor=_INK, lw=0.8))
            ins.plot([mu_e], [0.62], "s", color=_FILL, ms=6)
        ins.set_title(rf"$\mu_{{d_e}}={md:g}$", fontsize=7, pad=1)
    return fig


def fig_scale_invariance(tau: float = 0.25):
    """F6: relation probabilities are unchanged when all scales are multiplied together."""
    plt = _mpl()
    e = IntervalGaussian(0.0, 1.0, 2.0, 0.3)
    r = IntervalGaussian(1.5, 1.0, 1.5, 0.4)
    factors = [0.5, 1.0, 4.0, 20.0]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(RELATION_NAMES))
    width = 0.2
    shades = _ramp(len(factors), lo=0.15, hi=0.8)
    for i, c in enumerate(factors):
        es = replace(e, mu_t=e.mu_t * c, sigma_t=e.sigma_t * c, mu_d=e.mu_d * c, sigma_d=e.sigma_d * c)
        rs = replace(r, mu_t=r.mu_t * c, sigma_t=r.sigma_t * c, mu_d=r.mu_d * c, sigma_d=r.sigma_d * c)
        p = relation_probabilities(es, rs, tau=tau * c)
        ax.bar(x + (i - 1.5) * width, [p[n] for n in RELATION_NAMES], width,
               color=shades[i], edgecolor=_INK, lw=0.4, label=f"x{c:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(RELATION_NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("probability")
    ax.legend(title="scale factor", fontsize=8)
    ax.set_title("Scale invariance: identical bars across scale factors")
    fig.tight_layout()
    return fig


def _size_to_durations(s_hat: float, total: float) -> Tuple[float, float]:
    """Map a hourglass size contrast s_hat in (-1, 1) to durations (d_X, d_Y)."""
    return total * (1 + s_hat) / 2, total * (1 - s_hat) / 2


def fig_hourglass_heatmaps(tau: float = 0.2):
    """F7: probability of four relations over the hourglass (position, size) plane."""
    plt = _mpl()
    rels = ["before", "overlaps", "during", "contains"]
    n = 48
    total, st, sd = 3.0, 0.4, 0.2
    deltas = np.linspace(-6, 6, n)
    shats = np.linspace(-0.9, 0.9, n)
    grids = {r: np.zeros((n, n)) for r in rels}
    for i, s in enumerate(shats):
        dX, dY = _size_to_durations(s, total)
        for j, dl in enumerate(deltas):
            p = relation_probabilities(
                IntervalGaussian(0.0, st, dX, sd), IntervalGaussian(dl, st, dY, sd), tau=tau
            )
            for r in rels:
                grids[r][i, j] = p[r]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), sharex=True, sharey=True)
    im = None
    for ax, r in zip(axes.flat, rels):
        im = ax.imshow(
            grids[r], origin="lower", extent=[deltas[0], deltas[-1], shats[0], shats[-1]],
            aspect="auto", cmap="Greys", vmin=0, vmax=1,
        )
        ax.set_title(f"$P(\\mathtt{{{r}}})$", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel(r"relative position $\mu_Y-\mu_X$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"relative size $\hat s$")
    fig.colorbar(im, ax=axes, shrink=0.8, label="probability")
    fig.suptitle("Relation probabilities over the hourglass plane", fontsize=11)
    return fig


def fig_phase_diagram(tau: float = 0.15):
    """F9: the relation map over the hourglass plane, beside the 13->5->3 reduction ladder."""
    plt = _mpl()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from .decode import map_relation

    # Left: categorical relation map (the hourglass phase diagram from our model).
    n = 220
    total, st, sd = 3.0, 0.4, 0.2
    deltas = np.linspace(-6, 6, n)
    shats = np.linspace(-0.95, 0.95, n)
    idx = {name: k for k, name in enumerate(RELATION_NAMES)}
    grid = np.zeros((n, n), dtype=int)
    for i, s in enumerate(shats):
        dX, dY = _size_to_durations(s, total)
        for j, dl in enumerate(deltas):
            rel = map_relation(IntervalGaussian(0.0, st, dX, sd), IntervalGaussian(dl, st, dY, sd), tau=tau)
            grid[i, j] = idx[rel]
    # A monotone grey ramp ordered by relation index reads as a before(dark)->after(light)
    # gradient; drawn boundaries keep neighbouring greys separable in print.
    cmap = ListedColormap(_ramp(13, lo=0.08, hi=0.93))
    grid = grid[:, ::-1]  # reverse x so 'before' is on the left and 'after' on the right

    fig = plt.figure(figsize=(11.0, 4.8))
    axm = fig.add_axes([0.06, 0.12, 0.42, 0.78])
    axm.imshow(grid, origin="lower", extent=[deltas[0], deltas[-1], shats[0], shats[-1]],
               aspect="auto", cmap=cmap, vmin=-0.5, vmax=12.5, interpolation="nearest")
    gx = np.linspace(deltas[0], deltas[-1], n)
    gy = np.linspace(shats[0], shats[-1], n)
    axm.contour(gx, gy, grid, levels=np.arange(0.5, 12.5, 1.0), colors="black", linewidths=0.4)
    axm.set_xlabel(r"relative position $\mu_X-\mu_Y$")
    axm.set_ylabel(r"relative size $\hat s=(d_X-d_Y)/(d_X+d_Y)$")
    axm.set_title("Most-probable relation over the hourglass plane")
    present = sorted(set(grid.flat))
    axm.legend(
        handles=[Patch(facecolor=cmap(k), edgecolor="black", lw=0.4, label=RELATION_NAMES[k]) for k in present],
        fontsize=6.5, ncol=2, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False,
    )

    # Right: the 13 -> 5 -> 3 reduction ladder.
    axr = fig.add_axes([0.70, 0.12, 0.28, 0.78])
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.axis("off")
    levels = [
        (0.86, "interval–interval", "13 relations"),
        (0.52, "point–interval", "5: before, starts,\nduring, finishes, after"),
        (0.18, "point–point", "3: before, equals, after"),
    ]
    for y, head, body in levels:
        axr.text(0.5, y + 0.06, head, ha="center", va="bottom", fontsize=10, fontweight="bold")
        axr.text(0.5, y - 0.005, body, ha="center", va="top", fontsize=8.5)
    for y0, y1, lab in [(0.74, 0.64, r"$d_X\to0$"), (0.40, 0.30, r"$d_Y\to0$")]:
        axr.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.3"))
        axr.text(0.56, (y0 + y1) / 2, lab, ha="left", va="center", fontsize=8)
    axr.set_title("Dimensional reduction")
    return fig


def generate_all(outdir: str, ext: str = "svg") -> List[str]:
    """Generate every figure into ``outdir`` and return the written paths."""
    import os

    os.makedirs(outdir, exist_ok=True)
    jobs = {
        "robustness": fig_robustness,
        "inadequacy": fig_inadequacy,
        "case_study": fig_case_study,
        "catalogue": fig_relations_catalogue,
        "probability_vs_separation": fig_probability_vs_separation,
        "cascade": fig_relation_cascade,
        "partition_sum": fig_partition_sum_vs_tau,
        "tau_explainer": fig_tau_explainer,
        "mc_parity": fig_mc_parity,
        "limiting_transition": fig_limiting_transition,
        "scale_invariance": fig_scale_invariance,
        "hourglass_heatmaps": fig_hourglass_heatmaps,
        "phase_diagram": fig_phase_diagram,
    }
    paths = []
    for name, fn in jobs.items():
        paths.append(save(fn(), os.path.join(outdir, f"{name}.{ext}")))
    return paths
