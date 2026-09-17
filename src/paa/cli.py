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

"""Command-line interface to the probabilistic Allen algebra.

Exposes the core computational surface of :mod:`paa` as subcommands, reachable
both through the ``paa`` console script and ``python -m paa``::

    paa prob       --e SPEC --r SPEC [--tau T]      # the 13 leaf probabilities
    paa coarse     --e SPEC --r SPEC [--tau T]      # taxonomy-node probabilities
    paa decode     --e SPEC --r SPEC [--tau T]      # one "best" relation
    paa primitives --e SPEC --r SPEC [--tau T]      # the 4 A/B/G/H soft primitives
    paa refine     --e SPEC --r SPEC --parent NODE  # P(leaf | node)
    paa explain    RELATION                         # static reference for a relation
    paa list                                        # the 13 relations + taxonomy
    paa figures    [--out DIR --ext pdf]            # regenerate the paper figures
    paa demo                                        # the fixed demonstration

Intervals are given by a compact spec token that maps 1:1 to the constructors of
:mod:`paa.intervals`::

    point:MU[,SIGMA]                 a (possibly uncertain) time point
    mid:MU_T,SIGMA_T,MU_D,SIGMA_D    IntervalGaussian (midpoint + duration)
    start:MU_S,SIGMA_S,MU_D,SIGMA_D  IntervalGaussian.from_start (start + duration)
    end:MU_F,SIGMA_F,MU_D,SIGMA_D    IntervalGaussian.from_end (end + duration)
    endpoints:MU_S,SIGMA_S,MU_E,SIGMA_E  IntervalGaussian.from_endpoints (start + end)

Every data subcommand accepts ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .intervals import IntervalGaussian, point
from .relations import (
    CONVERSE,
    RELATION_NAMES,
    THICK_RELATIONS,
    relation_probabilities,
)
from .taxonomy import TREE, VIEWS, coarse_predicates, refine
from .decode import hierarchical_decode, map_relation, most_probable_relation
from .primitives import (
    CANONICAL_SIGNS,
    PRIMITIVE_POINTS,
    contacts,
    decompose,
    primitive_probabilities,
)

__all__ = ["main", "parse_interval"]


# ---------------------------------------------------------------------------
# Interval spec parsing
# ---------------------------------------------------------------------------

_ARITY = {"point": (1, 2), "mid": (4, 4), "start": (4, 4), "end": (4, 4), "endpoints": (4, 4)}


def parse_interval(spec: str) -> IntervalGaussian:
    """Build an :class:`IntervalGaussian` from a ``kind:n,n,...`` spec token.

    Parameters
    ----------
    spec:
        One of ``point:MU[,SIGMA]``, ``mid:MU_T,SIGMA_T,MU_D,SIGMA_D``,
        ``start:MU_S,SIGMA_S,MU_D,SIGMA_D`` or ``end:MU_F,SIGMA_F,MU_D,SIGMA_D``.

    Raises
    ------
    ValueError
        If the kind is unknown, the numbers do not parse, or the count is wrong.
    """
    kind, sep, rest = spec.partition(":")
    if not sep or kind not in _ARITY:
        raise ValueError(
            f"bad interval spec {spec!r}: expected one of "
            "point:/mid:/start:/end:/endpoints: followed by comma-separated numbers"
        )
    try:
        nums = [float(x) for x in rest.split(",")] if rest else []
    except ValueError as exc:
        raise ValueError(f"bad interval spec {spec!r}: {exc}") from exc
    lo, hi = _ARITY[kind]
    if not lo <= len(nums) <= hi:
        want = f"{lo}" if lo == hi else f"{lo}-{hi}"
        raise ValueError(
            f"bad interval spec {spec!r}: {kind}: takes {want} numbers, got {len(nums)}"
        )
    if kind == "point":
        return point(nums[0], nums[1] if len(nums) > 1 else 0.0)
    if kind == "mid":
        return IntervalGaussian(*nums)
    if kind == "start":
        return IntervalGaussian.from_start(*nums)
    if kind == "endpoints":
        return IntervalGaussian.from_endpoints(*nums)
    return IntervalGaussian.from_end(*nums)


def _pair(args: argparse.Namespace):
    """Parse the ``--e``/``--r`` interval specs, erroring cleanly on failure."""
    try:
        return parse_interval(args.e), parse_interval(args.r)
    except ValueError as exc:
        raise SystemExit(f"paa: error: {exc}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _emit(obj, as_json: bool, human) -> None:
    """Print ``obj`` as JSON, or defer to ``human()`` for the table form."""
    if as_json:
        print(json.dumps(obj, indent=2))
    else:
        human()


def _table(rows, gap: int = 2) -> None:
    """Print ``(label, value)`` rows as an aligned two-column table."""
    width = max((len(str(k)) for k, _ in rows), default=0)
    for k, v in rows:
        val = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"  {str(k):<{width}}{' ' * gap}{val}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prob(args: argparse.Namespace) -> None:
    e, r = _pair(args)
    probs = relation_probabilities(e, r, tau=args.tau, eps=args.eps)
    payload = {"tau": args.tau, "probabilities": probs, "sum": sum(probs.values())}

    def human() -> None:
        print(f"Probabilistic Allen relations  (e vs r),  tau = {args.tau}")
        _table([(n, probs[n]) for n in RELATION_NAMES] + [("sum", sum(probs.values()))])

    _emit(payload, args.json, human)


def cmd_coarse(args: argparse.Namespace) -> None:
    e, r = _pair(args)
    probs = relation_probabilities(e, r, tau=args.tau, eps=args.eps)
    coarse = coarse_predicates(probs)

    def human() -> None:
        print(f"Coarse predicates  (e vs r),  tau = {args.tau}")
        print("  taxonomy nodes (children of a node sum to it):")
        _table([(n, coarse[n]) for n in TREE])
        print("  overlapping views (not a partition):")
        _table([(n, coarse[n]) for n in list(VIEWS) + ["containment_either"]])

    _emit({"tau": args.tau, "coarse": coarse}, args.json, human)


def cmd_decode(args: argparse.Namespace) -> None:
    e, r = _pair(args)
    probs = relation_probabilities(e, r, tau=args.tau, eps=args.eps)
    restrict = THICK_RELATIONS if args.restrict == "thick" else None

    if args.method == "hierarchical":
        path = hierarchical_decode(probs)
        result = {"method": "hierarchical", "path": [{"node": n, "p": p} for n, p in path]}

        def human() -> None:
            print("  " + "  ->  ".join(f"{n} ({p:.3f})" for n, p in path))
    elif args.method == "argmax":
        name, p = most_probable_relation(probs, restrict=restrict)
        result = {"method": "argmax", "restrict": args.restrict, "relation": name, "p": p}

        def human() -> None:
            print(f"  argmax P(R) over {args.restrict}:  {name}  ({p:.3f})")
    else:  # map
        name = map_relation(e, r, tau=args.tau)
        result = {"method": "map", "relation": name}

        def human() -> None:
            print(f"  relation of the most probable arrangement:  {name}")

    _emit(result, args.json, human)


def cmd_primitives(args: argparse.Namespace) -> None:
    e, r = _pair(args)
    prims = primitive_probabilities(e, r, tau=args.tau)

    def human() -> None:
        print(f"Temporal primitives  (e vs r),  tau = {args.tau}")
        for prim, dist in prims.items():
            first, second = PRIMITIVE_POINTS[prim]
            states = "  ".join(f"{k} {v:.2f}" for k, v in dist.items())
            print(f"  {prim}  {first:>8s} vs {second:<8s}:  {states}")

    _emit({"tau": args.tau, "primitives": prims}, args.json, human)


def cmd_refine(args: argparse.Namespace) -> None:
    e, r = _pair(args)
    if args.parent not in TREE and args.parent not in VIEWS:
        nodes = ", ".join(list(TREE) + list(VIEWS))
        raise SystemExit(f"paa: error: unknown --parent {args.parent!r}. Choose one of: {nodes}")
    probs = relation_probabilities(e, r, tau=args.tau, eps=args.eps)
    cond = refine(probs, args.parent)

    def human() -> None:
        if not cond:
            print(f"  P({args.parent}) is ~0; no conditional distribution.")
            return
        print(f"Conditional leaf distribution  P(leaf | {args.parent}),  tau = {args.tau}")
        _table(list(cond.items()))

    _emit({"tau": args.tau, "parent": args.parent, "conditional": cond}, args.json, human)


def cmd_explain(args: argparse.Namespace) -> None:
    name = args.relation
    if name not in CANONICAL_SIGNS:
        raise SystemExit(
            f"paa: error: unknown relation {name!r}. One of: {', '.join(RELATION_NAMES)}"
        )
    parts = decompose(name)
    signs = CANONICAL_SIGNS[name]
    kind = "thick (strict)" if name in THICK_RELATIONS else "contact"
    payload = {
        "relation": name,
        "type": kind,
        "converse": CONVERSE[name],
        "contact_primitives": list(contacts(name)),
        "signs": dict(zip(("A", "B", "G", "H"), signs)),
        "decomposition": parts,
    }

    def human() -> None:
        print(f"{name}(X, Y)  <=>")
        for prim in ("A", "B", "G", "H"):
            print(f"    {prim}:  {parts[prim]}")
        con = contacts(name)
        print(f"  type: {kind}    converse: {CONVERSE[name]}")
        print(f"  contact primitives: {', '.join(con) if con else '(none)'}")
        print(f"  signs (A,B,G,H): {signs}")

    _emit(payload, args.json, human)


def cmd_list(args: argparse.Namespace) -> None:
    rows = [
        {"relation": n, "converse": CONVERSE[n],
         "type": "thick" if n in THICK_RELATIONS else "contact"}
        for n in RELATION_NAMES
    ]

    def human() -> None:
        print("The thirteen Allen relations:")
        for row in rows:
            print(f"  {row['relation']:14s} converse={row['converse']:14s} {row['type']}")
        print("\nTaxonomy (strict partition tree):")
        for node, children in TREE.items():
            print(f"  {node:16s} -> {', '.join(children)}")
        print("Overlapping views (not a partition):")
        for view, leaves in VIEWS.items():
            print(f"  {view:16s} = {', '.join(leaves)}")

    _emit({"relations": rows, "tree": TREE, "views": {k: list(v) for k, v in VIEWS.items()}},
          args.json, human)


def cmd_figures(args: argparse.Namespace) -> None:
    try:
        from .plotting import generate_all
    except Exception as exc:  # matplotlib missing, etc.
        raise SystemExit(
            f"paa: error: figures need the 'viz' extra (matplotlib): {exc}\n"
            "  install with:  uv sync --extra viz   (or  pip install 'probabilistic-allen-algebra[viz]')"
        )
    paths = generate_all(args.out, ext=args.ext)
    print(f"wrote {len(paths)} {args.ext.upper()} figures to {args.out}")


def cmd_demo(args: argparse.Namespace) -> None:
    from . import main as _demo
    _demo()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SPEC_HELP = (
    "interval spec: point:MU[,SIGMA] | mid:MU_T,SIGMA_T,MU_D,SIGMA_D | "
    "start:MU_S,SIGMA_S,MU_D,SIGMA_D | end:MU_F,SIGMA_F,MU_D,SIGMA_D"
)


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    p = argparse.ArgumentParser(
        prog="paa",
        description="Probabilistic Allen algebra: Allen's interval relations as probabilities.",
    )
    p.add_argument("--version", action="version", version=f"paa {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # shared parents
    io = argparse.ArgumentParser(add_help=False)
    io.add_argument("--json", action="store_true", help="machine-readable JSON output")

    pair = argparse.ArgumentParser(add_help=False, parents=[io])
    pair.add_argument("--e", required=True, metavar="SPEC", help=f"event interval ({_SPEC_HELP})")
    pair.add_argument("--r", required=True, metavar="SPEC", help="reference interval (same syntax)")
    pair.add_argument("--tau", type=float, default=0.0, help="tolerance tau >= 0 (default 0)")
    pair.add_argument("--eps", type=float, default=1e-12, help=argparse.SUPPRESS)

    s = sub.add_parser("prob", parents=[pair], help="the 13 leaf relation probabilities")
    s.set_defaults(func=cmd_prob)

    s = sub.add_parser("coarse", parents=[pair], help="taxonomy-node (coarse-predicate) probabilities")
    s.set_defaults(func=cmd_coarse)

    s = sub.add_parser("decode", parents=[pair], help="collapse to one 'best' relation")
    s.add_argument("--method", choices=["hierarchical", "argmax", "map"],
                   default="hierarchical", help="selection rule (default hierarchical)")
    s.add_argument("--restrict", choices=["all", "thick"], default="all",
                   help="for --method argmax: compare all 13 or only the 6 thick relations")
    s.set_defaults(func=cmd_decode)

    s = sub.add_parser("primitives", parents=[pair], help="the 4 soft A/B/G/H temporal primitives")
    s.set_defaults(func=cmd_primitives)

    s = sub.add_parser("refine", parents=[pair], help="conditional leaf distribution P(leaf | node)")
    s.add_argument("--parent", required=True, metavar="NODE",
                   help="taxonomy node or view to condition on (see 'paa list')")
    s.set_defaults(func=cmd_refine)

    s = sub.add_parser("explain", parents=[io], help="static reference for one relation (no intervals)")
    s.add_argument("relation", help="relation name, e.g. meets")
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("list", parents=[io], help="list the 13 relations and the taxonomy")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("figures", help="regenerate the paper figures (needs the 'viz' extra)")
    s.add_argument("--out", default="docs/figures", help="output directory (default docs/figures)")
    s.add_argument("--ext", choices=["svg", "pdf", "png"], default="pdf", help="file format")
    s.set_defaults(func=cmd_figures)

    s = sub.add_parser("demo", help="print a fixed demonstration")
    s.set_defaults(func=cmd_demo)

    return p


def main(argv: List[str] | None = None) -> None:
    """Entry point for the ``paa`` console script and ``python -m paa``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
