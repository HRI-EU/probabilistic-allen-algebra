<div align="center">

# The Probabilistic Allen Algebra

*a generative, complete probabilistic extension of Allen's interval relations*

`pip install probabilistic-allen-algebra` &nbsp;·&nbsp; `import paa`

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

</div>

**The probabilistic Allen algebra** computes Allen's thirteen interval relations (`before`,
`meets`, `overlaps`, `starts`, `during`, `finishes`, `equals`, and their converses) as
**probabilities** induced by uncertainty over the underlying temporal quantities — *derived*,
not assigned as primitives.

Each interval is a Gaussian midpoint plus a non-negative (truncated-Gaussian) duration, and
every relation is a conditional **multivariate-Gaussian orthant probability** over one common
latent space. With a single tolerance `tau >= 0`, the thirteen relations form a **true
partition** (they sum to 1 for any `tau`) and recover crisp Allen as `tau -> 0`.

> This package is the **core algebra**: pure `numpy`/`scipy`, no data and no network. The
> companion papers, their figure/fit scripts, and the LLM data-generation tooling live in the
> [`paa-papers`](https://github.com/julegg22/paa-papers) repository.

## Contents

- [Why](#why) · [Install](#install) · [Quickstart](#quickstart) · [Core ideas](#core-ideas)
- [Selecting a best-fitting relation](#selecting-a-best-fitting-relation) · [Command line](#command-line)
- [Guarantees](#guarantees) · [API](#api-at-a-glance) · [License](#license)

## Why

Crisp Allen relations are brittle: a tiny perturbation of an interval boundary flips `before`
into `meets` or `during` into `starts`, and continuous uncertainty makes every contact
relation a probability-zero event. This package derives relation **probabilities** from a
generative model of the uncertain endpoints, so that

- relations are graded and robust to boundary noise,
- coarse predicates (`precede` / `overlap` / `contain`) are **sums of leaf probabilities**
  (a taxonomy, not a flat 13-way vote), and
- the construction is **scale-invariant** and reduces consistently as intervals collapse to
  points (`13 -> 5 -> 3` relations).

## Install

```bash
pip install probabilistic-allen-algebra
```

Or from source:

```bash
git clone https://github.com/HRI-EU/probabilistic-allen-algebra
cd probabilistic-allen-algebra
uv sync
```

Requires Python ≥ 3.11, `numpy`, and `scipy`. Optional extra: `viz` (matplotlib) for the
built-in figure generators (`pip install "probabilistic-allen-algebra[viz]"`).

## Quickstart

```python
from paa import IntervalGaussian, relation_probabilities

# I = [t - d/2, t + d/2],  t ~ N(mu_t, sigma_t^2),  d = D|D>=0, D ~ N(mu_d, sigma_d^2)
e = IntervalGaussian(mu_t=0.0, sigma_t=1.0, mu_d=2.0, sigma_d=0.3)   # event interval
r = IntervalGaussian(mu_t=3.0, sigma_t=1.0, mu_d=1.5, sigma_d=0.4)   # reference interval

p = relation_probabilities(e, r, tau=0.25)
print(p["before"], p["meets"], p["overlaps"])
print(sum(p.values()))        # -> 1.0   (true partition)
```

A time **point** is the degenerate interval: `paa.point(mu_t, sigma_t)`.

The midpoint is a convention, not a requirement. The same class is built from whichever pair
of quantities the data actually gives, and the induced midpoint–duration correlation is carried
along, so the relation probabilities reflect where the uncertainty really sits:

```python
IntervalGaussian.from_start(mu_s, sigma_s, mu_d, sigma_d)       # known onset, uncertain length
IntervalGaussian.from_end(mu_e, sigma_e, mu_d, sigma_d)         # deadline, uncertain length
IntervalGaussian.from_endpoints(mu_s, sigma_s, mu_e, sigma_e)   # two independently uncertain ends
```

`from_endpoints` is the form a `start`/`end` pair of timestamps produces; with equal uncertainty
on both ends it reduces to the plain midpoint form.

## Core ideas

| concept | in code |
|---|---|
| uncertain interval / point | `IntervalGaussian(...)`, `point(...)` |
| the 13 relation probabilities | `relation_probabilities(e, r, tau)` |
| taxonomy (coarse = sum of leaves) | `coarse_predicates(p)`, `refine(p, "contained_in_r")` |
| best-fitting relation | `hierarchical_decode(p)`, `map_relation(e, r)`, `most_probable_relation(p)` |

```python
from paa import coarse_predicates, refine

c = coarse_predicates(p)
c["precede"]          # before + meets
c["contained_in_r"]   # starts + during + finishes + equals   (e inside r)
refine(p, "contained_in_r")    # P(leaf | contained_in_r)
```

## Selecting a best-fitting relation

A flat `argmax` over the 13 probabilities is **biased by relation measure**: the six *thick*
relations are full-dimensional, while the seven *contact* relations occupy thin bands of width
`~tau`, so they rarely win even when appropriate (and never, as `tau -> 0`). Three principled
selectors are provided for the three questions you might be asking:

```python
from paa import most_probable_relation, hierarchical_decode, map_relation, THICK_RELATIONS

most_probable_relation(p, restrict=THICK_RELATIONS)   # flat argmax among comparable siblings
hierarchical_decode(p)     # top-down MAP: [('non_separated', .7), ('e_inside_r', .5), ('during', .45)]
map_relation(e, r, tau=0.1)                            # relation of the most likely arrangement
```

`hierarchical_decode` is "argmax done right" — it decides the coarse family first (robust),
then refines. `map_relation` returns the relation of the single most likely configuration and
can name a contact relation (`meets`) exactly where a flat argmax would pick a thicker
neighbour.

## Command line

The same algebra is available as a CLI — `paa <command>` or `python -m paa <command>`.
Intervals are given by a spec token that maps 1:1 to the constructors
(`point:MU[,SIGMA]`, `mid:MU_T,SIGMA_T,MU_D,SIGMA_D`, `start:…`, `end:…`,
`endpoints:MU_S,SIGMA_S,MU_E,SIGMA_E`), and every data command takes `--tau` and `--json`.

```bash
paa prob    --e mid:2,0.4,4,0.4 --r mid:3.2,0.4,3,0.4 --tau 0.2   # 13 probabilities, sum = 1
paa decode  --e mid:2,0.4,4,0.4 --r mid:3.2,0.4,3,0.4 --tau 0.2   # non_separated -> partial_overlap -> overlaps
paa explain meets                                                # boundary decomposition, converse, contacts
paa prob --e point:0 --r point:3,1 --json | jq .probabilities.before   # machine-readable
```

| command | does |
|---|---|
| `prob` · `coarse` | the 13 leaf probabilities · taxonomy-node (coarse-predicate) sums |
| `decode` | one best relation (`--method hierarchical` / `argmax` / `map`) |
| `primitives` | the four soft `A/B/G/H` temporal primitives |
| `refine --parent NODE` | conditional `P(leaf \| node)` within a taxonomy node |
| `explain RELATION` · `list` | static reference for a relation · list relations + taxonomy |
| `figures` · `demo` | regenerate the paper figures (`viz` extra) · print a fixed demonstration |

Run `paa --help` (or `paa <command> --help`) for the full reference.

## Validated properties

Every claim below is enforced by the test suite (`uv run pytest`):

| property | test |
|---|---|
| the 13 relations sum to 1 for any `tau` (true partition) | `test_partition.py` |
| analytic probabilities match Monte-Carlo to ~1e-3 | `test_monte_carlo.py` |
| `13 -> 5 -> 3` reduction; exact `meets`; identical points | `test_limits.py` |
| scale invariance and converse symmetry | `test_invariance.py` |
| the best-fitting-relation selectors | `test_decode.py` |

## API at a glance

```text
paa
├── IntervalGaussian, point                         # uncertain temporal objects
├── relation_probabilities(e, r, tau)               # the 13 probabilities (dict)
├── ProbabilisticAllenRelations(e, r)               # engine; per-relation methods
├── RELATION_NAMES, CONVERSE, THICK_RELATIONS        # constants
├── classify_arrangement(A, B, G, H, tau)           # crisp single-config classifier
├── coarse_predicates(p), refine(p, parent), LEAVES # taxonomy calculus
└── most_probable_relation / hierarchical_decode / map_relation
```

## Development

```bash
uv sync                # create the env (numpy, scipy, pytest)
uv run pytest          # run the test suite
uv run paa --help      # the CLI (uv run paa demo prints a demonstration)
uv build               # build sdist + wheel
```

Requires [uv](https://docs.astral.sh/uv/); the suite runs under Python 3.11–3.13.

## License

[MIT](LICENSE) © 2024-2025 Honda Research Institute Europe GmbH
