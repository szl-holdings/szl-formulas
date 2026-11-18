# EML single-operator research: source-bound SZL experiment

**Status:** bounded offline numerical research only, not a new locked Lean
formula, trained model, universal symbolic compiler, or production service.

**Primary research:** Andrzej Odrzywołek, *All elementary functions from a
single operator*, arXiv:2603.21852v2 (4 April 2026), CC BY 4.0:
https://arxiv.org/html/2603.21852v2

**Author's code:** https://github.com/VA00/SymbolicRegressionPackage (MIT).
Our Python code is independently written and Apache-2.0; no upstream
compiler or training code has been copied.

## What the author demonstrated

The binary primitive is `EML(x,y) = exp(x)-ln(y)`, with constant `1`.
Paper Table 1 gives a constructive route to 36 familiar
scientific-calculator primitives; this is its concrete basis, not a theorem
about arbitrary differential-algebraic elementary functions.

The three shallow expressions implemented in the SZL witness are:

| Paper identity | Pure EML RPN | EML nodes | Real domain |
| --- | --- | ---: | --- |
| `exp(x)=EML(x,1)` | `x1E` | 1 | bounded real |
| `e=EML(1,1)` | `11E` | 1 | constant |
| `ln(x)=EML(1,EML(EML(1,x),1))` | `11xE1EE` | 3 | `x>0`, bounded intermediates |

For `x>0`, the paper's log identity reduces to
`e-ln(exp(e-ln(x)))=ln(x)`. Numerically, however, intermediate
`exp(e-ln(x))` can overflow at small `x` even if the exact result
would be finite. These domains are **non-negotiable**.

The study's larger constructions use complex logarithms and explicit
branch choices; its Eq. 5 may produce a `2*pi*i` discontinuity across
the negative real axis. The paper discusses `ln(0)=-infinity` in
extended reals and Lean's different definition at zero. The SZL
component deliberately refuses these cases instead of coercing them
to apparently valid real values.

The author reports approximately 100% blind recovery at depth 2,
25% at depths 3–4, below 1% at depth 5, and none in 448 depth-6
attempts (author's experimental conditions). These are not SZL
training results. No symbolic-regression, optimizer, CUDA, or model
training is performed in this branch.

## Our original adaptation: EML domain-and-evidence witness

Owned by `torch-ext/szl_formulas/eml_research.py`, this is an
explicitly separate research namespace, **not** one of the existing
21 canonical SZL formula registry entries or the locked eight.

1. Accept only the three named identities and their known EML trees.
   Every node is strictly one of `one`, `var(x)`, `eml(left,right)`.
   Maximum 31 nodes and depth 8. Emit stable RPN and SHA3-256 tree
   fingerprint; a fingerprint is **not** a signature or Lean proof.
2. Accept 1–32 canonical decimal-text inputs, with bounded magnitude,
   and compute using only Python `Decimal` (48-digit precision).
   No arbitrary expression evaluation, external HTTP or native loads.
3. On every node refuse `right <= 0`, non-finite values, excessive
   exponent arguments or intermediate magnitudes. This is
   `REAL_ONLY_BOUNDED`; no complex branch is evaluated.
4. Compare each allowed EML result against `Decimal.exp/ln` using
   `abs(observed-reference) <= 1e-24*max(1,abs(reference))`.
   That is a sampled numerical test, **not an analytic proof**.
5. Record exact denominator `N=requested samples`, match count
   `M`, refused count `R`, and coverage `M/N`. No silently
   reducing the denominator when a sample fails. Refusal reasons
   are finite codes and input values are not copied into receipts.
6. Publish `analytic_identity_proven_here=false`,
   `lean_proof_verified=false`, `source_issuer_verified=false`,
   `trained_model=false`, `execution_authority=NONE`,
   `production_qualified=false`.

A numerical failure or an out-of-domain point returns `REFUSED` or
`PARTIAL_REFUSAL`, **not** a fabricated success. This is the
SZL adaptation: keep mathematical simplicity, but make every
domain, sample denominator, cost and proof level inspectable.

## Integration with the existing estate

- **GitHub:** `szl-formulas` is the source owner; module is included by
  its existing setuptools kernel package layout. No source or rights
  from the paper's implementation are redistributed.
- **Second Brain:** future *untrusted* symbolic candidates can be
  evaluated on frozen, rights-cleared data, but this does not
  authorize source promotion or live model output.
- **Lake:** a future separately approved writer can record
  allowlisted sample counts and RPN digests as evidence; no existing
  signed receipt chains are modified by this module.
- **Command Center:** an operator-only read-only inspector can display
  these qualified source facts; it must never imply a deployed
  verified theorem from a successful numerical fixture.
- **Hugging Face Kernel Hub / PyPI:** source checkout, signed PR,
  CI and pinned publish are **distinct admission gates**. A GitHub
  merge is not an HF sync or a package release.
- **Cloudflare:** no route, wallet, DNS, Access or x402 rule is changed.

An actual research paper/claim about improved symbolic discovery
would require an independent baseline (e.g. SymPy, PySR, author toolkit),
multiple seeds, all failure cases, held-out tests, cost/latency,
license checks and formal derivations checked in Lean. The existing
`lutar-lean` locked set remains unchanged; Lambda uniqueness
is not a theorem or permission to execute.

## Reproduction and falsifiers

From the reviewed source checkout:

```sh
PYTHONPATH=torch-ext python -m szl_formulas.eml_research
python -m pytest tests/test_eml_research.py -q
```

Negative cases include `ln(0)`, `ln(-1)`, too-small positive x whose
intermediate expression becomes enormous, oversized `exp(x)`,
unbounded tree depth, unknown op names, NaN/Infinity and malformed
decimal inputs. The main scientific distinction is that a correct
algebraic identity may still be **numerically inadmissible** under
a particular finite arithmetic domain.

**Author attribution:** All mathematical universality and EML
identities above are due to Odrzywołek. This module's guards, corpus
limits, SHA3 structural witness, measured/refused denominators and
implementation are original SZL code.
