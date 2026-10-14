---
license: apache-2.0
library_name: kernels
tags:
  - kernel
  - szl-holdings
  - governance
  - formulas
  - proof-status
  - lean4
  - lambda-aggregate
  - doi:10.5281/zenodo.19944926
---

<!-- Native artwork: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon SHA-256 6e2b7d72a3af3d9602b79de41df0194ede6496a9661f057108ec224c0efc92e6. -->
<p><a href="https://github.com/szl-holdings"><img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="40" height="40" /></a></p>

<p><small>Orbit v2 artwork © SZL Holdings · <a href="https://github.com/szl-holdings/szl-brand/blob/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/LICENSE">CC BY 4.0</a>. Page documentation remains Apache-2.0.</small></p>

# {{KERNEL_NAME}}

The 21 canonical SZL formulas, pure and typed, each carrying its verbatim `PROOF_STATUS`,
plus the governed-loop composer and the attributed formula/quant atlas. Pure-Python
(noarch) software kernel: standard library only, it does not import torch.
**No trained weights.**

Published from GitHub
[{{GITHUB_REPOSITORY}}](https://github.com/{{GITHUB_REPOSITORY}}) by its
`hf-kernel-publish` workflow. GitHub is the source of truth; this repository is
written only by that workflow.

## Load it

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

fx = get_kernel("{{REPO_ID}}", version={{KERNEL_VERSION}}, trust_remote_code=True)

assert fx.registry_count() == 21
print(fx.proof_status("madhava_series"))      # verbatim dataset PROOF-STATUS
print(sorted(fx.LOCKED_PROVEN_FORMULA_IDS))   # exactly 8 F-ids

chain = fx.run_governed_loop([
    {"formula_name": "lambda_bounded", "args": [[0.9, 0.8, 0.95]]},
    {"formula_name": "reed_solomon_singleton", "args": [255, 223]},
])
assert chain["replay_ok"] is True and chain["halted"] is False
print(chain["lambda_label"])                  # ADVISORY (Λ = Conjecture 1)
```

- `version={{KERNEL_VERSION}}` resolves the `v{{KERNEL_VERSION}}` branch, the stable API channel. It is
  republished on every release. The `kernels` library refuses a call that names neither a
  version nor a revision. Kernel major `{{KERNEL_VERSION}}` is not the package version (`fx.__version__`).
- `trust_remote_code` is required: SZLHOLDINGS is not a trusted kernel publisher on the Hub.
  The narrowest form is `trust_remote_code=["{{REPO_ID}}"]`. It permits running this
  repository's Python; review the source commit named in `SZL_KERNEL_SOURCE.json` first.
- CPU hosts load `build/torch-cpu`; CUDA and other hosts load `build/torch-universal`.
  Both variants carry identical code, built from the same GitHub commit.

### Pin an exact build (production)

```python
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("{{REPO_ID}}", repo_type="kernel")
pin = next(b.target_commit for b in refs.branches if b.name == "v{{KERNEL_VERSION}}")
fx = get_kernel("{{REPO_ID}}", revision=pin, trust_remote_code=["{{REPO_ID}}"])
```

Record `pin` (a commit of this kernel repository) in your lockfile.
`SZL_KERNEL_SOURCE.json` in this repository names the GitHub commit each build came from.
Commits of the model repository
[{{MODEL_TWIN_URL}}]({{MODEL_TWIN_URL}}) are not kernel revisions and do not resolve here.

## Proof honesty

Two distinct things, never conflated:

1. **Per-formula `PROOF_STATUS`**, mirrored verbatim from
   [`SZLHOLDINGS/canonical-formulas-v1`](https://huggingface.co/datasets/SZLHOLDINGS/canonical-formulas-v1):
   one of `PROVEN / AXIOM / SORRY / CONJECTURE`. A `PROVEN(...)` tag is an obligation-level
   label; it is not a claim of membership in the locked-proven set. `proof_status(name)`
   raises `KeyError` for an unknown name.
2. **`LOCKED_PROVEN_FORMULA_IDS`**: exactly 8, `{F1, F4, F7, F11, F12, F18, F19, F22}`, as
   machine-enforced by the no-axiom Lean theorem `locked_count_eight` in
   [szl-holdings/lutar-lean](https://github.com/szl-holdings/lutar-lean). The F-numbering is
   lutar-lean's own corpus; its mapping onto the 21 registry entries is not asserted here.

Λ (`lambda_aggregate`, weighted geometric mean) uniqueness is Conjecture 1 (open). Every Λ
roll-up is advisory, never proven trust.

## The governed-loop composer

`run_governed_loop(calls)` chains formula calls into hash-linked receipts. Four hard-stop
validators gate every step (`state_transition`, `drift_bounds`, `human_gate`, `axis_floor`);
on any failure the loop halts and the chain seals at the last good step. A step that raises
(for example `pac_bayes_mcallester` outside its domain) halts before any receipt is added. A
risk-like output `v` scores `max(0, 1 - v)`, so a vacuous bound (`v >= 1`) scores zero and
halts on `axis_floor`. `verify_chain(chain, calls)` replays a chain and rejects a tampered one.

`load_formula_atlas()`, `atlas_summary()`, `quant_domains()` and `list_attributed_formulas()`
read the attributed atlas (`formula_atlas.v1.json`) shipped inside the package. The atlas does
not change locked-proven membership.

## Provenance

Mirrors `SZLHOLDINGS/canonical-formulas-v1` (`code/python/formulas.py` + `composer.py`). Lean 4
formalization: [szl-holdings/lutar-lean](https://github.com/szl-holdings/lutar-lean), DOI
[10.5281/zenodo.20434308](https://doi.org/10.5281/zenodo.20434308); concept DOI
[10.5281/zenodo.19944926](https://doi.org/10.5281/zenodo.19944926).

## What this is not

- Not trained weights, not a model. Do not `pickle`/`joblib.load` anything from the Hub.
- No measured latency, CUDA speedup, tokens/s or joules are claimed.
- A successful load, a replayed chain or an advisory Λ is not a product qualification or a
  certificate for the surrounding application.
