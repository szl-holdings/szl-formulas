# szl-formulas

Software kernel for SZL formula composition. **Not a model. No weights.**

<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

This GitHub tree is the source. The Hub package is the publish mirror: [`kernels/SZLHOLDINGS/szl-formulas`](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas). Card: [`SZLHOLDINGS/szl-formulas`](https://huggingface.co/SZLHOLDINGS/szl-formulas).

The only writer of both Hub repositories is [`hf-kernel-publish`](.github/workflows/hf-kernel-publish.yml), configured in [`.github/hf-kernels.json`](.github/hf-kernels.json). A published release builds `build/torch-cpu` and `build/torch-universal` from `torch-ext/` at the release commit, pushes them fast-forward to the kernel repo's `main` and `v1` branches with a `SZL_KERNEL_SOURCE.json` binding that names that commit, verifies the result through `get_kernel` in a fresh job, and only then turns the model repository into a pointer card to the kernel repo. A weekly job re-verifies what the Hub serves. Nothing under `build/` is committed here; the tests run against `torch-ext` and both generated variants ([`tests/conftest.py`](tests/conftest.py)).

## What

Python package under `torch-ext/szl_formulas/`. Formula composer + canonical formula table. Apache-2.0.

## What this is NOT

- `model.joblib` is **QUARANTINED** executable serialization. It is absent from the Hub model repository and `hf-kernel-publish` keeps it pruned. Do not `joblib.load` any copy. GitHub source is the approved path.

- Not trained weights, not a LoRA, not GGUF
- Not a CUDA/Triton speedup claim (no MEASURED benches in this repo)
- Not the TypeScript product `ouroboros` and not `lutar-lean`

## Load

Set `SZL_FORMULAS_HF_REVISION` to the immutable **first-class Kernel Hub** commit
from a verified publication of [`kernels/SZLHOLDINGS/szl-formulas`](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas). Use the `kernels`
client version qualified with that publication. The GitHub source commit,
model-repository commit, and Kernel Hub commit are separate identities.
An observed head, a branch name, or a successful import does not qualify a release.
A qualifying revision is a commit of the kernel repo's `v1` branch written by a
`hf-kernel-publish` run whose `verify` job passed; `SZL_KERNEL_SOURCE.json` at that
commit names the GitHub source commit. `version=1` (shown on the
[kernel card](hf/kernel-card.md)) resolves the moving head of that branch, not a
qualified release.

`trust_remote_code=True` permits execution of the selected repository's Python.
Review that exact revision, its provenance and publication evidence before enabling it.
The format check below only rejects missing or mutable revision inputs; it does not
verify hashes, publisher authorization or compatibility. If that evidence is unavailable,
stop the Hub load and use separately reviewed local source for development.

```python
import os
import re

hf_revision = os.environ.get("SZL_FORMULAS_HF_REVISION", "")
if re.fullmatch(r"[0-9a-f]{40}", hf_revision) is None:
    raise ValueError("A verified immutable Kernel Hub revision is required")

from kernels import get_kernel

get_kernel("SZLHOLDINGS/szl-formulas", revision=hf_revision, trust_remote_code=True)
```


## Source-only development

Take the exact 40-character GitHub source revision from the qualifying publication
receipt, then review `https://github.com/szl-holdings/szl-formulas/tree/<SOURCE_SHA>/torch-ext/szl_formulas`
at that immutable revision, separately from any Hub release.
With the source's dependencies already available, run from the reviewed checkout root:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("torch-ext").resolve()))
import szl_formulas as local_kernel
```

This selects local Python source rather than calling the Hub loader. Importing local
source also executes Python. This documentation check does not run that import,
install dependencies, qualify a runtime or establish a Hub publication.
## Honesty

| Claim | Label |
|---|---|
| Source on GitHub | REACHABLE |
| CUDA benches | UNAVAILABLE |
| Weights | not applicable |
| Λ | Conjecture 1 (advisory, never a theorem) |
| NVIDIA Nemotron 3 Nano | NVIDIA result, arXiv:2512.20848. Not an SZL formula. |

Doctrine v11 LOCKED. Owner: Stephen Lutar / SZL Holdings. The locked formulas stay SZL's. Nemotron stays NVIDIA's published model.

## License

Apache-2.0. Copyright 2026 SZL Holdings.
