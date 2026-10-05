# szl-formulas

Software kernel for SZL formula composition. **Not a model. No weights.**

This GitHub tree is the source. The Hub package is the publish mirror: [`kernels/SZLHOLDINGS/szl-formulas`](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas). Card: [`SZLHOLDINGS/szl-formulas`](https://huggingface.co/SZLHOLDINGS/szl-formulas).

The source-owned [runtime publication workflow](docs/hf-runtime-publication.md)
binds both Hub distributions to an exact tested GitHub revision and verifies
their published bytes while retaining curated Hub assets.

## What

Python package under `torch-ext/szl_formulas/`. Formula composer + canonical formula table. Apache-2.0.

## What this is NOT

- Hub `model.joblib` is **QUARANTINED** executable serialization. Do not `joblib.load` it. GitHub source is the approved path.

- Not trained weights, not a LoRA, not GGUF
- Not a CUDA/Triton speedup claim (no MEASURED benches in this repo)
- Not the TypeScript product `ouroboros` and not `lutar-lean`

## Load

Set `SZL_FORMULAS_HF_REVISION` to the immutable **first-class Kernel Hub** commit
from a verified publication of [`kernels/SZLHOLDINGS/szl-formulas`](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas). Use the `kernels`
client version qualified with that publication. The GitHub source commit,
model-type mirror commit, and Kernel Hub commit are separate identities.
An observed head, a branch name, or a successful import does not qualify a release.

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

Doctrine v11 LOCKED. Owner: Stephen Lutar / SZL Holdings.

## License

Apache-2.0. Copyright 2026 SZL Holdings.
