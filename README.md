# szl-formulas

Software kernel for SZL formula composition. **Not a model. No weights.**

This GitHub tree is the source. The Hub package is the publish mirror: [`kernels/SZLHOLDINGS/szl-formulas`](https://huggingface.co/kernels/SZLHOLDINGS/szl-formulas). Card: [`SZLHOLDINGS/szl-formulas`](https://huggingface.co/SZLHOLDINGS/szl-formulas).

## What

Python package under `torch-ext/szl_formulas/`. Formula composer + canonical formula table. Apache-2.0.

## What this is NOT

- Hub `model.joblib` is **QUARANTINED** executable serialization. Do not `joblib.load` it. GitHub source is the approved path.

- Not trained weights, not a LoRA, not GGUF
- Not a CUDA/Triton speedup claim (no MEASURED benches in this repo)
- Not the TypeScript product `ouroboros` and not `lutar-lean`

## Load

```python
from kernels import get_kernel
get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)
```

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
