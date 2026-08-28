# szl-formulas

Canonical GitHub source for **SZLHOLDINGS/szl-formulas**.

**GitHub is the source of truth.** The Hugging Face Hub kernel package is a **publish mirror** of this tree. Do not treat the Hub copy as canonical.

ATELIER owns Hub cards. This README is a source face for the Git repository. It is **not** a second model card.

## What this is

A **software kernel**: executable checks — 21 canonical typed formulas plus a hash-chained governed-loop composer. Pure Python, stdlib-only. **Not** trained weights. **Not** CUDA benches.

- Registry count: 21 formulas (`REGISTRY`).
- **locked-proven = exactly 8**: `{F1, F4, F7, F11, F12, F18, F19, F22}` (`LOCKED_PROVEN_COUNT`). Per-obligation `PROOF_STATUS` tags are not membership in that set.
- Composer Λ roll-up is **ADVISORY** only.
- **Λ = Conjecture 1**, never a theorem. Uniqueness is open.
- Doctrine v11.
- License: Apache-2.0.

## Load (via the Hub publish mirror)

```python
from kernels import get_kernel
fx = get_kernel("SZLHOLDINGS/szl-formulas", revision="main", trust_remote_code=True)
```

Hub package: https://huggingface.co/SZLHOLDINGS/szl-formulas

## Layout

- `build.toml` — kernel-builder manifest (`universal = true`)
- `build/torch-universal/szl_formulas/` — `__init__.py`, `_formulas.py`, `_composer.py`, `metadata.json`
- `tests/test_formulas.py` — honest registry / locked-8 / replay tests
- `LICENSE` — Apache-2.0
