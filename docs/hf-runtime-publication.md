# Source-owned Python publication

GitHub `szl-holdings/szl-formulas` owns the approved Python runtime. The manual
`hf-runtime-publication.yml` workflow is its only runtime publisher. Supply the
exact reviewed main SHA and the independently observed main SHA of each existing
Hub repository: the legacy model and first-class kernel `SZLHOLDINGS/szl-formulas`.
The workflow requires a verified source commit and successful canonical CPU
contract on that SHA, then reruns the full package and publication tests.

The fixed payload comes from immutable `git show SHA:path` blobs, not local
working files. Every tracked package file is copied from `torch-ext/szl_formulas`
to both `build/torch-universal/szl_formulas` and `build/torch-cpu/szl_formulas`.
This includes the atlas data and ELECTRE/Choquet implementation. Both variants
have source-bound loader metadata. `SZL_SOURCE_BINDING.json` records the GitHub
SHA, source paths, and SHA-256 of every managed runtime file.

The publisher uses ordinary Git fast-forward pushes. It prepares both targets
and checks authenticated write endpoints with Git dry-runs before the first
write. It checks that GitHub main and each target parent have not moved. A
competing write refuses publication rather than overwriting that writer. It
does not create repositories, delete files, move tags, alter visibility,
change credentials, or execute quarantined serialization.

Only the enumerated Python package slots, generated variant metadata, and source
binding are managed. Every other file retains its exact Git mode/type/blob ID,
including the curated README, license, existing provenance, quarantined model
files, and Hub-only evidence. After each push, a fresh fetch and remote-head
check must match the candidate revision. Every managed file is read back and
compared byte for byte; every unmanaged blob is compared to its baseline.

The two repositories cannot be committed atomically. A second-target failure
records `PARTIAL` and retains the first target's completed evidence. A moved
parent requires a new observed input, never a force push. A refused write or
failed readback is not successful publication. Immutable Actions artifacts
retain the source, parents, candidate and observed revisions, replacements,
preserved blobs, and actual readback hashes.

Kernel major version 1 in loader metadata is separate from the package's 0.1.0
version. This lane publishes software, not trained weights, GPU acceleration,
or model-quality evidence. Lambda remains advisory and Conjecture 1; the
locked-proven set remains exactly eight.
