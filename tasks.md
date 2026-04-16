# Tasks

- Validate section-level claim-worthiness thresholds on a representative paper set and tune false-positive/false-negative balance.
- Add configuration knobs for worthiness thresholds and suppression toggles in `config.yaml` to allow dataset-specific calibration.
- Add pipeline-level metrics logging for claim suppression reasons to support iterative prompt/filter refinement.
- Evaluate whether `ResultSubtype` should add a dedicated `scoped_result` subtype after collecting subtype-confusion metrics on a representative paper set.
