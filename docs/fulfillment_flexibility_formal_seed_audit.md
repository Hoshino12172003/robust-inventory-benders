# Formal fulfillment-flexibility seed audit

Status: frozen before formal instance generation and formal optimization.

## Selected formal block

The formal synthetic seed block is `230--239`. The same ten seed labels are
used for medium_large and large, creating twenty scale-seed instances. The
predeclared operational fallback is the prefix `230--234`; it may be activated
only by a reviewed pre-solve protocol amendment under the frozen hardware rule.

## Non-reuse audit

The audit recursively inspected structured seed-bearing fields in tracked YAML
and JSON under `configs`, `experiments/configs`, `analysis`, and
`real_data_studies`. It also searched tracked text outside the new formal
artifacts for candidate-seed references near a seed label. No tracked archive
file is present in the current checkout; archived experiment metadata under
`analysis` was included in the structured scan.

The repository history represented by current `main` contains the following
relevant synthetic allocations:

| Seeds | Existing role |
|---|---|
| 0--54 | early tuning, correctness, comparison, and extended evaluations |
| 75--79 | V3 development |
| 80--89 | V3 validation |
| 90--109 | V3 final evaluation |
| 110--119 | regional-service diagnostic |
| 120--129 | regional-service development |
| 130--159 | scalability, validation, and final-holdout allocations |
| 160--169 | D1/D2, remediation, and scalability evidence |
| 170--179 | cross-scale final holdout |
| 180--184 | minimal paired benchmark |
| 185--189 | high-Gamma external-solver benchmark |
| 190--192 | PR #79 development-only flexibility diagnostic, outside current main |
| 201--205 | pre-existing core-point unit-test instances |
| 211 | pre-existing V3-stall unit-test instance |
| 221--225 | pre-existing V3 integration-test instances |

Seeds 230--239 do not occur in prior structured metadata or candidate-specific
tracked-text evidence. Earlier gaps are left unused rather than retroactively
filled. Olist weeks and M5 temporal cases are different experimental units and
are not treated as synthetic seed values; they are nevertheless excluded from
tuning and interpretation by the protocol.

## Machine-readable evidence

The committed evidence file is
`analysis/fulfillment_flexibility_formal_protocol/seed_audit.json`. It records
the audited source commit, scan counts, exact structured and explicit test seed values, PR #79's
external reservation, candidate conflicts, and the frozen formal/fallback
lists. Tests require `formal_seeds_untouched: true` and fail on any new
candidate-seed occurrence outside the formal protocol artifacts.

No formal instance was generated or inspected during this audit.
