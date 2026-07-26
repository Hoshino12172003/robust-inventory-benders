# Fairness scalability S1 Attempt 1 Windows-path incident

Status: frozen read-only execution incident. These artifacts are scientifically
invalid for candidate selection and must never be resumed or reused.

## Frozen identity

- execution attempt: `1`
- stage: `scalability_s1_medium_large`
- status: `execution_incomplete`
- scientifically usable for candidate selection: `false`
- results reused: `false`
- seeds accessed: `[160]`
- failure class: `windows_path_length_pipeline_defect`
- working tree: `E:\rfs1`
- Git commit: `22ce2d63a4ad8cea021bf2b6cbe60273c0c2919c`
- input config SHA256:
  `3B5366F099A1B9BCB448D46D12E4391FE6FAF20499C837B7C1908930F6256ABE`
- protocol SHA256:
  `E4DBC0AE3C14F5907A3DE88EABC1BEBB33DE0D6D6A9F7C788060E655E9540DA5`
- resolved config SHA256 declared by the run:
  `170CF3B7DBAEEDF826A4413D35C449D1F641CEA9490473ABB5CAB1E2E248E70B`
- frozen V3 candidate SHA256:
  `7E8AAF39DE8C100B4CE9B46256A074FBD324B07DDC347D256494ED070D4E0EB6`

The evidence directory was read only. No file was changed, deleted, moved,
resumed, or imported.

## Execution state at failure

- generated instances: one, `instances/160.json`
- baseline records: one complete and certified
- frontier records: four complete records
- certified frontier records: three
- implementation-error frontier records: one
- `run.json`: 5
- `status.json`: 5
- algorithm checkpoints: 4
- committed post-evaluation chunks: 222 (three complete 1,831-scenario
  evaluations at chunk size 25)
- manifest completed/pending/solved/failed: `5 / 22 / 4 / 1`
- running status records: 0
- generated or accessed seeds other than 160: none found

The failed canonical run key is:

```text
fairness_scalability_development_medium_large__scalability_s1_frontier__rho_0__medium_large__seed_160__persistent_certified_cache_batch5
```

Its algorithm checkpoint was atomically complete. Failure occurred before the
first post-evaluation checkpoint could be committed. The attempted paths were:

```text
E:\rfs1\experiments\results_fairness_scalability\development_medium_large\runs\fairness_scalability_development_medium_large__scalability_s1_frontier__rho_0__medium_large__seed_160__persistent_certified_cache_batch5\post_evaluation\checkpoint\chunk_00000.json
```

Length: 259 characters.

```text
E:\rfs1\experiments\results_fairness_scalability\development_medium_large\runs\fairness_scalability_development_medium_large__scalability_s1_frontier__rho_0__medium_large__seed_160__persistent_certified_cache_batch5\post_evaluation\checkpoint\.chunk_00000.json.tmp
```

Length: 264 characters. The latter raised `FileNotFoundError`. The checkpoint
directory existed; the defect is the physical path design, not a missing manual
directory setup.

## Read-only SHA256 evidence

The directory contained 246 files. A deterministic inventory was formed by
sorting relative POSIX paths and hashing the UTF-8 lines
`SHA256<two spaces>relative_path\n`. Its SHA256 is:

`6C9D831DBF68AED69A25A35B73E3685FA2E09DE32EA6ED71EA172D999B5BEF49`.

Key files:

| File | SHA256 |
|---|---|
| `instances/160.json` | `3E62586E83912877EE74D11937523D98B0467A8AB8212FF72B9F61D6A61D29DF` |
| `resolved_config.yaml` | `A40F955BC7E19D58BC423F8B189F87D7846E046E072D99E964E9F53C3759DB8C` |
| `run_manifest.json` | `161D0379BF3953310254A734F286D1D4E84C7589F2BE1610CA38D3C2B99F1F27` |
| `scalability_development_manifest.json` | `2CD52713BDCEED483D88D0B55B9CA2C4B2114B5B60EE4490AFFDA81783EB35D1` |
| baseline `run.json` | `5D376F4B949504AEDF7382EA25FA7A9068DA6634F2CD68BF7384899079F0557B` |
| baseline `status.json` | `5392EDB4DF63513AA5ABFC282704CC696B56C08DD8D1B1CC3A6A5B719F85A778` |
| `single_cut` algorithm checkpoint | `4238CE6B387432F0F5241BAD55A6C12C2C181AB0B75E1A7B7F885097E44CABD7` |
| `single_cut` `run.json` | `FEB86E8FC74F6FE6740BF003C89E70FC3371F76878EB2518B39F0E5F817E08E1` |
| `single_cut` `status.json` | `D42E00EC6732A12234D1A3A09A4F279C54CDEF077CC7E82047796F71E5C18D28` |
| `persistent_separation` algorithm checkpoint | `1E45C7CCF4C63F3FEEB9111E4AF5FE996BBBF57222E5892B175755E9E50FD539` |
| `persistent_separation` `run.json` | `92025E9331F16F23B40CF49BBCCF979667B41DFF6E0ED84DD0256FC1E4333277` |
| `persistent_separation` `status.json` | `9C7813A346DA01950A21D8A03962506FC365AE8326F2E18D19579F36C68F35F4` |
| `persistent_certified_cache` algorithm checkpoint | `CC5680E2D234C6C0A5C1D4570A807ABFDF9B53384DB8517619D2406FDE8ADC24` |
| `persistent_certified_cache` `run.json` | `389E65A6F97B2FAE0A250B7BD1F5E5D43134004D59523C055AA1761445664898` |
| `persistent_certified_cache` `status.json` | `3DAD47D017F74EFC3B63696A6B498A09DC42EC4A71019121B3A8D04AA28E59B1` |
| failed frontier `algorithm_checkpoint.json` | `7950C88AAE82A7BDB3AD72EF05154C8535B5B64603C42905264A034D693BB80B` |
| failed frontier `run.json` | `F72371EBAD224A4D973EC372E021D556F60888275C38B29D6AB8ABDF786B86FA` |
| failed frontier `status.json` | `1A0395A7EBA8903D27D3CD1B16189181BFBD3EAE9B96DFABD14D9965EA7DF3B3` |

The remaining run/status hashes were independently recomputed during the
audit. The complete inventory digest above freezes their exact set and bytes
without committing any formal result artifact.

## Scientific governance

Attempt 1 is not a development sample. Its certified records, runtimes,
per-seed behavior, and partial candidate coverage may not enter candidate
comparison, selection, rho choice, statistics, figures, or paper results.
Attempt 2 must start from a nonexistent new output directory under a new Git
commit. It must not read any Attempt 1 instance, baseline, `C_anchor`, run,
checkpoint, result, summary, or manifest.
