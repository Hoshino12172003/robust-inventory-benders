# Hybrid Final Holdout 实验结果（冻结草稿）

## 结果依据与统计设计

本节仅使用 Hybrid `FINAL_HOLDOUT`，不混入 D1/D2。唯一最终依据由 `freeze_manifest.json` 固定：原始 ZIP SHA256 为 `BD253A4DE967143B8CD88E7DEADCB821D6062208CB71664B900F9BD8EBDF3839`，PR #51 merge commit 为 `13493ba63006604443f54f61799842dc2a3fbac9`，修正结果 CSV SHA256 为 `50EB5823F4C7138E65FA36546B90EE081B48949D2F961F5AFDFAE098A7F0A496`，`paper_metrics.json` SHA256 为 `044689ABF1ADD1C1FC217FCB5F46B8D280D8659865EE3A3707EBB9FE792F2E37`。全部 120 个任务（20 个 baseline、100 个 frontier）通过只读审计；100 个 frontier 均完成 exact certification，13,050 个 post-evaluation chunk 的 SHA 全部正确。独立实验单位为 seed；每个 scale×ρ 包含 10 个配对 seed。跨规模 overall 统计先在 seed 内聚合五个 ρ，再进行 seed-cluster bootstrap；五个 per-ρ 检验使用 Holm 校正。

## 指标解释

`robust_minimum_fill_rate` 是算法证书 `1-T`，表示所有精确不确定场景、所有适用区域的最低 fill-rate 保证。`wminfr` 是 exact post-evaluation 直接观测到的 $\min_s\min_r$ 区域 fill rate；两者在 100 个任务上的最大绝对差为 `1.499e-14`。`minimum_weighted_mean_fill_rate` 则是 `min_s(1 - total_shortage_s / total_demand_s)`，即最坏场景下按需求加权的系统平均 fill rate，不能标为“最低区域 fill rate”。成本采用 exact post-evaluation 的 first-stage cost 加最坏 recourse cost。算法 runtime、post-evaluation wall time 与 total wall time分别报告，PAR-2 仅以算法 runtime 为基础。

## 公平—成本权衡

Medium-large 的平均认证最低 fill rate 从 ρ=0 的 `0.716` 提升到 ρ=0.10 的 `0.950`；Large 从 `0.764` 提升到 `0.927`。对应的实际 price of fairness 均由近 0 增至约 `7.97%`（Medium-large）和 `7.48%`（Large）。图 `figure_fairness_cost_tradeoff.png` 展示了这一单调权衡；完整 mean、median、standard deviation、IQR、min、max 见 `table_all_descriptive_statistics.csv`，100 个 seed×ρ 明细见 `table_complete_seed_results.csv`。

## 可扩展性与跨规模比较

跨全部 50 个任务，Medium-large 的平均算法 runtime 为 `19.53` 秒，Large 为 `66.94` 秒；平均 total wall time 分别为 `31.54` 秒和 `122.46` 秒。运行时间分解见 `figure_runtime_scalability.png`，迭代、完整场景块和认证 Farkas cut 见 `figure_algorithm_structure.png`。Large−Medium-large 的 seed 内先聚合 ρ 后 T 差均值为 `0.0060`，cluster bootstrap 95% CI 为 `[-0.0651, 0.0767]`，配对置换 p 值为 `0.8398`。五个 per-ρ Holm 校正 p 值均为 1.0，因此本 holdout 不支持两个规模在 T 上存在系统性差异的结论；这不等同于证明二者相同。

## Bound reconciliation 与科学有效性

唯一 crossing 出现在 Large、seed 172、ρ=0.10。历史 max-LB 比最终 UB 高 `2.0694404762322538e-05`，位于冻结容差 `1e-4` 内。该历史值只作为轨迹 ledger；论文认证下界采用最终当前 master solver best bound。该 bound 与最终 incumbent/UB 相等，且 final exact separation optimal、objective bound 为 `-0.0`，完整认证同一解鲁棒可行。因此该异常定性为 reporting/ledger 语义问题，不要求优化重跑，也不改变 100/100 certified 的科学状态。

## 报告边界

表和图均直接由冻结的 `paper_metrics.json` 生成。D1/D2 只能作为 development/scalability evidence 单独讨论，不进入本节估计、置信区间或显著性检验。任何改变 ZIP SHA、merge commit、修正 CSV SHA 或 paper-metrics SHA 的分析均不得沿用“唯一最终依据”标签。
