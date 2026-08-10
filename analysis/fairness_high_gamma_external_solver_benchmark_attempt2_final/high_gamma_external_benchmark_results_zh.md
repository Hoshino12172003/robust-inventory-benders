---
source_zip_sha256: 4F8CEC3F9EAF69B053AE3DBAE6C29D5AAEF7DAAD87C49A798039DBCF9FADD783
run_git_commit: 797caafd12c006e85bc3394b01905bbfb137b0a9
config_sha256: A377D5B040FED160B323B58D42D9FFD1DE57E52F6C64D2050D6667E47DCA9334
protocol_sha256: 4C76B5C7A02E245174BE02B6FCEBBCD744EB6B684A1F0CA71D05964EB1F1A32F
generation_schema: fairness_high_gamma_attempt2_final_reporting_v1
---

# 高 Gamma 压力测试与外部通用求解器基准

## 实验定位

本实验是在 4 个仓库、4 个产品和 5 个区域的小规模实例上进行的压力测试，不进入 Final Holdout 主样本，也不替代 Gamma=0/1/2 的跨规模敏感性分析。需求不确定维度为20，完整预算场景数由 Gamma=2 的211个增长至 Gamma=3 的1,351个和 Gamma=4 的6,196个。比较对象统一称为“基于直接确定性等价模型的通用求解器基准”（general-purpose solver benchmark based on the direct deterministic equivalent formulation）。

## 认证与计算结果

Hybrid 在三个 Gamma 水平的15个单元上均获得 `certified_robust_optimal`，且全部完成最终精确分离和完整 post-evaluation。Direct 在 Gamma=2和3各认证5/5；Gamma=4的5个单元均达到1800秒时限、没有 incumbent，因而只能按 time-limit/PAR-2 计入，不能报告 Direct 的目标值或最优性 gap。Gamma=2和3中双方均认证单元的最大绝对 T 差分别为 5.079e-07 和 1.511e-06。

## Rolling proposal pool 与 master

rolling pool 只限制候选场景 proposal 的记忆，不删除已经提交到 master 的场景块。其不变量为

$$
S_{k+1} = S_k \cup \{\hat s_k\}.
$$

Gamma=4 的平均 eviction、rediscovery 和 duplicate proposal 数分别为 24.8、11.8 和 0.0；15个 Hybrid 任务的 committed blocks/cuts 账本均为 append-only，并以 final exact separation 完成认证。在这一小规模压力测试中，没有观察到由候选池淘汰导致的循环或认证失败。该经验结果不能外推为任意 Gamma 的理论或计算可扩展性证明，也不代表 Hybrid 优于所有先进算法。

## 成本锚点解释

每个单元使用对应 Gamma 的认证 baseline 上界作为 `C_anchor`。严格表述是 $C^* \in [LB,C_{anchor}]$，而不是 $C_{anchor}=C^*$。15个锚点的最大相对认证 gap 为 0.00009349，小于冻结容差 $10^{-4}$，相对于 $\rho=0.025$ 很小，因而不足以解释主要公平变化。

## 结论边界

结论仅限于本次 five-seed、小规模、Gamma=2/3/4 的预注册实验。Direct Gamma=4 的结果是“无 incumbent 的 time limit”，不是认证解；Gurobi 也不代表全部行业算法。
