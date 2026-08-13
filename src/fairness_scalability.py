from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB

from .instance import InventoryInstance
from .robust_regional_fairness import (
    FAIRNESS_FEASIBILITY_TOLERANCE,
    FairnessFeasibilityCut,
    FairnessSeparationResult,
    FixedScenarioCertificate,
    _add_binary_product,
    _add_normalized_farkas_cone,
    certify_fixed_scenario_fairness_feasibility,
    fairness_cut_from_ray,
    first_stage_cost_value,
    scenario_demand,
    separation_partition_certifies,
)
from .status import gurobi_status_name


SINGLE_CUT = "single_cut"
PERSISTENT = "persistent_separation"
PERSISTENT_CACHE = "persistent_certified_cache"
PERSISTENT_CACHE_BATCH5 = "persistent_certified_cache_batch5"
SCALABILITY_CANDIDATES = (
    SINGLE_CUT,
    PERSISTENT,
    PERSISTENT_CACHE,
    PERSISTENT_CACHE_BATCH5,
)


def validate_scalability_strategy(value: str) -> str:
    strategy = str(value)
    if strategy not in SCALABILITY_CANDIDATES:
        raise ValueError(
            "fairness_scalability_strategy must be one of "
            + ", ".join(SCALABILITY_CANDIDATES)
        )
    return strategy


def pattern_key(active: list[dict[str, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((int(item["region"]), int(item["product"])) for item in active)
    )


def pattern_payload(key: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [{"region": r, "product": j} for r, j in key]


@dataclass(frozen=True)
class CertifiedCacheBatch:
    cuts: list[FairnessFeasibilityCut]
    candidate_count: int
    hit_count: int
    certified_cut_count: int
    runtime: float
    uncertified_count: int


class CertifiedScenarioCache:
    """Store patterns only; every use is recertified at the current point."""

    def __init__(self) -> None:
        self._patterns: list[tuple[tuple[int, int], ...]] = []
        self._pattern_set: set[tuple[tuple[int, int], ...]] = set()

    def add(self, active: list[dict[str, int]]) -> bool:
        key = pattern_key(active)
        if key in self._pattern_set:
            return False
        self._pattern_set.add(key)
        self._patterns.append(key)
        return True

    @property
    def size(self) -> int:
        return len(self._patterns)

    def certify_current_point(
        self,
        instance: InventoryInstance,
        *,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
        cost_budget_value: float,
        time_limit: float,
        feasibility_tolerance: float,
        max_cuts: int,
        output_flag: bool,
        certifier: Callable[..., FixedScenarioCertificate] = (
            certify_fixed_scenario_fairness_feasibility
        ),
        instrumentation: Any | None = None,
        instrumentation_call_id: str | None = None,
    ) -> CertifiedCacheBatch:
        start = time.perf_counter()
        cuts: list[FairnessFeasibilityCut] = []
        candidates = 0
        hits = 0
        uncertified = 0
        for key in self._patterns:
            remaining = float(time_limit) - (time.perf_counter() - start)
            if remaining <= 0.0 or len(cuts) >= int(max_cuts):
                break
            if instrumentation is None:
                candidates += 1
                active = pattern_payload(key)
                demand_values = scenario_demand(instance, key)
            else:
                with instrumentation.phase(instrumentation_call_id, "cache_candidate_processing_ns"):
                    candidates += 1
                    active = pattern_payload(key)
                    demand_values = scenario_demand(instance, key)
                instrumentation.increment(instrumentation_call_id, "cache_patterns_considered")
            certifier_kwargs = {
                "y_values": y_values, "x_values": x_values,
                "t_value": float(t_value), "cost_budget_value": float(cost_budget_value),
                "demand_values": demand_values, "time_limit": remaining,
                "feasibility_tolerance": float(feasibility_tolerance),
                "output_flag": output_flag,
            }
            if instrumentation is not None:
                certifier_kwargs.update(
                    instrumentation=instrumentation,
                    instrumentation_call_id=instrumentation_call_id,
                )
            certificate = certifier(instance, **certifier_kwargs)
            if certificate.infeasibility_certified and certificate.ray is not None:
                cut = fairness_cut_from_ray(
                    instance,
                    cost_budget_value=float(cost_budget_value),
                    demand_values=demand_values,
                    ray=certificate.ray,
                    active_deviations=active,
                )
                violation = -cut.value(y_values, x_values, float(t_value))
                if violation > float(feasibility_tolerance):
                    hits += 1
                    cuts.append(cut)
                    if instrumentation is not None:
                        instrumentation.increment(instrumentation_call_id, "cache_hits")
                else:
                    uncertified += 1
            elif not certificate.primal_feasible:
                # A cache is only a heuristic. An unavailable current-point
                # certificate cannot create a cut or certify feasibility; the
                # complete separation MILP must still run.
                uncertified += 1
        if instrumentation is not None:
            instrumentation.increment(
                instrumentation_call_id, "cache_misses", max(0, candidates - hits)
            )
        return CertifiedCacheBatch(
            cuts=cuts,
            candidate_count=candidates,
            hit_count=hits,
            certified_cut_count=len(cuts),
            runtime=time.perf_counter() - start,
            uncertified_count=uncertified,
        )


class PersistentFairnessSeparation:
    """Persistent normalized-Farkas MILP with call-local exclusions.

    The model contains only uncertainty and normalized Farkas variables.
    Across master iterations it reuses that model and replaces the objective
    whose coefficients depend on the current ``y, x, T, B_rho``. Any no-good
    constraints are removed before the call returns, so a scenario excluded at
    one master point can be selected again at the next point.
    """

    def __init__(
        self,
        instance: InventoryInstance,
        *,
        gamma: int,
        feasibility_tolerance: float = FAIRNESS_FEASIBILITY_TOLERANCE,
        output_flag: bool = False,
        instrumentation: Any | None = None,
        instrumentation_run_key: str | None = None,
    ) -> None:
        instrumentation_start_ns = (
            time.perf_counter_ns() if instrumentation is not None else None
        )
        start = time.perf_counter()
        self.instance = instance
        self.gamma = int(gamma)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.model = gp.Model("persistent_robust_regional_fairness_separation")
        self.model.Params.OutputFlag = 1 if output_flag else 0
        self.model.Params.FeasibilityTol = self.feasibility_tolerance
        self.z = self.model.addVars(instance.R, instance.J, vtype=GRB.BINARY, name="z")
        self.a, self.b, self.c, self.k, self.ell = _add_normalized_farkas_cone(
            self.model, instance
        )
        self.model.addConstr(
            gp.quicksum(self.z[r, j] for r in instance.R for j in instance.J)
            <= self.gamma,
            name="gamma",
        )
        self.za = {
            (r, j): _add_binary_product(
                self.model, self.z[r, j], self.a[r, j], f"za[{r},{j}]"
            )
            for r in instance.R
            for j in instance.J
        }
        self.zc = {
            (r, j): _add_binary_product(
                self.model, self.z[r, j], self.c[j], f"zc[{r},{j}]"
            )
            for r in instance.R
            for j in instance.J
        }
        self.zl = {
            (r, j): _add_binary_product(
                self.model, self.z[r, j], self.ell[r], f"zl[{r},{j}]"
            )
            for r in instance.R
            for j in instance.J
        }
        self.model.update()
        self.model_build_runtime = time.perf_counter() - start
        if instrumentation_start_ns is not None and instrumentation_run_key is not None:
            instrumentation.record_persistent_model_setup(
                run_key=instrumentation_run_key,
                start_ns=instrumentation_start_ns,
                end_ns=time.perf_counter_ns(),
            )
        self.total_optimize_runtime = 0.0
        self._build_runtime_reported = False
        self._disposed = False

    def _set_current_objective(
        self,
        *,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
        cost_budget_value: float,
    ) -> None:
        instance = self.instance
        first_stage = first_stage_cost_value(instance, y_values, x_values)
        objective = gp.quicksum(
            instance.base_demand[r][j] * self.a[r, j]
            + instance.demand_deviation[r][j] * self.za[r, j]
            for r in instance.R
            for j in instance.J
        )
        objective -= gp.quicksum(
            float(x_values[i][j]) * self.b[i, j]
            for i in instance.I
            for j in instance.J
        )
        objective -= gp.quicksum(
            (1.0 - instance.service_level[j])
            * (
                sum(instance.base_demand[r][j] for r in instance.R) * self.c[j]
                + gp.quicksum(
                    instance.demand_deviation[r][j] * self.zc[r, j]
                    for r in instance.R
                )
            )
            for j in instance.J
        )
        objective -= (float(cost_budget_value) - first_stage) * self.k
        objective -= float(t_value) * gp.quicksum(
            sum(instance.base_demand[r][j] for j in instance.J) * self.ell[r]
            + gp.quicksum(
                instance.demand_deviation[r][j] * self.zl[r, j]
                for j in instance.J
            )
            for r in instance.R
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def _pool_patterns(
        self, *, max_candidates: int, threshold: float
    ) -> tuple[list[tuple[tuple[int, int], ...]], int]:
        patterns: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        duplicates = 0
        solution_count = min(int(self.model.SolCount), max(1, int(max_candidates)))
        for number in range(solution_count):
            self.model.Params.SolutionNumber = number
            try:
                pool_objective = float(self.model.PoolObjVal)
            except (AttributeError, gp.GurobiError):
                pool_objective = float(self.model.ObjVal) if number == 0 else -math.inf
            if pool_objective <= threshold:
                continue
            key = tuple(
                (r, j)
                for r in self.instance.R
                for j in self.instance.J
                if float(self.z[r, j].Xn) >= 0.5
            )
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            patterns.append(key)
        self.model.Params.SolutionNumber = 0
        return patterns, duplicates

    def separate(
        self,
        *,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
        cost_budget_value: float,
        mip_gap: float,
        time_limit: float,
        max_cuts: int = 1,
        use_solution_pool: bool = False,
        output_flag: bool = False,
        certifier: Callable[..., FixedScenarioCertificate] = (
            certify_fixed_scenario_fairness_feasibility
        ),
        instrumentation: Any | None = None,
        instrumentation_call_id: str | None = None,
        final_exact_certification: bool = False,
    ) -> FairnessSeparationResult:
        if self._disposed:
            raise RuntimeError("Persistent separation model has been disposed.")
        if int(max_cuts) < 1 or int(max_cuts) > 5:
            raise ValueError("Persistent fairness separation supports 1 to 5 cuts.")
        start = time.perf_counter()
        first_build_report = not self._build_runtime_reported
        build_runtime = self.model_build_runtime if first_build_report else 0.0
        optimize_runtime = 0.0
        pool_candidates = 0
        duplicates = 0
        false_positive_evidence: list[dict[str, Any]] = []
        temporary_exclusions: list[Any] = []
        cuts: list[FairnessFeasibilityCut] = []
        prepare_phase = "final_exact_prepare_ns" if final_exact_certification else "separation_model_prepare_ns"
        self._build_runtime_reported = True
        if instrumentation is None:
            self._set_current_objective(
                y_values=y_values, x_values=x_values, t_value=t_value,
                cost_budget_value=cost_budget_value,
            )
            self.model.Params.MIPGap = max(0.0, float(mip_gap))
            self.model.Params.PoolSearchMode = 2 if use_solution_pool else 0
            self.model.Params.PoolSolutions = max(1, int(max_cuts))
        else:
            with instrumentation.phase(instrumentation_call_id, prepare_phase):
                self._set_current_objective(
                    y_values=y_values, x_values=x_values, t_value=t_value,
                    cost_budget_value=cost_budget_value,
                )
                self.model.Params.MIPGap = max(0.0, float(mip_gap))
                self.model.Params.PoolSearchMode = 2 if use_solution_pool else 0
                self.model.Params.PoolSolutions = max(1, int(max_cuts))
        last_status = "unknown"
        last_status_code = -1
        last_objective: float | None = None
        last_bound: float | None = None
        last_gap: float | None = None
        last_active: list[dict[str, int]] = []
        last_fixed: FixedScenarioCertificate | None = None
        try:
            while True:
                remaining = float(time_limit) - (time.perf_counter() - start)
                if remaining <= 0.0:
                    return FairnessSeparationResult(
                        status="time_limit",
                        has_incumbent=False,
                        objective=last_objective,
                        objective_bound=last_bound,
                        mip_gap=last_gap,
                        runtime=time.perf_counter() - start,
                        requested_mip_gap=float(mip_gap),
                        robust_feasibility_certified=False,
                        certification_reason="time_exhausted_before_certified_separation",
                        candidate_active_deviations=last_active,
                        fixed_scenario_certificate=last_fixed,
                        false_positive_scenarios_excluded=len(false_positive_evidence),
                        excluded_candidate_evidence=false_positive_evidence,
                        cuts=cuts,
                        separation_model_build_runtime=build_runtime,
                        separation_optimize_runtime=optimize_runtime,
                        pool_candidate_count=(pool_candidates if use_solution_pool else 0),
                        certified_batch_cut_count=(len(cuts) if use_solution_pool else 0),
                        duplicate_pattern_count=duplicates,
                    )
                self.model.Params.TimeLimit = max(1.0e-3, remaining)
                optimize_start = time.perf_counter()
                optimize_phase = "final_exact_optimize_ns" if final_exact_certification else "separation_milp_optimize_ns"
                if instrumentation is None:
                    self.model.optimize()
                else:
                    instrumentation.increment(instrumentation_call_id, "final_exact_calls" if final_exact_certification else "separation_milp_optimize_calls")
                    with instrumentation.phase(instrumentation_call_id, optimize_phase):
                        self.model.optimize()
                elapsed = time.perf_counter() - optimize_start
                optimize_runtime += elapsed
                self.total_optimize_runtime += elapsed
                last_status_code = int(self.model.Status)
                last_status = gurobi_status_name(last_status_code)
                has_incumbent = self.model.SolCount > 0
                last_objective = float(self.model.ObjVal) if has_incumbent else None
                last_bound = (
                    float(self.model.ObjBound)
                    if last_status_code not in {GRB.INFEASIBLE, GRB.UNBOUNDED}
                    and math.isfinite(float(self.model.ObjBound))
                    else None
                )
                last_gap = (
                    float(self.model.MIPGap)
                    if has_incumbent and self.model.IsMIP
                    else None
                )
                certified, reason = separation_partition_certifies(
                    last_status_code,
                    last_bound,
                    self.feasibility_tolerance,
                    false_positive_evidence,
                )
                if instrumentation is not None:
                    prefix = "final_exact" if final_exact_certification else "separation_milp"
                    def attr(name: str) -> Any:
                        try:
                            return getattr(self.model, name)
                        except (AttributeError, gp.GurobiError):
                            return None
                    instrumentation.add_numeric_diagnostic(
                        instrumentation_call_id, f"{prefix}_node_count", attr("NodeCount")
                    )
                    instrumentation.diagnostic(instrumentation_call_id, f"{prefix}_status", last_status)
                    instrumentation.diagnostic(instrumentation_call_id, f"{prefix}_objective_bound", last_bound)
                    if not final_exact_certification:
                        instrumentation.diagnostic(instrumentation_call_id, "separation_milp_solution_count", attr("SolCount"))
                        instrumentation.diagnostic(instrumentation_call_id, "separation_milp_incumbent", last_objective)
                        instrumentation.diagnostic(instrumentation_call_id, "separation_milp_gap", last_gap)
                if has_incumbent:
                    if instrumentation is None:
                        patterns, duplicate_count = self._pool_patterns(
                            max_candidates=(int(max_cuts) if use_solution_pool else 1),
                            threshold=self.feasibility_tolerance,
                        )
                    else:
                        with instrumentation.phase(instrumentation_call_id, "solution_pool_extract_ns"):
                            patterns, duplicate_count = self._pool_patterns(
                                max_candidates=(int(max_cuts) if use_solution_pool else 1),
                                threshold=self.feasibility_tolerance,
                            )
                        instrumentation.increment(instrumentation_call_id, "pool_patterns_extracted", len(patterns))
                else:
                    patterns, duplicate_count = [], 0
                pool_candidates += len(patterns)
                duplicates += duplicate_count
                if not patterns:
                    return FairnessSeparationResult(
                        status=last_status,
                        has_incumbent=has_incumbent,
                        objective=last_objective,
                        objective_bound=last_bound,
                        mip_gap=last_gap,
                        runtime=time.perf_counter() - start,
                        requested_mip_gap=float(mip_gap),
                        robust_feasibility_certified=certified,
                        certification_reason=reason,
                        false_positive_scenarios_excluded=len(false_positive_evidence),
                        excluded_candidate_evidence=false_positive_evidence,
                        cuts=cuts,
                        separation_model_build_runtime=build_runtime,
                        separation_optimize_runtime=optimize_runtime,
                        pool_candidate_count=(pool_candidates if use_solution_pool else 0),
                        certified_batch_cut_count=(len(cuts) if use_solution_pool else 0),
                        duplicate_pattern_count=duplicates,
                    )
                added_feasible_exclusion = False
                for key in patterns:
                    if len(cuts) >= int(max_cuts):
                        break
                    remaining = float(time_limit) - (time.perf_counter() - start)
                    if remaining <= 0.0:
                        break
                    active = pattern_payload(key)
                    last_active = active
                    demand_values = scenario_demand(self.instance, key)
                    certifier_kwargs = {
                        "y_values": y_values, "x_values": x_values,
                        "t_value": float(t_value), "cost_budget_value": float(cost_budget_value),
                        "demand_values": demand_values, "time_limit": remaining,
                        "feasibility_tolerance": self.feasibility_tolerance,
                        "output_flag": output_flag,
                    }
                    if instrumentation is not None:
                        certifier_kwargs.update(
                            instrumentation=instrumentation,
                            instrumentation_call_id=instrumentation_call_id,
                        )
                    fixed = certifier(self.instance, **certifier_kwargs)
                    last_fixed = fixed
                    if fixed.infeasibility_certified and fixed.ray is not None:
                        cut = fairness_cut_from_ray(
                            self.instance,
                            cost_budget_value=float(cost_budget_value),
                            demand_values=demand_values,
                            ray=fixed.ray,
                            active_deviations=active,
                        )
                        if -cut.value(y_values, x_values, float(t_value)) > self.feasibility_tolerance:
                            cuts.append(cut)
                        continue
                    if fixed.primal_feasible:
                        evidence = {
                            "active_deviations": active,
                            "fixed_scenario_certificate": fixed.to_dict(),
                            "reason": "fixed_scenario_primal_feasible",
                        }
                        false_positive_evidence.append(evidence)
                        active_set = set(key)
                        constraint = self.model.addConstr(
                            gp.quicksum(self.z[r, j] for r, j in key)
                            - gp.quicksum(
                                self.z[r, j]
                                for r in self.instance.R
                                for j in self.instance.J
                                if (r, j) not in active_set
                            )
                            <= len(key) - 1,
                            name=f"call_local_fixed_feasible[{len(false_positive_evidence)-1}]",
                        )
                        temporary_exclusions.append(constraint)
                        added_feasible_exclusion = True
                        continue
                    if not cuts:
                        return FairnessSeparationResult(
                            status=f"uncertified_{fixed.primal_status}",
                            has_incumbent=True,
                            objective=last_objective,
                            objective_bound=last_bound,
                            mip_gap=last_gap,
                            runtime=time.perf_counter() - start,
                            requested_mip_gap=float(mip_gap),
                            robust_feasibility_certified=False,
                            certification_reason=fixed.certification_reason,
                            candidate_active_deviations=active,
                            fixed_scenario_certificate=fixed,
                            false_positive_scenarios_excluded=len(false_positive_evidence),
                            excluded_candidate_evidence=false_positive_evidence,
                            separation_model_build_runtime=build_runtime,
                            separation_optimize_runtime=optimize_runtime,
                            pool_candidate_count=(pool_candidates if use_solution_pool else 0),
                            duplicate_pattern_count=duplicates,
                        )
                if cuts:
                    primary = cuts[0]
                    return FairnessSeparationResult(
                        status=last_status,
                        has_incumbent=True,
                        objective=last_objective,
                        objective_bound=last_bound,
                        mip_gap=last_gap,
                        runtime=time.perf_counter() - start,
                        requested_mip_gap=float(mip_gap),
                        robust_feasibility_certified=False,
                        certification_reason="violated_scenarios_fixed_lp_certified",
                        cut=primary,
                        cuts=cuts,
                        candidate_active_deviations=primary.active_deviations,
                        fixed_scenario_certificate=last_fixed,
                        false_positive_scenarios_excluded=len(false_positive_evidence),
                        excluded_candidate_evidence=false_positive_evidence,
                        cut_certificate_source="fixed_scenario_normalized_farkas_lp",
                        separation_model_build_runtime=build_runtime,
                        separation_optimize_runtime=optimize_runtime,
                        pool_candidate_count=(pool_candidates if use_solution_pool else 0),
                        certified_batch_cut_count=(len(cuts) if use_solution_pool else 0),
                        duplicate_pattern_count=duplicates,
                    )
                if added_feasible_exclusion:
                    self.model.update()
                    continue
                return FairnessSeparationResult(
                    status=last_status,
                    has_incumbent=has_incumbent,
                    objective=last_objective,
                    objective_bound=last_bound,
                    mip_gap=last_gap,
                    runtime=time.perf_counter() - start,
                    requested_mip_gap=float(mip_gap),
                    robust_feasibility_certified=False,
                    certification_reason="candidate_scenario_not_fixed_lp_certified",
                    candidate_active_deviations=last_active,
                    fixed_scenario_certificate=last_fixed,
                    separation_model_build_runtime=build_runtime,
                    separation_optimize_runtime=optimize_runtime,
                    pool_candidate_count=(pool_candidates if use_solution_pool else 0),
                    duplicate_pattern_count=duplicates,
                )
        finally:
            if temporary_exclusions:
                self.model.remove(temporary_exclusions)
                self.model.update()

    def dispose(self) -> None:
        if not self._disposed:
            self.model.dispose()
            self._disposed = True


class FairnessScalabilitySeparator:
    """Compose persistence, current-point cache certification, and batching."""

    def __init__(
        self,
        instance: InventoryInstance,
        *,
        strategy: str,
        gamma: int,
        feasibility_tolerance: float,
        output_flag: bool,
    ) -> None:
        self.strategy = validate_scalability_strategy(strategy)
        self.cache = (
            CertifiedScenarioCache()
            if self.strategy in {PERSISTENT_CACHE, PERSISTENT_CACHE_BATCH5}
            else None
        )
        self.persistent = (
            PersistentFairnessSeparation(
                instance,
                gamma=gamma,
                feasibility_tolerance=feasibility_tolerance,
                output_flag=output_flag,
            )
            if self.strategy != SINGLE_CUT
            else None
        )
        self.instance = instance
        self.gamma = int(gamma)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.output_flag = bool(output_flag)

    @property
    def max_cuts(self) -> int:
        return 5 if self.strategy == PERSISTENT_CACHE_BATCH5 else 1

    def separate(
        self,
        *,
        y_values: list[float],
        x_values: list[list[float]],
        t_value: float,
        cost_budget_value: float,
        mip_gap: float,
        time_limit: float,
    ) -> FairnessSeparationResult:
        start = time.perf_counter()
        cache_batch: CertifiedCacheBatch | None = None
        if self.cache is not None and self.cache.size:
            cache_batch = self.cache.certify_current_point(
                self.instance,
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
                cost_budget_value=cost_budget_value,
                time_limit=time_limit,
                feasibility_tolerance=self.feasibility_tolerance,
                max_cuts=self.max_cuts,
                output_flag=self.output_flag,
            )
            if cache_batch.cuts:
                return FairnessSeparationResult(
                    status="certified_cache_candidate",
                    has_incumbent=True,
                    objective=None,
                    objective_bound=None,
                    mip_gap=None,
                    runtime=cache_batch.runtime,
                    requested_mip_gap=float(mip_gap),
                    robust_feasibility_certified=False,
                    certification_reason="cached_patterns_recertified_at_current_point",
                    cut=cache_batch.cuts[0],
                    cuts=cache_batch.cuts,
                    candidate_active_deviations=cache_batch.cuts[0].active_deviations,
                    cut_certificate_source="current_point_fixed_scenario_normalized_farkas_lp",
                    cache_candidate_count=cache_batch.candidate_count,
                    cache_hit_count=cache_batch.hit_count,
                    certified_cached_cut_count=cache_batch.certified_cut_count,
                    certified_batch_cut_count=(
                        len(cache_batch.cuts)
                        if self.strategy == PERSISTENT_CACHE_BATCH5
                        else 0
                    ),
                )
        remaining = float(time_limit) - (time.perf_counter() - start)
        if self.strategy == SINGLE_CUT:
            from .robust_regional_fairness import separate_robust_fairness_feasibility

            result = separate_robust_fairness_feasibility(
                self.instance,
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
                cost_budget_value=cost_budget_value,
                gamma=self.gamma,
                mip_gap=mip_gap,
                time_limit=max(1.0e-3, remaining),
                feasibility_tolerance=self.feasibility_tolerance,
                output_flag=self.output_flag,
            )
        else:
            assert self.persistent is not None
            result = self.persistent.separate(
                y_values=y_values,
                x_values=x_values,
                t_value=t_value,
                cost_budget_value=cost_budget_value,
                mip_gap=mip_gap,
                time_limit=max(1.0e-3, remaining),
                max_cuts=self.max_cuts,
                use_solution_pool=self.max_cuts > 1,
                output_flag=self.output_flag,
            )
        if self.cache is not None:
            for cut in result.cuts or ([result.cut] if result.cut is not None else []):
                self.cache.add(cut.active_deviations)
        if cache_batch is not None:
            result = replace(
                result,
                cache_candidate_count=cache_batch.candidate_count,
                cache_hit_count=cache_batch.hit_count,
                certified_cached_cut_count=cache_batch.certified_cut_count,
            )
        return result

    def dispose(self) -> None:
        if self.persistent is not None:
            self.persistent.dispose()
