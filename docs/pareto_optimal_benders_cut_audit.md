# Pareto-optimal Benders cut audit

## Existing optimality cut

For a fixed demand pattern `z`, the recourse dual feasible set is

\[
\lambda_{rj}-\mu_{ij}\le c_{irj},\qquad
\lambda_{rj}-\nu_j\le p_{rj},\qquad
0\le\nu_j\le e_j,
\]

with nonnegative `lambda`, `mu`, and `nu`. Every feasible dual point defines

\[
\theta\ge \alpha_z(\lambda,\nu)-\sum_{i,j}\mu_{ij}x_{ij},
\]

where

\[
\alpha_z=\sum_{r,j}d_{rj}(z)\lambda_{rj}
-\sum_j(1-a_j)\left(\sum_r d_{rj}(z)\right)\nu_j.
\]

This is the ordinary cut generated from the robust dual MILP incumbent.

## Audit of the existing core-point implementation

The existing `core_point` mode fixes the robust MILP pattern, solves the
fixed-pattern dual at the current point, and then maximizes its affine value at
the historical core point. Its second LP imposes only

\[
\alpha+\beta^\top x^k\ge q_k-\delta,
\qquad
\delta=\text{absTol}+\text{relTol}\max(1,|q_k|).
\]

It is a valid epsilon-strengthening scheme, but it is not a strict
Magnanti-Wong construction because it permits loss of current-point tightness.
The historical convex average was feasible, but it was not initialized with a
proved relative-interior point. The auxiliary solution may also coincide with
the ordinary dual optimum; in that case the acceptance test falls back to the
ordinary cut.

## New `pareto_optimal_mw` mode

For the active pattern `z^k`, stage 1 computes

\[
q_k=\max_{\pi\in D(z^k)}\Psi(\pi,x^k).
\]

Stage 2 solves

\[
\max_{\pi\in D(z^k)}\Psi(\pi,\bar x)
\quad\text{subject to}\quad
\Psi(\pi,x^k)=q_k.
\]

No coefficient perturbation is used. The equality keeps the new dual solution
on the current optimal face. The core point is initialized analytically with
strict slack in variable bounds, logic, capacity, and budget constraints of
the continuous master relaxation. The existing positive convex-combination
update then preserves relative-interior membership.

For every `pi` feasible in `D(z^k)`, weak duality gives

\[
\Psi(\pi,x)\le Q_{z^k}(x)\le\max_z Q_z(x)=Q(x)
\]

for every first-stage-feasible `x`. The cut is therefore globally valid. If
`z^k` is an active worst-case pattern, the optimal-face equality also gives
`Psi(pi,x^k) = Q(x^k)`, so the cut is tight at the current point. Numerical
acceptance separately verifies dual feasibility and equality within the
existing core-point numerical tolerance; auxiliary objectives never update the
incumbent upper bound.
