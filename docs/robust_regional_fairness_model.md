# Robust regional service fairness model

## Scope

This model concerns **regional service fairness** in the existing robust
inventory network.  It does not contain demographic, income, age, race, or
vulnerability attributes and therefore cannot support claims about socially
disadvantaged groups.  The existing `transport_cost[i][r][j]` is a unit cost;
it is not a physical distance.

The frozen baseline is `joint_v1_core_point_strengthened`.  Its robust model,
uncertainty set, precision parameters, and core-point implementation are not
changed.  For each instance the baseline first supplies a certified robust
cost, denoted by \(C^*\).

## Max-min service formulation

Let \(i\), \(r\), and \(j\) index warehouses, regions, and products.  The
first-stage variables are warehouse openings \(y_i\) and inventories
\(x_{ij}\).  For scenario \(z\), the adaptive recourse variables are shipment
\(q_{irj}(z)\), shortage \(u_{rj}(z)\), and service violation \(e_j(z)\).

For an applicable region,

\[
D_r(z)=\sum_j d_{rj}(z),\qquad
U_r(z)=\sum_j u_{rj}(z),\qquad
FR_r(z)=1-U_r(z)/D_r(z).
\]

Regions with \(D_r(z)=0\) are marked not applicable and excluded from the
scenario fairness comparison.  Define the worst regional shortage rate

\[
T=\max_{z\in\mathcal U_\Gamma}\max_{r:D_r(z)>0} U_r(z)/D_r(z).
\]

For a frozen price-of-fairness allowance \(\rho\), set

\[
B_\rho=(1+\rho)C^*.
\]

The primary model minimizes \(T\), subject to the unchanged first-stage
constraints and, for every uncertainty scenario, the unchanged recourse
constraints plus

\[
c^{\mathsf T}_{\rm first}(y,x)+c^{\mathsf T}_{\rm recourse}
 (q(z),u(z),e(z))\le B_\rho,
\]

\[
\sum_j u_{rj}(z)\le T D_r(z),\qquad D_r(z)>0.
\]

The same \((q(z),u(z),e(z))\) appears in both constraints.  Combining a
cost-optimal recourse from one solve with a fairness-optimal recourse from
another solve is prohibited and is not implemented.

The reported robust minimum fill rate is \(1-T\).  WGap, WWD, and the
demand-weighted mean fill rate remain reporting metrics rather than the sole
objective, avoiding a leveling-down objective that can improve equality by
reducing high service.

After obtaining \(T^*\), the frozen optional lexicographic stage adds
\(T\le T^*+10^{-7}\) and minimizes the actual robust total cost.  This stage
does not alter the primary max-min objective.

## Farkas separation and valid cuts

For fixed \((\bar y,\bar x,\bar T)\), scenario demand \(d\), and budget
\(B_\rho\), write recourse feasibility as \(Av\le b\), \(v\ge0\), using:

- demand: \(-\sum_iq_{irj}-u_{rj}\le-d_{rj}\);
- supply: \(\sum_rq_{irj}\le x_{ij}\);
- product service: \(\sum_ru_{rj}-e_j\le s_j(d)\);
- shared cost: \(c_{q}q+c_{u}u+c_e e\le B_\rho-c_{\rm first}(y,x)\);
- regional service: \(\sum_ju_{rj}\le TD_r(d)\).

Associate nonnegative multipliers \(a_{rj},b_{ij},c_j,k,\ell_r\) with these
rows.  A normalized Farkas ray satisfies

\[
-a_{rj}+b_{ij}+k c^q_{irj}\ge0,
\]

\[
-a_{rj}+c_j+k c^u_{rj}+\ell_r\ge0,
\]

\[
-c_j+k c^e_j\ge0,
\]

and all multipliers are nonnegative.  Every such ray yields the master cut

\[
\begin{aligned}
0\le{}&-\sum_{rj}d_{rj}a_{rj}
+\sum_{ij}x_{ij}b_{ij}
+\sum_j s_j(d)c_j\\
&+\bigl(B_\rho-c_{\rm first}(y,x)\bigr)k
+T\sum_rD_r(d)\ell_r.
\end{aligned}
\]

The implementation maximizes the negative of this expression over both the
normalized ray and the binary budgeted-deviation pattern.  Products of a
binary deviation and a ray multiplier use exact McCormick constraints because
the normalized multipliers lie in \([0,1]\).  Thus the separation problem is
a MILP and does not put all extreme-point recourse models in the master.

A separation incumbent with positive violation produces a valid cut.  Only a
separation objective bound at or below the frozen feasibility tolerance can
certify robust feasibility.  The restricted/incumbent value never certifies
feasibility by itself.

## Bounds and core-point boundary

The master objective bound is a valid lower bound on \(T^*\).  A candidate
\(T\) becomes an upper bound only after robust feasibility is certified by the
separation bound.  Termination requires the frozen global relative gap and a
zero-gap final certification solve.

The baseline calculation of \(C^*\) continues to use the frozen V3
Magnanti-Wong-type core-point strengthened recourse cuts.  New fairness cuts
depend on \((y,x,T)\) and arise from a Farkas cone, not the frozen affine
recourse-cut family.  The old core-point auxiliary LP is therefore not applied
to fairness cuts; doing so without a new validity proof would be unsafe.  This
is the meaning of retaining the V3 mechanism only where it is mathematically
valid.

## Extensive-form oracle

`solve_fairness_extensive_form` explicitly creates one recourse block per
extreme point.  It exists only for hand-built tiny-instance verification.  It
is compared against the separation algorithm for \(T\), first-stage decisions,
cost caps, scenario policies, and robust bounds.  It is not the formal
medium-large or large solution method.
