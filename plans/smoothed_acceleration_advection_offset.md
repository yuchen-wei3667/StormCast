# StormCast Architecture Recommendations: Boosting Accuracy & Reducing MAE

## Executive Summary
After a deep review of the entire StormCast codebase — including core.py, blending.py, kalman.py, diagnostics.py, forecast.py, uncertainty.py, and the verification results — I've identified 7 architectural upgrades that target the dominant sources of forecast error. The current best 30-min MAE is 11.25 km with a 79.3% hit rate. These recommendations are ordered by expected impact.

## Current Architecture Bottlenecks

| Bottleneck | Module | Impact on MAE |
|-----------|--------|---------------|
| Linear-only advection (constant velocity assumption) | forecast.py | High — storms accelerate, curve, deviate |
| No acceleration state in Kalman filter | kalman.py | High — velocity trends are invisible to the tracker |
| Static per-call blending weights | blending.py | Medium — weights don't evolve with real-time forecast error |
| Single-model steering (no ensemble spread) | diagnostics.py | Medium — no hedge against RAP bias |
| No temporal trend in environmental diagnostics | diagnostics.py | Medium — environment is treated as stationary |
| Isotropic uncertainty cones | uncertainty.py | Low (accuracy) — but high for calibration |
| No feedback loop from forecast verification | core.py | Medium — the system never learns from its own errors |

## Recommendation 1: Extended Kalman State — 8D with Acceleration + Advection Offset

### Problem
The current Kalman filter tracks a 4D state [x, y, u, v]. It has zero awareness of velocity trends or persistent velocity biases. A storm consistently accelerating eastward is treated identically to one at constant speed, and a storm systematically deviating from the blended velocity (due to propagation, boundary interactions, or RAP steering bias) has no mechanism to learn that correction.

### Proposed Design
Extend the state vector to **8D**: [x, y, u, v, ax, ay, ou, ov], adding four new tracked components:

- **ax, ay** — acceleration (m/s²), capturing velocity trends
- **ou, ov** — advection offset velocity (m/s), capturing persistent systematic deviations between the filter's predicted motion and the actual observed motion

```diff
- State: [x, y, u, v]                       → 4D
+ State: [x, y, u, v, ax, ay, ou, ov]       → 8D
```

**Transition model:**
$$x_{k+1} = x_k + (u_k + o_{u,k}) \Delta t + \frac{1}{2} a_{x,k} \Delta t^2$$
$$u_{k+1} = u_k + a_{x,k} \Delta t$$
$$a_{x,k+1} = a_{x,k}$$
$$o_{u,k+1} = \lambda \cdot o_{u,k}$$

(Identical structure for y, v, ay, ov.)

The acceleration is modeled as a **random walk** (constant acceleration assumption). The advection offset is modeled as a **decaying random walk** with decay factor λ (e.g., 0.95–0.99), meaning persistent biases are remembered but slowly relax toward zero if the signal disappears. This prevents the offset from locking onto transient jitter while still capturing sustained propagation effects.

**State transition matrix F (8×8):**

```
     x    y    u    v    ax   ay   ou   ov
x  [ 1    0    Δt   0    ½Δt² 0    Δt   0  ]
y  [ 0    1    0    Δt   0    ½Δt² 0    Δt ]
u  [ 0    0    1    0    Δt   0    0    0  ]
v  [ 0    0    0    1    0    Δt   0    0  ]
ax [ 0    0    0    0    1    0    0    0  ]
ay [ 0    0    0    0    0    1    0    0  ]
ou [ 0    0    0    0    0    0    λ    0  ]
ov [ 0    0    0    0    0    0    0    λ  ]
```

**Why advection offset as a Kalman state?** The current blending pipeline computes propagation as a snapshot (`V_obs - V_mean*`) with no memory. By embedding the offset in the Kalman filter itself, the filter learns it from the innovation residuals — if the filter consistently under-predicts position in one direction, the offset accumulates to compensate. This subsumes much of the value of Recommendation 7 (propagation memory) directly into the tracker.

### Expected Impact
- MAE reduction: 8–18% at 30–60 min lead times
- Acceleration captures turning/accelerating storms; offset captures persistent propagation biases
- The decaying offset naturally handles storm lifecycle: builds during active propagation, relaxes during dissipation
- Computational overhead is modest (8×8 matrices vs. 4×4)

### Implementation Complexity: Medium–High
Changes to: kalman.py (8D state vector, 8×8 F/Q/P matrices, observation matrix H expands to 2×8), core.py (state extraction), forecast.py (advection with offset), config.py (new parameters: λ, q_accel, q_offset), types.py (ax, ay, ou, ov fields on StormState)

## Recommendation 2: Quadratic Advection with Offset

### Problem
forecast.py uses pure linear advection:

```python
x_forecast = state.x + state.u * dt
```

This assumes constant velocity, no curvature, and no persistent bias correction. For a storm moving 20 m/s and turning 15° over 30 minutes, linear advection accumulates ~3–5 km of cross-track error by 60 min.

### Proposed Design
Replace with full kinematic advection using all four new state components from Recommendation 1:

$$x(t + \Delta t) = x(t) + (u + o_u) \cdot \Delta t + \frac{1}{2} a_x \cdot \Delta t^2$$

The advection offset `ou` provides a persistent velocity correction applied linearly, while the acceleration `ax` provides the quadratic curvature term. Together they produce a curved, bias-corrected forecast trajectory.

### Expected Impact
- MAE reduction: 5–10% at 30–60 min lead times (larger than pure quadratic because offset captures systematic bias)
- Pairs naturally with Rec. 1 — the 8D Kalman produces all four terms (ax, ay, ou, ov) that this formula consumes
- Zero degradation at short lead times (both terms are small at 15 min)

### Implementation Complexity: Low
Changes to: forecast.py only (swap the advection formula), types.py (ensure ou/ov fields exist on StormState)

## Recommendation 3: Adaptive Online Weight Learning (Bayesian Blending)

### Problem
The current blending module in blending.py uses rule-based weight adjustments based on storm maturity, depth, and shear. These weights are determined by fixed heuristic thresholds (e.g., "if track_history < 3, shift 15% from obs to env"). The system never knows if these heuristics are actually optimal for this specific storm.

### Proposed Design
Implement an online Bayesian weight updater that adjusts blending weights based on rolling forecast verification:

At each scan, compute the 1-step-ahead prediction error for each component:

$e_{obs} = |x_{actual} - x_{predicted_from_obs}|$
$e_{mean} = |x_{actual} - x_{predicted_from_mean}|$
$e_{bunkers} = |x_{actual} - x_{predicted_from_bunkers}|$

Maintain running inverse-error weights: 
$$w_i^{(t)} = \frac{1/e_i^{(t)}}{\sum_j 1/e_j^{(t)}}$$

Exponentially smooth these weights over time to prevent oscillation.

This makes the weights self-tuning per storm rather than relying on pre-set heuristics.

### Expected Impact
- MAE reduction: 5–12% across all lead times
- Eliminates the need for hand-tuned weight adjustment constants
- Automatically handles edge cases (e.g., a well-tracked storm where Bunkers is consistently wrong)

### Implementation Complexity: Medium
Changes to: blending.py (new class/function), core.py (track component-level predictions and errors)

## Recommendation 4: Environmental Trend Extrapolation

### Problem
The current system treats the RAP environment as stationary — whatever the latest wind profile says is used verbatim. But RAP updates every hour, and environments evolve. A strengthening jet streak or approaching shortwave can shift the steering flow 5–10 m/s over 30 minutes.

### Proposed Design
Maintain a temporal buffer of the last 2–3 RAP profiles and compute a linear trend in the environmental vectors:

$$V_{mean}^(t + \Delta t) = V_{mean}^(t) + \frac{dV_{mean}^*}{dt} \cdot \Delta t$$

Apply the trended environmental vector at each forecast lead time rather than the static current value. Gate this with a confidence check: only apply the trend if the temporal gradient is consistent (i.e., the last 2 RAP updates agree on the sign of the change).

### Expected Impact
- MAE reduction: 2–5% at 30–60 min lead times
- Particularly impactful during rapidly evolving mesoscale environments (approaching MCS, dryline surges)
- Negligible in quasi-stationary environments (no harm)

### Implementation Complexity: Low–Medium
Changes to: core.py (buffer RAP profiles), diagnostics.py (compute trended vectors), forecast.py (apply trended vectors at each lead time)

## Recommendation 5: Anisotropic & Flow-Dependent Uncertainty

### Problem
Current uncertainty cones use max(sigma_x, sigma_y) for a circular radius, discarding directional information. Storm position errors are inherently anisotropic — along-track errors grow faster than cross-track errors because velocity uncertainty directly feeds the along-track axis.

### Proposed Design
- Propagate the full 2D covariance matrix instead of scalar uncertainties
- Rotate the covariance into the storm-relative coordinate frame (along-track vs. cross-track)
- Render elliptical cones oriented along the storm motion vector

Additionally, make the uncertainty flow-dependent:

- Higher uncertainty when the storm is entering a region of strong environmental gradients (e.g., near boundaries)
- Lower uncertainty when the storm has been on a straight, stable track

### Expected Impact
- MAE impact: Minimal (this is a calibration improvement, not a deterministic accuracy improvement)
- Hit Rate improvement: 3–8% — better-calibrated cones capture more true positions without over-expanding
- Operationally more useful threat areas

### Implementation Complexity: Medium
Changes to: uncertainty.py (full covariance propagation), forecast.py (elliptical generation), core.py (rotation into storm-relative frame)

## Recommendation 6: Multi-Hypothesis Tracking (Left-Mover / Right-Mover)

### Problem
The current system computes only the right-mover Bunkers estimate, hardcoded in core.py L163:

```python
v_bunkers = compute_bunkers_motion(self.environment, self.current_h_core, right_mover=True)
```

Left-moving supercells, splitting storms, and storm pairs are systematically misforecast.

### Proposed Design
Run parallel hypotheses: compute both right-mover and left-mover Bunkers vectors. At each update:

1. Score each hypothesis against the observed storm motion direction
2. Weight the final blend toward the hypothesis that best matches recent motion
3. For storms with no clear preference (symmetric motion), average both

This can be extended to a full Interactive Multiple Model (IMM) Kalman filter, where multiple Kalman filters run in parallel with different motion models (linear, right-mover, left-mover) and are probabilistically combined.

### Expected Impact
- MAE reduction: 2–5% for the subset of storms that are left-movers or splitters (roughly 10–20% of supercellular storms)
- Overall MAE reduction: 1–3% when averaged across all storm types
- Particularly impactful for high-shear environments

### Implementation Complexity: Medium–High
Changes to: core.py (parallel hypothesis management), diagnostics.py (already supports right_mover=False), new module imm.py if going full IMM

## Recommendation 7: Propagation Memory with Exponential Decay

### Problem
blending.py computes the propagation vector as a single snapshot: V_prop = V_obs - V_mean*. This has no temporal memory — a persistent propagation signal from an outflow boundary is treated the same as momentary centroid jitter.

### Proposed Design
Maintain a running exponentially-smoothed propagation vector that accumulates evidence of persistent storm-scale forcing:

$$V_{prop}^{(t)} = \beta \cdot V_{prop}^{(t-1)} + (1 - \beta) \cdot (V_{obs}^{(t)} - V_{mean}^{*(t)})$$

Where β ≈ 0.6–0.8. Then add this smoothed propagation as a fourth blending component:

$$V_{final} = w_o V_{obs} + w_m V_{mean}^* + w_b V_{bunkers}^* + w_p V_{prop}^{smoothed}$$

This gives the system "memory" of persistent deviations from the environment, which is exactly what cold pool surges, boundary interactions, and updraft cycling produce.

### Expected Impact
- MAE reduction: 3–7% for storms with persistent propagation (outflow-dominated, training storms)
- Particularly powerful for flash-flood-producing quasi-stationary storms
- Negligible overhead

### Implementation Complexity: Low
Changes to: blending.py (add smoothed propagation tracking), core.py (maintain propagation buffer), config.py (add w_prop weight and β parameter)

## Priority Matrix

| # | Recommendation | Expected MAE Reduction | Complexity | Priority |
|---|----------------|------------------------|------------|----------|
| 1 | 8D Kalman (accel + offset) | 8–18% | Medium–High | 🔴 Critical |
| 2 | Quadratic advection w/ offset | 5–10% | Low | 🔴 Critical |
| 3 | Adaptive online weights | 5–12% | Medium | 🟠 High |
| 7 | Propagation memory | 3–7% | Low | 🟠 High |
| 4 | Environmental trends | 2–5% | Low–Med | 🟡 Medium |
| 6 | Multi-hypothesis (IMM) | 1–3% overall | Med–High | 🟡 Medium |
| 5 | Anisotropic uncertainty | Hit Rate +3–8% | Medium | 🔵 Nice-to-have |

## Synergistic Combinations

Recommendations 1 + 2 are highly synergistic and should be implemented together. The 8D Kalman naturally produces all four correction terms (ax, ay, ou, ov) that the quadratic advection formula consumes. Combined, they address the two largest error sources: the constant-velocity assumption and unmodeled persistent velocity biases. Additionally, the advection offset subsumes much of the value of Recommendation 7 (propagation memory) directly into the Kalman tracker.

## Important Note about Recommendation 3

Recommendation 3 (adaptive weights) has the potential to be the single most impactful change because it replaces hand-tuned heuristics with data-driven optimization per storm. However, it requires careful implementation to avoid weight oscillation during the first few scans.

## Suggested Implementation Order

### Phase 1: Foundation
- Rec 1 + 2 (8D Kalman + Quadratic Advection with Offset)

### Phase 2: Intelligence
- Rec 3 + 7 (Adaptive Weights + Propagation Memory)

### Phase 3: Environment
- Rec 4 (Trend Extrapolation)

### Phase 4: Advanced
- Rec 5 + 6 (Anisotropic Uncertainty + IMM)

Each phase should be benchmarked against the current 30-min MAE baseline (11.25 km) using the existing evaluate_forecast_accuracy.py script before proceeding to the next phase.

## Affected Modules Summary

| Module | Changes Required |
|--------|------------------|
| kalman.py | 8D state vector [x,y,u,v,ax,ay,ou,ov], 8×8 F/Q/P matrices, 2×8 H matrix |
| core.py | State extraction (accel + offset), RAP profile buffering, hypothesis management |
| forecast.py | Quadratic advection, trended environmental vector application, elliptical cone generation |
| blending.py | Smoothed acceleration estimation, adaptive weight learning, propagation memory |
| diagnostics.py | Trended environmental vector computation |
| uncertainty.py | Full covariance propagation, storm-relative rotation, flow-dependent scaling |
| config.py | New parameters: λ (offset decay), q_accel, q_offset, plus all other recommendations |
| types.py | Add ax, ay, ou, ov fields to StormState
