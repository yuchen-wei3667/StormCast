# StormCast: A Design Framework for Predicting Thunderstorm Motion Using Observed Storm Tracks and RAP Wind Profiles

## Abstract
This paper presents a physically grounded yet operationally practical framework for predicting thunderstorm motion by blending observed storm displacement vectors with environmental wind information from the Rapid Refresh (RAP) model. The design targets traditional spring and summer convection and emphasizes short-term forecast skill (0–60 minutes), robustness to noisy observations, and adaptability across storm modes. Using storm-centric motion estimates derived from radar tracking and high-resolution multi-level environmental winds (1000 mb to 100 mb), the framework decomposes storm motion into advective and propagative components, constrains forecasts using deep-layer shear, and produces probabilistic motion guidance suitable for warning operations and object-based nowcasting systems.

## 1. Introduction
Thunderstorm motion prediction is fundamental to severe weather operations, influencing warning lead time, impact-based decision support, and downstream hazards such as flash flooding. While numerical weather prediction (NWP) provides valuable environmental context, short-term storm motion is best constrained by recent observations. Conversely, purely kinematic extrapolation of storm tracks fails when storms deviate due to propagation, boundary interactions, or storm-scale dynamics.

This paper proposes a hybrid design that fuses observed storm motion with environmental wind structure from RAP analyses. The goal is not long-range prediction, but high-skill nowcasting over short lead times where warning decisions are made. The framework is intentionally modular, allowing implementation in research prototypes or operational systems.

## 2. Design Objectives
The system is designed to satisfy the following objectives:
- **Physical Consistency:** Forecast motion should be meteorologically plausible given the environmental wind and shear profile.
- **Observational Anchoring:** Recent storm motion must dominate the forecast when motion is steady and well-observed.
- **Adaptability:** The framework must accommodate discrete cells, supercells, and early-stage convective systems.
- **Stability:** Noise in radar-derived positions should not produce erratic motion forecasts.
- **Operational Efficiency:** Computation must be fast enough for real-time use with frequent updates.

## 3. Data Inputs

### 3.1 Storm-Centric Observations
Storm motion is derived from a time series of storm object positions obtained via radar-based tracking algorithms. For each time step, tracking provides zonal displacement ($dx$), meridional displacement ($dy$), and time interval ($dt$). These yield an observed storm motion vector:
$$ u_{obs} = \frac{dx}{dt}, \quad v_{obs} = \frac{dy}{dt} $$
To reduce centroid jitter and sampling noise, the observed motion is smoothed using an exponential filter across recent tracking history (e.g., the last 9 observations). Storm core height ($H_{core}$) is concurrently estimated using the average of the 30 dBZ and 50 dBZ echo-top heights.

### 3.2 Environmental Wind Profiles
Environmental winds are extracted from RAP analyses across a comprehensive vertical profile, sampling from 1000 mb up to 100 mb in 25 mb increments. This high-resolution extraction adequately samples the full depth of warm-season convective storms and allows for continuous height-adaptive weighting.

## 4. Environmental Diagnostics

### 4.1 Height-Adaptive Deep-Layer Mean Wind
Rather than a static layer mean, the framework calculates a dynamically weighted "steering flow" ($\vec{V}_{mean}^*$). Wind vectors at each pressure level are weighted by a Gaussian function centered at the storm's current core height ($H_{core}$):
$$ \vec{V}_{mean}^* = \sum_{p=1000}^{100} w_p(H_{core}) \vec{V}_p $$
This ensures that shallow storms are heavily steered by lower tropospheric winds, while deep convection incorporates upper-level jets.

### 4.2 Effective Vertical Wind Shear
Vertical wind shear is recalculated over the effective depth of the storm rather than using a fixed 0–6 km layer:
$$ \vec{S}_{eff} = \vec{V}_{top}(H_{core}) - \vec{V}_{base} $$
where $\vec{V}_{base}$ is typically the 850 mb wind, and $\vec{V}_{top}$ is adaptively selected based on storm depth: 250 mb for deep storms ($H_{core} \geq 10$ km), 500 mb for moderate storms ($H_{core} \geq 5.5$ km), and 700 mb for shallower storms ($H_{core} \geq 3.0$ km).

### 4.3 Bunkers-Type Storm Motion Estimate
Using the mean wind and effective shear vector, a Bunkers-style storm motion estimate is computed:
$$ \vec{V}_{Bunkers}^* = \vec{V}_{mean}^* + D(H_{core}) \hat{n} $$
where $\hat{n}$ is the unit vector perpendicular to the shear (e.g., 90° clockwise for right-movers) and $D$ is a height-scaled deviation magnitude. $D$ is linearly scaled from 3.0 m s⁻¹ for shallow storms ($H_{core} \leq 6$ km) up to the canonical 7.5 m s⁻¹ for deep supercells ($H_{core} \geq 10$ km).

## 5. Motion Decomposition
Observed storm motion is explicitly decomposed:
$$ \vec{V}_{obs} = \vec{V}_{mean}^* + \vec{V}_{prop} $$
where $\vec{V}_{prop}$ represents the propagation vector due to storm-scale processes (cold pools, boundary interactions, updraft regeneration). This decomposition isolates dynamically driven deviations from purely advective environmental steering.

## 6. Motion Blending Strategy
The final forecast vector is a weighted blend of observed motion and physical environmental models:
$$ \vec{V}_{final} = w_o \vec{V}_{obs} + w_m \vec{V}_{mean}^* + w_b \vec{V}_{Bunkers}^* $$
Empirically optimized baseline weights are:
- $w_o = 0.4$ (Observed Motion)
- $w_m = 0.3$ (Mean Steering)
- $w_b = 0.3$ (Bunkers Deviant Motion)

Weights are dynamically adjusted during runtime based on:
- **Storm Maturity:** Tracking history under 3 samples shifts 15% weight from observations toward the environment. Over 10 samples shifts 10% toward observations.
- **Storm Depth:** Shallow storms suppress Bunkers weighting in favor of mean wind; deep storms slightly increase Bunkers influence.
- **Shear Magnitude:** High effective shear ($> 25$ m s⁻¹) shifts weight to the Bunkers deviation vector to anticipate supercellular organization.
- **Stratiform Mode:** In stable environments ($MUCAPE \leq 250$ J/kg) where storms are shallow ($H_{core} \leq 6$ km), Bunkers weighting is negated ($w_b = 0$) and steering weights are increased to prioritize environmental advection.

## 7. Temporal Filtering and State Estimation with Kalman Smoothing
To ensure stability and continuity, the framework is implemented within a 4-dimensional Kalman filtering architecture. 

### 7.1 State Vector and Track Smoothing
The storm state is represented as $\mathbf{x}_k = [x_k, y_k, u_k, v_k]^T$. To condition the noisy centroid coordinate inputs prior to state estimation, position tracks are processed using a Savitzky-Golay filter. This fundamentally reduces spatial tracking jitter without introducing unacceptable latency.

The temporal evolution operates via linear advection:
$$ \mathbf{x}_{k+1} = \mathbf{F}_k \mathbf{x}_k + \mathbf{w}_k $$
Observations of the velocity are exponentially smoothed over time using a low-pass filter coefficient $\alpha$ (e.g., 0.25) to heavily dampen severe scan-to-scan oscillations before blending with the environmental profile.

### 7.2 Explicit Noise Matrices
- **Process Noise ($\mathbf{Q}$):** Adaptively scales depending on the storm's velocity volatility, defaulting to parameters optimized for minimal Mean Absolute Error (MAE) ($Q_{pos} \approx 2000$ m², $Q_{vel} \approx 28.8$ m²/s²).
- **Observation Error ($\mathbf{R}$):** Represents uncertainty in radar-derived positions. This is strongly tied to tracking history, with the standard observation deviation $\sigma_{pos}$ scaling as $\sigma_{base} \times (1.0 + 5.0 / \max(N_{hist}, 1))$. Newer storms have heavily inflated observation uncertainty, forcing the filter to restrict updates and rely on environmental priors.

## 8. Forecast Generation with Adaptive Advection Offset
Linear advection of a purely environmental steering vector often neglects necessary momentum from ongoing storm-scale processes. Conversely, purely projecting raw radar velocities introduces runaway error at longer lead times due to track noise.

To balance these dynamics, the framework calculates an **adaptive advection offset**. This offset ($\vec{V}_{offset}$) defines the systematic short-term bias between the raw smoothed storm motion and the blended environmental steering over a recent tracking history (e.g., the last 12 observation scans).

$$ \vec{V}_{offset} = \frac{1}{N} \sum_{i=k-N}^{k} (\vec{V}_{obs, i} - \vec{V}_{final, i}) $$

This calculated bias is dampened by an empirical weight (e.g., 0.15) to prevent track divergence, and is explicitly added to the blended environmental velocity during the forward spatial projection:

$$ x(t+\Delta t) = x(t) + (u_{final} + u_{offset}) \Delta t $$
$$ y(t+\Delta t) = y(t) + (v_{final} + v_{offset}) \Delta t $$

Forecasts are calculated for standard operational lead times (e.g., 15, 30, 45, and 60 minutes) and empirically clipped at 60 minutes to maintain skill boundaries.

## 9. Uncertainty Representation and Forecast Spread
StormCast explicitly provides probabilistic motion guidance via expanding error ellipses.

### 9.1 Sources of Uncertainty
Uncertainty accumulates from three explicitly modeled sources, combined in quadrature:
$$ \sigma_{total}^2 = \sigma_{obs}^2 + \sigma_{env}^2 + \sigma_{hist}^2 $$
Where $\sigma_{obs}$ models centroid jitter (default 4.0 m/s), and $\sigma_{env}$ models environmental steering variance (default 2.0 m/s).

### 9.2 History-Decayed Tracking Confidence
The history uncertainty term shrinks non-linearly with the number of stable tracks ($N$):
$$ \sigma_{hist} = \sigma_{min} + \frac{\sigma_{range}}{N^\alpha} + (\gamma \times \text{jitter}) $$
Empirically derived defaults use $\sigma_{min} = 1.2$, $\sigma_{range} = 2.5$, $\alpha = 0.5$, and an additive penalty proportional to the standard deviation of recent velocity vectors ($\gamma = 0.1$). This punishes highly erratic tracks with wider uncertainty bounds even if they have long tracking histories.

### 9.3 Position Uncertainty Propagation
Velocity variance forces positional uncertainty to expand quadratically with lead time $\Delta t$:
$$ \mathbf{P}_{pos}(t+\Delta t) = \mathbf{P}_{pos}(t) + \mathbf{P}_v \Delta t^2 $$
Forecast error ellipses are rendered at desired confidence intervals (e.g., 95% via Chi-Squared distributions), producing actionable probabilistic motion cones for operational impact assessments.

## 10. Applicability and Limitations
The proposed design is optimized for traditional spring and summer convection, including discrete cells and supercells. Performance may mathematically degrade for:
- Very shallow or low-topped convection
- Cold-season stratiform systems
- Terrain-dominated flows

## 11. Conclusion
By fusing heavily smoothed and maturity-gated storm observations with a dynamically weighted, height-adaptive environmental wind profile, this framework achieves a remarkable balance between realism and robustness. The probabilistic uncertainty bounds explicitly mirror human forecaster intuition regarding track stability, making the system uniquely suited for next-generation object-based nowcasting.
