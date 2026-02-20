# StormCast Module: Architectural and Scientific Overview

## 1. Introduction
The StormCast core engine is a sophisticated, unified framework designed for short-term storm motion prediction (nowcasting). It seamlessly integrates environmental diagnostic data (e.g., RAP profiles) with real-time radar observations, applying advanced tracking, filtering, and blending techniques to produce highly accurate predictive tracks with quantified uncertainty bounds. The module combines physical meteorology principles with robust state estimation mathematics.

## 2. Core Orchestration (`core.py`)
The `StormCastEngine` serves as the central orchestrator for the forecasting pipeline. For each storm cell, it maintains a unified chronological state history and environmental profile. The lifecycle of a forecast update follows these sequential stages:
1. **Observation Ingestion**: Receives new physical $(x, y)$ positions and calculates the storm core height ($H_{core}$) from the 30 dBZ and 50 dBZ echo top heights.
2. **State Estimation**: Updates the internal Kalman filter by digesting the latest position to refine velocity tracking.
3. **Environmental Diagnostics**: Computes steering flow and Bunkers right-mover motion based on the current environment and $H_{core}$.
4. **Motion Blending**: Fuses smoothed historical observations with the physical environmental estimates using dynamic, maturity-based weighting.
5. **Forecast Advection**: Projects the final blended motion vector forward across user-specified lead times.
6. **Uncertainty Quantification**: Calculates statistically rigorous uncertainty tracking cones based on tracking history, trajectory jitter, and environmental variance.

## 3. State Estimation: Advanced Kalman Filtering (`kalman.py`)
At the heart of the observation tracking is a custom 4-dimensional Kalman Filter estimating the state vector $[x, y, u, v]$. 
- **Forecast Smoothing (Alpha Filter)**: To prevent erratic velocity over-corrections between radar scans, the prediction step applies an exponential smoothing factor ($\alpha$) to the velocity components.
- **Adaptive Process Noise**: Process noise matrices ($Q$) can be scaled dynamically to account for erratic storm behavior or rapid acceleration.
- **History-Scaled Observation Uncertainty**: The observation noise matrix ($R$) scales inversely with the track history count. A storm tracked for a longer duration is trusted more (smaller $R$), mathematically tightening the Kalman gain and relying heavily on the established velocity trend.

## 4. Environmental Diagnostics (`diagnostics.py`)
The environment acts as the physical boundary condition guiding the storm, heavily parameterized by the storm's vertical depth.
- **Height-Adaptive Steering ($V_{mean}^*$)**: Instead of relying on a static steering level (e.g., 700 mb or 500 mb), StormCast computes a dynamically weighted mean wind vector. Gaussian weight distributions, centered around the storm core height $H_{core}$, assign influence bounds to wind vectors at discrete pressure levels. 
- **Effective Shear**: Computed by finding the wind vector difference between the base level (typically 850 mb) and an adaptively selected top level (ranging from 850 mb up to 250 mb depending on $H_{core}$).
- **Bunkers Deviant Motion ($V_{bunkers}^*$)**: Supercellular storms deviate from the mean wind due to dynamic pressure perturbations driven by vertical wind shear. StormCast locates the perpendicular deflection vector relative to the effective shear, and applies a height-scaled Bunkers deviation magnitude. Shallow convection uses a severely reduced deviation multiplier compared to deep convection.

## 5. Motion Blending Engine (`blending.py`)
StormCast acknowledges that radar observations contain jitter but represent direct ground truth, while environmental models are perfectly smooth but represent indirect physical proxies. 
The blended forecast velocity is given by:
$$ V_{final} = w_{obs} V_{obs} + w_{mean} V_{mean}^* + w_{bunkers} V_{bunkers}^* $$
Crucially, these weights are **dynamic**:
- **Storm Maturity**: Newly initiated storms heavily favor the specific environmental vectors ($V_{mean}^*$, $V_{bunkers}^*$) as the radar track history is statistically insufficient. Mature, well-tracked storms flip the weights, confidently favoring the smoothed radar observations ($V_{obs}$).
- **Storm Depth**: Shallow storms actively suppress the Bunkers influence, favoring the simple mean wind advection. Deep storms increase the Bunkers influence as organized supercellular dynamics become physically probable.
- **Shear Magnitude**: Higher environmental shear shifts weight away from pure tracking toward the Bunkers deviation model to pre-emptively catch right-moving propagation.

(Note: Prior to blending, historical raw observations ($V_{obs}$) are smoothed via an exponential temporal filter to eliminate high-frequency spatial jitter from the radar tracking centroid).

## 6. Forecast Generation & Advection (`forecast.py`)
The `forecast_with_uncertainty` module performs linear advection based on the final blended state vector $[x, y, u, v]$, generating geographic prediction points at predefined intervals (typically 15, 30, 45, and 60 minutes). The maximum reliable lead time is empirically capped at 60 minutes, mirroring the scientifically accepted degradation limit of persistence-based severe storm nowcasting.

## 7. Uncertainty Quantification (`uncertainty.py`)
Predictability inherently decreases with time, and StormCast models this explicitly to draw conservative threat visualization cones:
- **Velocity Covariance Setup**: Total systemic velocity variance ($\sigma_{total}^2$) is modeled as the sum of theoretical observational error ($\sigma_{obs}^2$), environmental data error ($\sigma_{env}^2$), and a mathematical tracking error ($\sigma_{hist}^2$).
- **History Decay Function**: The tracking error ($\sigma_{hist}$) decays non-linearly with the number of tracking samples $N$: $\sigma_{hist} = \sigma_{min} + \frac{\sigma_{range}}{N^\alpha}$. Real-time motion "jitter" (the standard deviation of historical velocity vectors) adds a further penalty to the base scalar.
- **Position Uncertainty Propagation**: Geolocation variance grows quadratically with the forecast lead time $\Delta t$. The propagation equation is $P_{pos}(t+\Delta t) = P_{pos}(t) + \sigma_{vel}^2 \Delta t^2$.
- **Error Ellipses**: The resulting location uncertainty bounds form expanding ellipses derived using the appropriate Chi-Squared distribution values for a 2-degree-of-freedom normal distribution, yielding statistical confidence regions (e.g., standard 95% confidence intervals).
