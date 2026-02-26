# StormCast Baseline Comparison: Symmetric Lead-Time Evaluation

## Overview
This report provides a definitive four-way comparison between the StormCast champion architecture and its fundamental components (Pure Bunkers, Pure Corfidi, and Pure Kinematic extrapolation). All modes were evaluated on the **total population of 51,549 storm tracks**.

## Symmetric Performance Metrics (MAE in km)
| Lead Time | Pure Bunkers | Pure Corfidi | Pure Kinematic | **STORMCAST (Champion)** |
|-----------|--------------|---------------|----------------|--------------------------|
| **15 min** | 9.91         | 8.34          | 7.49           | **6.74**                 |
| **30 min** | 17.87        | 14.57         | 13.34          | **11.41**                |
| **45 min** | 27.19        | 21.69         | 19.92          | **16.80**                |
| **60 min** | 35.43        | 27.75         | 25.84          | **21.62**                |

## Symmetric Hit Rate (%)
| Lead Time | Pure Bunkers | Pure Corfidi | Pure Kinematic | **STORMCAST (Champion)** |
|-----------|--------------|---------------|----------------|--------------------------|
| **15 min** | 68.7%        | 79.1%         | 82.8%          | **86.4%**                |
| **30 min** | 74.1%        | 86.3%         | 87.6%          | **92.7%**                |
| **45 min** | 75.5%        | 88.7%         | 89.4%          | **94.6%**                |
| **60 min** | 75.7%        | 90.0%         | 89.9%          | **95.3%**                |

## Key Observations
- **Corfidi Strength**: The Corfidi Vector (specifically Downshear MCS motion) is a significantly stronger environmental baseline than Bunkers, especially at longer lead times where it matches kinematic hit rates.
- **Kinematic Foundation**: Pure radar displacement logic remains a very strong baseline for individual convective cells, outperforming pure environmental methods in MAE.
- **StormCast Superiority**: The hybrid approach remains the unchallenged global champion, reducing forecast error by **10-16%** compared to radar only, and by **20-40%** compared to environmental steering.
