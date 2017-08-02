# TimeSeriesES-Cell

This repository contains code for time series analysis, developed in the paper:  
[Time Series Using Exponential Smoothing Cells.](https://arxiv.org/abs/1706.02829)

## Overview
---
Exponential smoothing (ES) techniques such as the Holt-Winters model, break down in challenging situations, including
  * high level of noise
  * large or frequent outliers
  * significant portions of missing data
  * nonstationary features. 

We developed a global approach for fitting time series, formulated as a convex optimization problem. 
The approach is built using the notion of linked ES cells, equipped with robust losses for outlier 
detection and denoising. The links enforce time series structure but allow non-stationary signals.  


## Experiments
---
The paper features two experiments: 
* [synthetic example](https://github.com/UW-AMO/TimeSeriesES-Cell/blob/master/Illustration%20-%20Paper%20-%20Synthetic%20Example.ipynb)
* [twitter user engagement dataset](https://github.com/UW-AMO/TimeSeriesES-Cell/blob/master/Illustration%20-%20Paper%20-%20Twitter%20Data.ipynb) 

## Optimization
---
Right now the problem is built and solved using cvx.py, which does not exploit [special problem structure.](https://arxiv.org/abs/1609.06369) Future releases will feature a custom solver. 
