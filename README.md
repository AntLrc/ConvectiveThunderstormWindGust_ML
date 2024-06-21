# Convective thunderstorm wind gusts

This repository corresponds to my current work for my project at UNIL: **Post-Processing of Precipitation & Wind Gusts from Alpine Thunderstorms**.

## Initial scientific motivation

Motivated by recent developments in machine learning (ML)-based downscaling, this project combines extreme-value theory (EVT) and super-resolution algorithms to develop algorithms that can generate spatially-resolved extremes that have not been seen during training. 

Methodologically, the optimal way of combining EVT distributions, such as Generalized Extreme-Value (GEV) distribution for block maxima and Generalized Pareto Distribution (GPD) for threshold exceedances, with ML/statistical modeling tools for atmospheric applications remains an open question. Current approaches _(see Boulaghiem et al., 2022, or Diaz et al., 2022)_ use EVT to describe marginal distributions and ML to represent the spatial dependence structure (copula). An unresolved issue is how much EVT can constrain ML algorithms without losing the necessary descriptive ability for atmospheric applications. This varies from loose constraints, including no strict parameterization, even of the marginals, to strict constraints involving EVT-based parameterization of spatial dependence structures, e.g., through max-stable or generalized Pareto processes. Another open question concerns the most suitable architectures for statistical constraints. Successful applications have used various algorithms, such as conditional generative adversarial networks _(Stengel et al., 2020)_ and diffusion probabilistic models _(Addison et al., 2022; Mardani et al., 2023)_.

More specifically, this project aims at using publically available data from wind stations along with precipitation data to predict alpine thunderstorm from short-term to medium-term forecasts. THe idea would be to post-process well-known AI weather forecast models _(e.g. PanguWeather)_ to identify thunderstorm formation / evolution.

### Bibliography
 - *Addison, H., Kendon, E., Ravuri, S., Aitchison, L., & Watson, P. A. (2022). Machine learning emulation of a local-scale UK climate model. arXiv preprint arXiv:2211.16116.*
 - *Boulaguiem, Y., Zscheischler, J., Vignotto, E., van der Wiel, K., & Engelke, S. (2022). Modeling and simulating spatial extremes by combining extreme value theory with generative adversarial networks. Environmental Data Science, 1, e5.*
 - *Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer London.*
 - *Davison, A. C., Padoan, S. A., & Ribatet, M. (2012). Statistical modeling of spatial extremes. Statistical Science 27(2), 161–186.*
 - *Diaz, J. L. G., Zadrozny, B., Watson, C., Szwarcman, D., Lima, D., Queiroz, D., … & Jones, A. (2022). Direct Sampling for Extreme Events Generation and Spatial Variability Enhancement of Weather Generators. Authorea Preprints.*
 - *Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C. Y., Liu, C. C., … & Pritchard, M. (2023). Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling. arXiv preprint arXiv:2309.15214.*
 - *Stengel, K., Glaws, A., Hettinger, D., & King, R. N. (2020). Adversarial super-resolution of climatological wind and solar data. Proceedings of the National Academy of Sciences, 117(29), 16805-16815.*