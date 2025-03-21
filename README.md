# Convective thunderstorm wind gusts

This repository comes with paper : **Improving Predictions of Convective Storm Wind Gusts through Statistical Post-Processing of Neural Weather Models**.

## Initial scientific motivation

Issuing timely severe weather warnings helps mitigate potentially disastrous consequences. Recent advancements in Neural Weather Models (NWMs) offer a computationally inexpensive and fast approach for forecasting atmospheric environments on a 0.25$^\circ$ global grid. For thunderstorms, these environments can be empirically post-processed to predict wind gust distributions at specific locations. With the Pangu-Weather NWM, we apply a hierarchy of statistical and deep learning post-processing methods to forecast hourly wind gusts up to three days ahead. We constrain our probabilistic forecasts using Generalised Extreme-Value distributions across five regions in Switzerland. Using a convolutional neural network to post-process the predicted convective environment's spatial patterns yields the best results, outperforming direct forecasting approaches across lead times and wind gust speeds. Our results confirm the added value of NWMs for extreme wind forecasting, especially for designing more responsive early-warning systems.