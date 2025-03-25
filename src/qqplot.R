##### Script to draw QQ plots and associated overall confidence bounds

#### Author: Erwan Koch
#### Last update: 11.03.2024

library(evd)
library(boot)
library(ggplot2)
library(latex2exp)


#### Some functions -----

## Creates the QQ plot with associated confidence bounds
## data must contain the columns v_inf, emp_quant and v_sup
gg_QQplot <- function(data, theor_quant, graph_par) {
  
  size_numb <- graph_par[1] # size of the numbers on the axis
  size_txt <- graph_par[2] # size of the title of the axis.
  nticks_x <- graph_par[3]
  nticks_y <- graph_par[4]
  
  plot <- ggplot(data = data) +
    geom_ribbon(mapping = aes(x = theor_quant, ymin = v_inf, ymax = v_sup), 
                fill = "grey60") +
    geom_abline(intercept = 0, slope = 1) + # y=x line
    geom_point(mapping = aes(x = theor_quant, y = emp_quant), col = "black", 
               size = 1) + # we put geom_point at the end to avoid overprint
    labs(x = TeX('Theoretical quantiles ($m$ $s^{-1}$)'), y = TeX('Empirical quantiles ($m$ $s^{-1}$)')) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = nticks_x)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = nticks_y)) +
    theme(axis.text.x = element_text(size = size_numb), axis.title.x = element_text(size = size_txt),
          axis.text.y = element_text(size = size_numb), axis.title.y = element_text(size = size_txt),
          axis.ticks.length=unit(.15, "cm"))
  
}

## Computes the pointwise envelope
comp_bounds <- function(vec, S, l, alpha){
  m_vec <- matrix(vec, nrow = S, ncol = l) # m stands for matrix
  m_vec_s <- t(apply(m_vec, 1, sort)) # one sorts the lines
  # For each quantile, we take the alpha/2 and 1-alpha/2 values obtained 
  # on the S samples
  qu <- apply(m_vec_s, 2, quantile, probs = c(alpha / 2, 1 - alpha / 2))
  v_inf <- qu[1, ]
  v_sup <- qu[2, ]
  return(rbind(v_inf, v_sup))
}

## Computes the overall envelope
## The 1st line of res corresponds to the sup and the second to the inf
comp_boundsoverall <- function(vec, S, l, alpha){
  m_vec <- matrix(vec, nrow = S, ncol = l) # m stands for matrix
  m_vec_s <- t(apply(m_vec, 1, sort)) # one sorts the lines
  res <- envelope(boot.out = NULL, mat = m_vec_s, level = 1 - alpha, 
                  index = 1:ncol(m_vec_s))$overall
  return(res)
}

# #### Confidence bounds ----

# #vec <- Observations from csv file /work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/delete_me/GVE.csv

# # find parameter of gev distribution
# df <- read.csv("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/data/stationlug.csv", header = TRUE)
# vec <- df$wind_speed_of_gust
# vec <- vec[101:200]
# fit_gev <- fgev(vec)

# eta <- fit_gev$estimate[1]
# tau <- fit_gev$estimate[2]
# xi <- fit_gev$estimate[3]

# S <- 10^5
# l <- length(vec) # number of temporal observations
# alpha <- 0.05

# set.seed(1)
# sim_gev <- rgev(S * l, loc = eta, scale = tau, shape = xi)

# bounds <- comp_boundsoverall(sim_gev, S, l, alpha)
# v_sup <- bounds[1, ]
# v_inf <- bounds[2, ]
# v_quant <- (1:l)/(l + 1)
# emp_quant <- sort(vec)
# theor_quant <- qgev(v_quant, loc = eta, scale = tau, shape = xi)


# dat <- cbind(v_inf, emp_quant, pmin(v_sup,60))
# dat <- as.data.frame(dat)

# # Graphical parameters for the ggplots
# size_numb <- 10 
# size_txt <- 13 
# nticks_x <- 6
# nticks_y <- 6
# graph_par <- c(size_numb, size_txt, nticks_x, nticks_y)

# # KS test
# ks_test <- ks.test(vec, pgev, loc = eta, scale = tau, shape = xi)
# print(ks_test)
# # add p-value to the plot
# # add title
# plot <- gg_QQplot(dat, theor_quant, graph_par) + ggtitle("Lugano meteorological station") + geom_text(x = 5, y = 50, label = TeX(sprintf("KS test p-value: %.2f", ks_test$p.value)), size = 3)
# print(plot)

# # save to delete_me/qqplot.pdf
# ggsave("/work/FAC/FGSE/IDYST/tbeucler/downscaling/alecler1/ConvectiveThunderstormWindGust_ML/figures/LUG_qqplot.png", plot, width = 10, height = 10, units = "cm")
