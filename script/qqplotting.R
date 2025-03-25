library(argparse)

# Parse command line arguments
parser <- ArgumentParser()
parser$add_argument("--source",
                    help = "Path to the source file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--response",
                    help = "Path to the response file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--sample-begin",
                    help = "Beginning of the sample",
                    required = TRUE,
                    type = "integer")
parser$add_argument("--sample-end",
                    help = "End of the sample",
                    required = TRUE,
                    type = "integer")
parser$add_argument("--output",
                    help = "Path to the output file",
                    required = TRUE,
                    type = "character")
# parser$add_argument("--station-name",
#                     help = "Name of the station",
#                     required = TRUE,
#                     type = "character")
args <- parser$parse_args()

# Load the response as data frame
df <- read.csv(args$response)
df <- as.data.frame(df)
vec <- df$wind_speed_of_gust
vec <- vec[args$sample_begin:args$sample_end]

# convert to vector without accessing wind_speed_of_gust


# Load the source file
source(args$source)

# Sample and gev
fit_gev <- fgev(vec)

# Get the parameters of the GEV distribution
eta <- fit_gev$estimate[1]
tau <- fit_gev$estimate[2]
xi <- fit_gev$estimate[3]

# Set the number of simulations
S <- 10^5
l <- length(vec)
alpha <- 0.05

# Simulation
set.seed(1)
sim_gev <- rgev(S * l, loc = eta, scale = tau, shape = xi)

# Compute the bounds
bounds <- comp_boundsoverall(sim_gev, S, l, alpha)
v_sup <- bounds[1, ]
v_inf <- bounds[2, ]
v_quant <- (1:l)/(l + 1)
emp_quant <- sort(vec)
theor_quant <- qgev(v_quant, loc = eta, scale = tau, shape = xi)

# Create the data frame
dat <- cbind(v_inf, emp_quant, pmin(v_sup, 60))
dat <- as.data.frame(dat)

# Graphical parameters for the ggplots
size_numb <- 10
size_txt <- 10
nticks_x <- 6
nticks_y <- 6
graph_par <- c(size_numb, size_txt, nticks_x, nticks_y)

# Kolmogorov-Smirnov test
ks_test <- ks.test(vec, pgev, loc = eta, scale = tau, shape = xi)
title <- sprintf("QQ-plot for %s", args$station_name)
# Plot positioning ks p value at the top left corner
# plot <- gg_QQplot(dat, theor_quant, graph_par) + ggtitle(title) + geom_text(x = 5, y = 20, label = TeX(sprintf("Kolmogorov-Smirnov test p-value:\n %.2f", ks_test$p.value)), size = 3)
# plot <- gg_QQplot(dat, theor_quant, graph_par) + ggtitle(title) +\
#     geom_text(x = 0.3*max(theor_quant),
#                 y = 0.8*max(theor_quant),
#                 label = TeX(sprintf("Kolmogorov-Smirnov \ntest p-value:\n %.2f", ks_test$p.value)),
#                 size = 3) +\
#     coord_cartesian(xlim = c(0, max(theor_quant)), ylim = c(0, max(theor_quant)))
plot <- gg_QQplot(dat, theor_quant, graph_par) + coord_cartesian(xlim = c(0, max(theor_quant)), ylim = c(0, max(theor_quant)))
ggsave(args$output, plot, width = 6, height = 6, units = "cm")




