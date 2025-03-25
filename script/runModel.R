library(argparse)

# Parse command line arguments
parser <- ArgumentParser()
parser$add_argument("--predictors",
                    help = "Path to the predictors file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--response",
                    help = "Path to the response file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--model",
                    help = "Model to use (vglm or vgam)",
                    required = TRUE,
                    type = "character")
parser$add_argument("--test-predictors",
                    help = "Path to the test predictors file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--output",
                    help = "Path to the output file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--model-file",
                    help = "Path to the model file",
                    required = TRUE,
                    type = "character")
parser$add_argument("--source",
                    help = "Path to the source file",
                    required = TRUE,
                    type = "character")
args <- parser$parse_args()

# Load the predictors and response
predictors <- read.csv(args$predictors)
response <- read.csv(args$response)
testPredictors <- read.csv(args$test_predictors)

# Load the source file
source(args$source)

# Fit the model
model <- fitmodel(predictors, response, model = args$model)

# Predict the response variable
predictions <- predictmodel(model, testPredictors)

# Save the predictions
savePredictions(predictions, args$output)

# Save the model
saveRDS(model, args$model_file)

