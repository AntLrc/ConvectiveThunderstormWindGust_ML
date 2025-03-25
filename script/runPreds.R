library(argparse)

# Parse command line arguments
parser <- ArgumentParser()
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

# Load the predictors
testPredictors <- read.csv(args$test_predictors)

# Load the source file
source(args$source)

# Load the model
model <- readRDS(args$model_file)

# Predict the response variable
predictions <- predictmodel(model, testPredictors)

# Save the predictions
savePredictions(predictions, args$output)

