library(evgam)

fitmodel <- function(predictors, response, model = "vglm") {
    # Create a data frame with the predictors and response
    data <- data.frame(predictors, response)
    inputvars <- colnames(predictors)
    targetvar <- colnames(response)

    if (model == "vglm") {
        frml <- as.list(c(as.formula(paste(targetvar, "~", paste(inputvars, collapse = "+"))),
                            as.formula(paste(targetvar, "~", paste(inputvars, collapse = "+"))),
                            as.formula(paste(targetvar, "~", "1"))))
    } else if (model == "vgam") {
        frml <- as.list(c(as.formula(paste(targetvar, "~", paste(sapply(inputvars, function(x) paste0("s(", x, ", k=3)")), collapse = "+"))),
                            as.formula(paste(targetvar, "~", paste(sapply(inputvars, function(x) paste0("s(", x, ", k=3)")), collapse = "+"))),
                            as.formula(paste(targetvar, "~", "1"))))
    } else if (model == "one"){
        frml <- as.list(c(as.formula(paste(targetvar, "~", "1")),
                            as.formula(paste(targetvar, "~", "1")),
                            as.formula(paste(targetvar, "~", "1"))))
    } else {
        stop("Model not recognized")
    }

    # Fit the model
    model <- evgam(formula = frml, data = data, family = "gev")

    return(model)
}

predictmodel <- function(model, newdata) {
    # Predict the response variable
    return(predict(model, newdata, type = "response"))
}
savePredictions <- function(predictions, filename) {
    write.csv(predictions, filename, row.names = FALSE)
}    