get_variable_predictors_current <- function(predictor) {
    predictors <- c(
        hash('name'=predictor, 'type'=1, 'function'=paste0(predictor,' + prev_',predictor,' + prev2_',predictor,' + prev3_',predictor))
    )

    return(predictors)
}

get_variable_predictors_all <- function(predictor) {
    predictors <- get_variable_predictors_current(predictor)
    return(predictors)
}
