
get_surprisal_predictors <- function() {
  predictors <- c(
    # All, but one
    hash('name'='prev3_surprisal', 'type'=-3, 'function'='surprisal + prev_surprisal + prev2_surprisal'),
    hash('name'='prev2_surprisal', 'type'=-3, 'function'='surprisal + prev_surprisal + prev3_surprisal'),
    hash('name'='prev_surprisal', 'type'=-3, 'function'='surprisal + prev2_surprisal + prev3_surprisal'),
    hash('name'='surprisal', 'type'=-3, 'function'='prev_surprisal + prev2_surprisal + prev3_surprisal'),

    # # Surprisals + entropy
    # hash('name'='prev3_surprisal', 'type'=-4, 'function'='surprisal + entropy + prev_surprisal + prev2_surprisal'),
    # hash('name'='prev2_surprisal', 'type'=-4, 'function'='surprisal + entropy + prev_surprisal + prev3_surprisal'),
    # hash('name'='prev_surprisal', 'type'=-4, 'function'='surprisal + entropy + prev2_surprisal + prev3_surprisal'),
    # hash('name'='surprisal', 'type'=-4, 'function'='entropy + prev_surprisal + prev2_surprisal + prev3_surprisal'),

    # # Surprisals + renyi
    # hash('name'='prev3_surprisal', 'type'=-5, 'function'='surprisal + renyi_0.50 + prev_surprisal + prev2_surprisal'),
    # hash('name'='prev2_surprisal', 'type'=-5, 'function'='surprisal + renyi_0.50 + prev_surprisal + prev3_surprisal'),
    # hash('name'='prev_surprisal', 'type'=-5, 'function'='surprisal + renyi_0.50 + prev2_surprisal + prev3_surprisal'),
    # hash('name'='surprisal', 'type'=-5, 'function'='renyi_0.50 + prev_surprisal + prev2_surprisal + prev3_surprisal'),

    # All surprisals
    hash('name'='surprisal', 'type'=1, 'function'='surprisal+prev_surprisal+prev2_surprisal+prev3_surprisal')
  )
#   predictors <- c(predictors, get_variable_predictors_next('surprisal'))

  return(predictors)
}


get_variable_predictors_prev <- function(predictor_base) {
    predictor <- paste0('prev_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('surprisal + prev_surprisal + ',predictor,' + prev2_surprisal + prev3_surprisal')),
        hash('name'=predictor, 'type'=6, 'function'=paste0('surprisal + ',predictor,' + prev2_surprisal + prev3_surprisal'))
        # hash('name'=predictor, 'type'=9, 'function'=paste0('surprisal + entropy + prev_surprisal + ',predictor,' + prev2_surprisal + prev3_surprisal')),
        # hash('name'=predictor, 'type'=11, 'function'=paste0('surprisal + renyi_0.50 + prev_surprisal + ',predictor,' + prev2_surprisal + prev3_surprisal'))
    )

    return(predictors)
}

get_variable_predictors_prev2 <- function(predictor_base) {
    predictor <- paste0('prev2_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('surprisal + prev_surprisal + prev2_surprisal + ',predictor,' + prev3_surprisal')),
        hash('name'=predictor, 'type'=6, 'function'=paste0('surprisal + prev_surprisal + ',predictor,' + prev3_surprisal'))
        # hash('name'=predictor, 'type'=9, 'function'=paste0('surprisal + entropy + prev_surprisal + prev2_surprisal + ',predictor,' + prev3_surprisal')),
        # hash('name'=predictor, 'type'=11, 'function'=paste0('surprisal + renyi_0.50 + prev_surprisal + prev2_surprisal + ',predictor,' + prev3_surprisal'))
    )

    return(predictors)
}

get_variable_predictors_prev3 <- function(predictor_base) {
    predictor <- paste0('prev3_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('surprisal + prev_surprisal + prev2_surprisal + prev3_surprisal + ',predictor)),
        hash('name'=predictor, 'type'=6, 'function'=paste0('surprisal + prev_surprisal + prev2_surprisal + ',predictor))
        # hash('name'=predictor, 'type'=9, 'function'=paste0('surprisal + entropy + prev_surprisal + prev2_surprisal + prev3_surprisal + ',predictor)),
        # hash('name'=predictor, 'type'=11, 'function'=paste0('surprisal + renyi_0.50 + prev_surprisal + prev2_surprisal + prev3_surprisal + ',predictor))
    )

    return(predictors)
}

# get_variable_predictors_next <- function(predictor_base) {
#     predictor <- paste0('next_',predictor_base)

#     predictors <- c(
#         hash('name'=predictor, 'type'=5, 'function'=paste0('surprisal + prev_surprisal + prev2_surprisal + prev3_surprisal + ',predictor))
#         # hash('name'=predictor, 'type'=9, 'function'=paste0('surprisal + entropy + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal')),
#         # hash('name'=predictor, 'type'=11, 'function'=paste0('surprisal + renyi_0.50 + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal'))
#     )

#     return(predictors)
# }

get_variable_predictors_current <- function(predictor) {
    predictors <- c(
        hash('name'=predictor, 'type'=1, 'function'=paste0(predictor,' + prev_',predictor,' + prev2_',predictor,' + prev3_',predictor)),
        hash('name'=predictor, 'type'=5, 'function'=paste0('surprisal + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal')),
        hash('name'=predictor, 'type'=6, 'function'=paste0(predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal'))
        # hash('name'=predictor, 'type'=9, 'function'=paste0('surprisal + entropy + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal')),
        # hash('name'=predictor, 'type'=10, 'function'=paste0('surprisal + entropy + next_entropy + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal')),
        # hash('name'=predictor, 'type'=11, 'function'=paste0('surprisal + renyi_0.50 + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal')),
        # hash('name'=predictor, 'type'=12, 'function'=paste0('surprisal + renyi_0.50 + next_renyi_0.50 + ',predictor,' + prev_surprisal + prev2_surprisal + prev3_surprisal'))
    )

    return(predictors)
}

get_variable_predictors_all <- function(predictor) {
    predictors_current <- get_variable_predictors_current(predictor)
    predictors_prev <- get_variable_predictors_prev(predictor)
    predictors_prev2 <- get_variable_predictors_prev2(predictor)
    predictors_prev3 <- get_variable_predictors_prev3(predictor)

    predictors <- c(predictors_current, predictors_prev,
                    predictors_prev2, predictors_prev3)
    return(predictors)
}
