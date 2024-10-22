#!/usr/bin/env Rscript

source("src/h02_rt_model/r_utils.R")
source("src/h02_rt_model/predictors.R")
source("src/h02_rt_model/baselines.R")

# Get command line arguments
list[input_fname,output_fname,params_output_fname_base,merge_workers,is_linear] <- get_args()
is_linear = as.logical(is_linear)
merge_workers = as.logical(merge_workers)
n_folds = 10
tgt_var = 'time'

set.seed(42)
df <- load_and_preprocess_data(input_fname, merge_workers)

if (is_linear) {
  predictors <- c()

  # variables <- c('surprisal', 'surprisal_buggy')
  variables <- c('freq', 'freq_full_wiki', 'freq_3_4_wiki', 'freq_half_wiki', 'freq_1_5_wiki', 'freq_1_10_wiki')
  for (variable in variables) {
    predictors <- c(predictors, get_variable_predictors_all(variable))
  }
} else {
  sys.exit()
}

print(paste0('Processing dataset ',input_fname))

# Get baseline log likelihood
if (is_linear & merge_workers) {
  # essential_predictors <- 'word_len*freq + prev_freq*prev_word_len + prev2_freq*prev2_word_len + prev3_freq*prev3_word_len'
  essential_predictors <- 'word_len + prev_word_len + prev2_word_len + prev3_word_len'
  baselines <- get_baselines()

  baseline_llhs <- hash()
  for (baseline in baselines) {
    baseline_name <- baseline[['name']]
    baseline_function <- baseline[['function']]

    baseline_llhs[[baseline_name]] <- get_baseline_score(tgt_var, essential_predictors, baseline_function)
  }
} else if (is_linear && !merge_workers) {
  sys.exit()
} else if (!is_linear && merge_workers) {
  sys.exit()
} else if (!is_linear && !merge_workers) {
  sys.exit()
}


# Create container for results
full_diffs <- list()
full_diffs['predictor'] <- c()
full_diffs['predictor_type'] <- c()
full_diffs['name'] <- c()
full_diffs['fold'] <- c()
for(baseline_name in keys(baseline_llhs)) {
  full_diffs[paste0('diff_',baseline_name)] <- c()
}

for(i in 1:length(predictors)){
  predictor = predictors[[i]][['function']]
  predictor_type = predictors[[i]][['type']]
  name = predictors[[i]][['name']]

  print(paste0('Processing dataset ',input_fname,' with predictor ',predictor))
  # Get loglikelihood of predictor
  if (is_linear && merge_workers) {
    formula = paste0(tgt_var," ~ ",predictor," + ",essential_predictors)
    list[full_llh,full_models] <- lme_cross_val(formula, df, tgt_var, random_effects=FALSE)
  } else if (is_linear && !merge_workers) {
    sys.exit()
  } else if (!is_linear && merge_workers) {
    sys.exit()
  } else if (!is_linear && !merge_workers) {
    sys.exit()
  }

  # Save each fold results
  full_diffs[['fold']] <- c(full_diffs[['fold']], c(1:n_folds))
  full_diffs[['predictor']] <- c(full_diffs[['predictor']], rep(predictor, n_folds))
  full_diffs[['predictor_type']] <- c(full_diffs[['predictor_type']], rep(predictor_type, n_folds))
  full_diffs[['name']] <- c(full_diffs[['name']], rep(name, n_folds))

  # Save diffs against baselines
  for(baseline_name in keys(baseline_llhs)) {
    full_diffs[[paste0('diff_',baseline_name)]] <- c(
      full_diffs[[paste0('diff_',baseline_name)]],
      full_llh - baseline_llhs[[baseline_name]]
    )
  }

  # Save model parameters
  models_params <- c()
  for(j in 1:length(full_models)) {
    models_params <- rbind(models_params, full_models[[j]][[1]])
  }
  params_fname <- paste0(params_output_fname_base,'-predictor_',name,'-type_',predictor_type,'.tsv')
  write.table(models_params,params_fname, quote=FALSE, sep='\t')
}

# Append all dataset results to dataframe as two new columns in it
df_full <- c()
df_full <- cbind(df_full,
  full_diffs[['predictor']], full_diffs[['predictor_type']],
  full_diffs[['name']], full_diffs[['fold']],
  full_diffs[['diff_empty']], full_diffs[['diff_full_surprisal']] 
  )

colnames(df_full) <- c(
  'predictor', 'predictor_type', 'name', 'fold', 
  'diff_empty', 'diff_full_surprisal'
  )

write.table(df_full, output_fname, quote=FALSE, sep='\t')
