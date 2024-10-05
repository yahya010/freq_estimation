get_baselines <- function() {
  baselines <- c(
    hash('name'='empty', 'function'=''),
    hash('name'='full_surprisal', 'function'='+surprisal_buggy+prev_surprisal_buggy+prev2_surprisal_buggy+prev3_surprisal_buggy')
  )

  return(baselines)
}