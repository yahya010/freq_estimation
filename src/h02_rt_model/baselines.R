get_baselines <- function() {
  baselines <- c(
    hash('name'='empty', 'function'=''),
    # hash('name'='full_surprisal', 'function'='+surprisal_buggy+prev_surprisal_buggy+prev2_surprisal_buggy+prev3_surprisal_buggy')
    hash('name'='full_surprisal', 'function'='+surprisal+prev_surprisal+prev2_surprisal+prev3_surprisal')
  )

  return(baselines)
}
