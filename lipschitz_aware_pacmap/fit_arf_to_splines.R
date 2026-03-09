library(data.table)
library(arf)
source("lipschitz_aware_pacmap/pac_map.R")

df <- fread("lipschitz_aware_pacmap/data/spline_data.csv")
train <- df[, -c("prob")]
arf <- adversarial_rf(train)
psi <- forde(arf, train)
lik(psi, train, arf = arf, log = FALSE)

pac_map_lipschitz <- function(params,
                              arf,
                              query,
                              evidence,
                              err_tol = 0.05,
                              fail_prob = 0.05,
                              warm_start = FALSE,
                              map_subcircuits_fn = map_subcircuits) {
    if (warm_start) {
        # Get the MMAP solution from each tree
        candidate_set <- map_subcircuits_fn(
            params = params,
            query = query,
            evidence = evidence
        )
        # Ensure columns are in correct order (for some reason they are often not,
        # and this is needed for fsetdiff to work). Your query cols have to start
        # with q for this to work tho!
        col_names <- names(candidate_set)
        numeric_parts <- as.numeric(gsub("q", "", col_names))
        sorted_names <- col_names[order(numeric_parts)]
        setcolorder(candidate_set, sorted_names)
    } else {
        # Create empty data.table first
        candidate_set <- data.table(matrix(character(0), nrow = 0, ncol = length(query)))
        setnames(candidate_set, query)

        # Convert to factors
        candidate_set[, (query) := lapply(.SD, function(x) factor(x, levels = c("0", "1"))),
            .SDcols = query
        ]
    }
    # Need to reclass, for some reason
    params$cat[, val := as.character(val)]

    # Initialization
    m <- candidate_set[, .N]
    max_m <- Inf
    n_batch <- 2000
    final_results <- list()
    prob_total <- 0
    p_hat <- 0
    e_rep <- evidence[rep(1, m)]
    warm_start <- cbind(e_rep, candidate_set)
    col_names <- names(warm_start) # Use warm_start instead of candidate_set
    numeric_parts <- as.numeric(gsub("[^0-9]", "", col_names)) # Extract all numbers, not just after "q"
    sorted_names <- col_names[order(numeric_parts)]
    setcolorder(warm_start, sorted_names)

    old_samples <- warm_start[0]

    # Normalizing constant
    prob_e <- lik(params = params, query = evidence, log = FALSE)

    while (m < max_m) {
        # Update count, draw samples
        m <- m + n_batch
        samples <- forge(params = params, evidence = evidence, n_synth = n_batch)
        if (m == candidate_set[, .N] + n_batch) {
            samples <- rbind(samples, warm_start)
        }

        # Divide the new from the old
        new_samples <- fsetdiff(samples, old_samples)
        old_samples <- unique(rbind(old_samples, new_samples))

        # Find probs of each candidate, find leading candidate (q_hat) and its
        # associated prob (p_hat)
        probs <- lik(
            params = params, query = new_samples,
            arf = arf, log = FALSE
        ) / prob_e
        prob_total <- prob_total + sum(probs)

        # Update p_hat and q_hat, but only if the highest prob from the
        # new samples is higher than what has been seen previously
        p_hat_new <- max(probs)
        if (p_hat_new > p_hat) {
            p_hat <- p_hat_new
            q_hat <- new_samples[which(probs == max(probs))]
        }

        # Calculate p_tick (residual probability mass)
        p_tick <- 1 - prob_total

        # Check stopping condition
        if (p_tick <= p_hat / (1 - err_tol)) {
            break
        }

        # Update count
        max_m <- log(fail_prob) / log(1 - (p_hat / (1 - err_tol)))
    }
    return(cbind(
        q_hat,
        data.table(
            p_hat = p_hat,
            fail_prob = fail_prob,
            err_tol = err_tol
        )
    ))
}
