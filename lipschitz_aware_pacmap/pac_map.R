library(data.table)

add_missing_zeros <- function(params_cat) {
    # Get all unique values
    all_f_idx <- unique(params_cat$f_idx)
    all_variables <- unique(params_cat$variable)
    all_vals <- c("0", "1")

    # Create all combinations
    all_combinations <- CJ(
        f_idx = all_f_idx,
        variable = all_variables,
        val = all_vals
    )

    # Find which combinations are missing
    setkey(params_cat, f_idx, variable, val)
    setkey(all_combinations, f_idx, variable, val)
    missing_combinations <- all_combinations[!params_cat]

    # Add zero probability rows
    if (nrow(missing_combinations) > 0) {
        missing_combinations[, prob := 0]
        missing_combinations[, NA_share := 0]

        # Combine with existing data
        complete_cat <- rbind(params_cat, missing_combinations)
    } else {
        complete_cat <- params_cat
    }

    return(complete_cat)
}

map_subcircuits <- function(params, query, evidence, mode = "forest") {
    # Add rows with variable & val combos that have 0 probability back into the
    # table, so that impossible assignments get zeroed out when multiplying
    # over leaves to get joint probabilities
    params$cat <- add_missing_zeros(params$cat)

    # Merge with params$forest to get tree information
    params$cat <- merge(params$cat,
        params$forest[, .(f_idx, tree, cvg)],
        by = "f_idx", sort = FALSE, allow.cartesian = TRUE
    )
    setkey(params$cat, tree, f_idx, variable)

    # Set all leaves to evidence values
    d_evi <- ncol(evidence)
    evi <- melt(evidence, measure.vars = 1:d_evi, value.name = "val")
    e_params <- merge(params$cat, evi, by = c("variable", "val"))

    # Looping over each tree
    out <- data.table()
    for (b in params$cat[, unique(tree)]) {
        # Starting at the bottom of the circuit, find the map state in each
        # input distro of the circuit
        q_params_b <- params$cat[variable %in% query & tree == b,
            .SD[which.max(prob)],
            by = .(f_idx, variable)
        ]

        # Moving up to product nodes over inputs, multiply the probs of
        # assignments to each variable (both query and evidence vars) to find
        # the joint prob of the product node/leaf map state
        prodnode_probs <- rbind(q_params_b, e_params[tree == b, ])
        prodnode_probs[, joint_prob := prod(prob), by = f_idx]

        # Many assignments will be impossible because the constraints of the
        # leaves they are in do not permit the evidence. These will end up with
        # prob == 0, so we take them out.
        prodnode_probs <- prodnode_probs[joint_prob > 0]
        if (nrow(prodnode_probs) == 0) {
            # Come up with a random assignment if none of the leaves can permit
            # the evidence
            res <- data.table(
                f_idx = 1,
                variable = query,
                val = sample(c("0", "1"),
                    length(query),
                    replace = TRUE
                ),
                prob = 0,
                NA_share = 0,
                tree = b,
                cvg = 0,
                joint_prob = 0,
                joint_prob_wt = 0
            )
            out <- rbind(out, res)
            next
        }

        # At the sum node, weight each joint prob of the leaf's map state by
        # the leaf's coverage and find the map state with the highest weighted
        # joint prob
        prodnode_probs[, joint_prob_wt := joint_prob * cvg]
        if (mode == "decision_tree") {
            # In this mode, if more than one assignment are joint highest, we
            # just want to return one of them
            max_prob <- prodnode_probs[which.max(joint_prob_wt)]$joint_prob_wt
            res <- prodnode_probs[joint_prob_wt == max_prob]
            if (length(res[, unique(f_idx)]) > 1) {
                id <- res[, unique(f_idx)][1]
                out <- rbind(out, res[f_idx == id, ])
            } else {
                out <- rbind(out, res)
            }
        } else {
            # Otherwise, in forest mode (if we are using this function to find
            # warm start candidates) then all joint highest prob assignments
            # might be good candidates, so we'll take all of them
            max_prob <- prodnode_probs[which.max(joint_prob_wt)]$joint_prob_wt
            out <- rbind(out, prodnode_probs[joint_prob_wt == max_prob])
        }
    }
    out <- dcast(out, f_idx ~ variable, value.var = "val")[, -"f_idx"]
    out <- unique(out)[, ..query]

    out[, (query) := lapply(.SD, function(x) factor(x, levels = c("0", "1"))),
        .SDcols = query
    ]
    return(out)
}

################################################################################

pac_mmap_circuit <- function(params,
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

################################################################################


### Budget PAC-MAP, aka Inverse PAC-MAP

budget_pacmap <- function(params, arf, query, evidence, m) {
    # Draw samples
    samples <- forge(params = params, evidence = evidence, n_synth = m)

    # Compute likelihoods
    probs <- lik(
        params = params, query = samples, evidence = evidence,
        arf = arf, log = FALSE
    )

    # Estimated MAP
    solution <- samples[which.max(probs)]
    solution[, p_hat := max(probs)]

    # Calculate p_tick (residual probability mass)
    p_tick <- 1 - sum(probs)

    if (p_tick <= p_hat) {
        # Solution is exact
        pac_params <- data.table("epsilon" = 0, "delta" = 0)
    } else {
        # Compute Pareto frontier
        epsilon <- seq(0, 1 - p_hat - 1e-6, length.out = 200)
        delta <- (1 - p_hat / (1 - epsilon))^m
        pac_params <- data.table(epsilon, delta)
    }

    # Export
    return(list("solution" = solution, "pac_params" = pac_params))
}


################################################################################

# setwd('~/Downloads/MAP_For_ARFs/arf_latest')

# source("adversarial_rf.R")
# source("forde.R")
# source("lik.R")
# source("utils.R")
# source("forge.R")


# Simulate data
# n <- 1e4
# d <- 5
# dat <- as.data.table(matrix(
#     rbinom(n * (d + 1), size = 1, prob = 0.5),
#     nrow = n
# ))
# # colnames(dat) <- c(paste0("q", 1:d), "e1")

# # Train model, compile circuit
# arf <- adversarial_rf(dat, num_trees = 200)
# psi <- forde(arf, dat, finite_bounds = "global", epsilon = 0.01)

# # Query, evidence
# query <- paste0("q", 1:d)
# evidence <- dat[1, "e1"]

# pac_mmap_circuit(psi, arf, query, evidence)

# library(ggplot2)
# budget_res <- budget_pacmap(psi, arf, query, evidence, m = 200)
# g <- ggplot(budget_res$pac_params, aes(epsilon, delta)) +
#   geom_line() +
#   theme_bw()


################################################################################
