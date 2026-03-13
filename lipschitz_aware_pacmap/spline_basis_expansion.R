library(splines2)
library(glmnet)
library(ggplot2)
library(tidyr)
library(dplyr)


# ═════════════════════════════════════════════
# Constructor
# ═════════════════════════════════════════════

new_cubic_spline_expansion <- function(
    n_knots = 5L,
    degree = 3L,
    include_interactions = FALSE,
    weight_scale = 1.0,
    weight_vector = NULL,
    sparsity = 0.0,
    x_low = 0.0,
    x_high = 1.0,
    random_seed = 42L) {
    structure(
        list(
            # Hyperparameters
            n_knots = n_knots,
            degree = degree,
            include_interactions = include_interactions,
            weight_vector = weight_vector,
            weight_scale = weight_scale,
            sparsity = sparsity,
            x_low = x_low,
            x_high = x_high,
            random_seed = random_seed,

            # Set by fit()
            knots_per_var = NULL, # List of knot vectors, one per variable
            n_basis_per_var = NULL,
            n_vars = NULL,
            n_features = NULL,
            interaction_pairs = NULL,
            weights = NULL,
            variable_names = NULL,
            query_cols = NULL,
            evidence_cols = NULL,
            is_fitted = FALSE
        ),
        class = "CubicSplineBasisExpansion"
    )
}


# ═════════════════════════════════════════════
# Fit
# ═════════════════════════════════════════════

fit.CubicSplineBasisExpansion <- function(
    model,
    X,
    query_cols = NULL,
    evidence_cols = NULL) {
    stopifnot(is.matrix(X))
    set.seed(model$random_seed)

    n_samples <- nrow(X)
    n_vars <- ncol(X)

    query_cols <- query_cols %||% paste0("Q", seq_len(n_vars))
    evidence_cols <- evidence_cols %||% character(0)
    variable_names <- c(query_cols, evidence_cols)

    stopifnot(length(variable_names) == n_vars)

    n_interior <- max(model$n_knots - 2L, 0L)
    probs <- seq(0, 1, length.out = n_interior + 2L)[c(-1, -(n_interior + 2L))]

    knots_per_var <- lapply(seq_len(n_vars), function(i) {
        if (n_interior > 0) quantile(X[, i], probs = probs) else numeric(0)
    })

    # Store boundary knots from training data so they are reused consistently
    # in transform(), gradient, and Lipschitz functions
    boundary_knots_per_var <- lapply(seq_len(n_vars), function(i) {
        range(X[, i])
    })

    # Compute n_basis_per_var from a test evaluation using stored boundaries
    test_basis <- .eval_basis(
        X[1, 1],
        knots_per_var[[1]],
        model$degree,
        boundary_knots_per_var[[1]]
    )
    n_basis_per_var <- ncol(test_basis)

    interaction_pairs <- if (model$include_interactions) {
        combn(n_vars, 2, simplify = FALSE)
    } else {
        list()
    }

    n_main <- n_vars * n_basis_per_var
    n_interactions <- length(interaction_pairs) * n_basis_per_var^2
    n_features <- n_main + n_interactions

    if (is.null(model$weight_vector)) {
        weights <- abs(rnorm(n_features, mean = 0, sd = model$weight_scale))
    } else {
        weights <- abs(model$weight_vector)
    }

    if (model$sparsity > 0) {
        n_zero <- as.integer(model$sparsity * n_features)
        zero_idx <- sample.int(n_features, n_zero, replace = FALSE)
        weights[zero_idx] <- 0.0
    }

    model$knots_per_var <- knots_per_var
    model$boundary_knots_per_var <- boundary_knots_per_var
    model$n_basis_per_var <- n_basis_per_var
    model$n_vars <- n_vars
    model$n_features <- n_features
    model$interaction_pairs <- interaction_pairs
    model$weights <- weights
    model$variable_names <- variable_names
    model$query_cols <- query_cols
    model$evidence_cols <- evidence_cols
    model$is_fitted <- TRUE

    cat("CubicSplineBasisExpansion fitted:\n")
    cat("  n_vars              :", n_vars, "\n")
    cat("  n_basis_per_var     :", n_basis_per_var, "\n")
    cat("  n_main_effects      :", n_main, "\n")
    cat("  n_interaction_pairs :", length(interaction_pairs), "\n")
    cat("  n_interaction_feats :", n_interactions, "\n")
    cat("  n_features_total    :", n_features, "\n")

    model
}


# ═════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════

# Null coalescing operator
`%||%` <- function(a, b) if (!is.null(a)) a else b


.check_fitted <- function(model) {
    if (!model$is_fitted) {
        stop("Model is not fitted. Call fit() first.", call. = FALSE)
    }
}


# Evaluate B-spline basis for a single variable using stored knots.
# Returns matrix of shape (n_samples, n_basis).
.eval_basis <- function(x, knots, degree, boundary_knots) {
    bSpline(x,
        knots          = knots,
        degree         = degree,
        intercept      = TRUE,
        Boundary.knots = boundary_knots
    )
}


# Evaluate exact first derivative of B-spline basis.
# Returns matrix of shape (n_samples, n_basis).
.eval_basis_deriv <- function(x, knots, degree, boundary_knots) {
    dbs(x,
        knots          = knots,
        degree         = degree,
        intercept      = TRUE,
        derivs         = 1L,
        Boundary.knots = boundary_knots
    )
}


# Compute all K^2 pairwise products of two basis matrices row-wise.
# phi_i: (n x K), phi_j: (n x K) -> (n x K^2)
.interaction_features <- function(phi_i, phi_j) {
    n <- nrow(phi_i)
    K <- ncol(phi_i)
    # Expand dims and multiply: (n x K x 1) * (n x 1 x K) -> (n x K x K)
    # then reshape to (n x K^2)
    matrix(
        as.vector(phi_i[, rep(seq_len(K), each = K)] *
            phi_j[, rep(seq_len(K), times = K)]),
        nrow = n,
        ncol = K^2
    )
}


# Unpack weight vector into named list of main-effect and interaction blocks
.unpack_weights <- function(weights, n_vars, n_basis_per_var, interaction_pairs) {
    K <- n_basis_per_var
    offset <- 1L

    w_main <- vector("list", n_vars)
    for (i in seq_len(n_vars)) {
        w_main[[i]] <- weights[offset:(offset + K - 1L)]
        offset <- offset + K
    }

    w_inter <- list()
    for (pair in interaction_pairs) {
        key <- paste(pair, collapse = "_")
        w_inter[[key]] <- matrix(weights[offset:(offset + K^2 - 1L)], K, K)
        offset <- offset + K^2
    }

    list(w_main = w_main, w_inter = w_inter)
}


# ═════════════════════════════════════════════
# Transform
# ═════════════════════════════════════════════

transform.CubicSplineBasisExpansion <- function(model, X) {
    .check_fitted(model)
    stopifnot(is.matrix(X), ncol(X) == model$n_vars)

    bases <- lapply(seq_len(model$n_vars), function(i) {
        .eval_basis(
            X[, i],
            model$knots_per_var[[i]],
            model$degree,
            model$boundary_knots_per_var[[i]] # use stored boundaries
        )
    })

    main_effects <- do.call(cbind, bases)

    if (!model$include_interactions) {
        return(main_effects)
    }

    interaction_blocks <- lapply(model$interaction_pairs, function(pair) {
        .interaction_features(bases[[pair[1]]], bases[[pair[2]]])
    })

    cbind(main_effects, do.call(cbind, interaction_blocks))
}


# ═════════════════════════════════════════════
# Prediction
# ═════════════════════════════════════════════

logits.CubicSplineBasisExpansion <- function(model, X) {
    .check_fitted(model)
    as.vector(transform.CubicSplineBasisExpansion(model, X) %*% model$weights)
}


predict_proba.CubicSplineBasisExpansion <- function(model, X) {
    logits.CubicSplineBasisExpansion(model, X)
}


# ═════════════════════════════════════════════
# Exact gradients
# ═════════════════════════════════════════════

gradient_logit.CubicSplineBasisExpansion <- function(model, X) {
    .check_fitted(model)
    stopifnot(is.matrix(X), ncol(X) == model$n_vars)

    n_vars <- model$n_vars
    K <- model$n_basis_per_var
    n <- nrow(X)

    bases <- lapply(seq_len(n_vars), function(i) {
        .eval_basis(
            X[, i],
            model$knots_per_var[[i]],
            model$degree,
            model$boundary_knots_per_var[[i]] # use stored boundaries
        )
    })

    dbases <- lapply(seq_len(n_vars), function(i) {
        .eval_basis_deriv(
            X[, i],
            model$knots_per_var[[i]],
            model$degree,
            model$boundary_knots_per_var[[i]] # use stored boundaries
        )
    })

    unpacked <- .unpack_weights(
        model$weights, n_vars, K, model$interaction_pairs
    )
    w_main <- unpacked$w_main
    w_inter <- unpacked$w_inter

    grad <- matrix(0.0, nrow = n, ncol = n_vars)

    # Main effect contributions: dphi_k(x_i) * w_ik
    for (i in seq_len(n_vars)) {
        grad[, i] <- as.vector(dbases[[i]] %*% w_main[[i]])
    }

    # Interaction contributions
    for (pair in model$interaction_pairs) {
        i <- pair[1]
        j <- pair[2]
        key <- paste(pair, collapse = "_")
        W <- w_inter[[key]] # K x K, rows index i, cols index j

        # d/dx_i [ phi_k(x_i) * phi_l(x_j) * W_kl ]
        # = sum_{k,l} W_kl * dphi_k(x_i) * phi_l(x_j)
        # = rowSums( (dbases[[i]] %*% W) * bases[[j]] )
        grad[, i] <- grad[, i] +
            rowSums((dbases[[i]] %*% W) * bases[[j]])

        # d/dx_j [ phi_k(x_i) * phi_l(x_j) * W_kl ]
        # = sum_{k,l} W_kl * phi_k(x_i) * dphi_l(x_j)
        # = rowSums( bases[[i]] %*% W * dbases[[j]] )
        # = rowSums( (bases[[i]] %*% W) * dbases[[j]] )
        grad[, j] <- grad[, j] +
            rowSums((bases[[i]] %*% W) * dbases[[j]])
    }

    grad
}


gradient_prob.CubicSplineBasisExpansion <- function(model, X) {
    p <- predict_proba.CubicSplineBasisExpansion(model, X) # (n,)
    dlogit <- gradient_logit.CubicSplineBasisExpansion(model, X) # (n, p)
    # Chain rule: dP/dx_i = P(1-P) * df/dx_i
    as.vector(p * (1 - p)) * dlogit
}


gradient_norm.CubicSplineBasisExpansion <- function(model, X) {
    grad <- gradient_prob.CubicSplineBasisExpansion(model, X)
    sqrt(rowSums(grad^2))
}


# ═════════════════════════════════════════════
# Lipschitz constant
# ═════════════════════════════════════════════

lipschitz_constant.CubicSplineBasisExpansion <- function(
    model,
    n_grid_points = 10000L,
    X_samples = NULL) {
    .check_fitted(model)
    set.seed(model$random_seed)

    X_eval <- if (!is.null(X_samples)) {
        X_samples
    } else {
        matrix(
            runif(n_grid_points * model$n_vars, model$x_low, model$x_high),
            nrow = n_grid_points,
            ncol = model$n_vars
        )
    }

    norms <- gradient_norm.CubicSplineBasisExpansion(model, X_eval)
    idx <- which.max(norms)
    L <- norms[idx]
    X_max <- X_eval[idx, ]

    cat(sprintf("Lipschitz constant (exact gradients): L = %.6f\n", L))
    cat(sprintf(
        "  Achieved at X = [%s]\n",
        paste(round(X_max, 4), collapse = ", ")
    ))
    cat(sprintf(
        "  P(X_max) = %.4f\n",
        predict_proba.CubicSplineBasisExpansion(model, matrix(X_max, nrow = 1))
    ))

    list(L = L, X_max = X_max)
}


lipschitz_upper_bound.CubicSplineBasisExpansion <- function(model) {
    .check_fitted(model)

    max_dphi_per_var <- lapply(seq_len(model$n_vars), function(i) {
        x_fine <- seq(
            model$boundary_knots_per_var[[i]][1],
            model$boundary_knots_per_var[[i]][2],
            length.out = 10000L
        )
        dphi <- .eval_basis_deriv(
            x_fine,
            model$knots_per_var[[i]],
            model$degree,
            model$boundary_knots_per_var[[i]] # use stored boundaries
        )
        apply(abs(dphi), 2, max)
    })

    # Main effect contributions to C_phi
    main_contribs <- unlist(max_dphi_per_var)

    # Interaction contributions: B-splines are partition of unity so phi_l <= 1,
    # hence d/dx_i [phi_k(x_i) * phi_l(x_j)] <= max|phi_k'(x_i)| * 1
    interaction_contribs <- unlist(lapply(model$interaction_pairs, function(pair) {
        c(max_dphi_per_var[[pair[1]]], max_dphi_per_var[[pair[2]]])
    }))

    all_contribs <- c(main_contribs, interaction_contribs)
    C_phi <- sqrt(sum(all_contribs^2))
    w_norm <- sqrt(sum(model$weights^2))
    L_upper <- 0.25 * w_norm * C_phi

    cat(sprintf("Analytical Lipschitz upper bound: L <= %.6f\n", L_upper))
    cat(sprintf("  ||w||_2 = %.4f\n", w_norm))
    cat(sprintf("  C_phi   = %.4f\n", C_phi))

    L_upper
}


# ═════════════════════════════════════════════
# Data generation
# ═════════════════════════════════════════════

generate_data.CubicSplineBasisExpansion <- function(
    model,
    n_samples,
    n_query_vars,
    n_evidence_vars) {
    set.seed(model$random_seed)

    n_vars <- n_query_vars + n_evidence_vars
    query_cols <- paste0("Q", seq_len(n_query_vars))
    evidence_cols <- paste0("E", seq_len(n_evidence_vars))

    X <- matrix(
        runif(n_samples * n_vars, model$x_low, model$x_high),
        nrow = n_samples,
        ncol = n_vars
    )

    if (!model$is_fitted) {
        model <- fit.CubicSplineBasisExpansion(
            model, X,
            query_cols = query_cols,
            evidence_cols = evidence_cols
        )
    }

    probs <- predict_proba.CubicSplineBasisExpansion(model, X)

    df <- as.data.frame(X)
    colnames(df) <- c(query_cols, evidence_cols)
    df$prob <- probs

    metadata <- list(
        n_samples            = n_samples,
        n_query_vars         = n_query_vars,
        n_evidence_vars      = n_evidence_vars,
        n_knots              = model$n_knots,
        degree               = model$degree,
        include_interactions = model$include_interactions,
        weight_scale         = model$weight_scale,
        sparsity             = model$sparsity,
        x_low                = model$x_low,
        x_high               = model$x_high,
        random_seed          = model$random_seed,
        n_features           = model$n_features,
        n_basis_per_var      = model$n_basis_per_var,
        query_cols           = query_cols,
        evidence_cols        = evidence_cols,
        interaction_pairs    = model$interaction_pairs
    )

    cat(sprintf("\nGenerated %d samples\n", n_samples))
    cat(sprintf("  Mean P(X): %.4f\n", mean(probs)))
    cat(sprintf("  Min  P(X): %.4f\n", min(probs)))
    cat(sprintf("  Max  P(X): %.4f\n", max(probs)))

    list(model = model, df = df, metadata = metadata)
}


# ═════════════════════════════════════════════
# Feature summary
# ═════════════════════════════════════════════

feature_summary.CubicSplineBasisExpansion <- function(model) {
    .check_fitted(model)
    K <- model$n_basis_per_var

    main_rows <- do.call(rbind, lapply(seq_len(model$n_vars), function(i) {
        data.frame(
            type = "main_effect",
            var_1 = model$variable_names[i],
            var_2 = NA_character_,
            basis_1 = seq_len(K) - 1L,
            basis_2 = NA_integer_,
            name = sprintf("phi_%d(%s)", seq_len(K) - 1L, model$variable_names[i]),
            stringsAsFactors = FALSE
        )
    }))

    if (!model$include_interactions || length(model$interaction_pairs) == 0) {
        return(main_rows)
    }

    inter_rows <- do.call(rbind, lapply(model$interaction_pairs, function(pair) {
        vi <- model$variable_names[pair[1]]
        vj <- model$variable_names[pair[2]]
        expand.grid(basis_1 = seq_len(K) - 1L, basis_2 = seq_len(K) - 1L) |>
            transform(
                type  = "interaction",
                var_1 = vi,
                var_2 = vj,
                name  = sprintf("phi_%d(%s) * phi_%d(%s)", basis_1, vi, basis_2, vj)
            )
    }))

    rbind(main_rows, inter_rows)
}


# ═════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════

plot_marginal_probs.CubicSplineBasisExpansion <- function(
    model,
    df,
    metadata,
    save_path = NULL) {
    all_vars <- c(metadata$query_cols, metadata$evidence_cols)

    plot_df <- df |>
        select(all_of(c(all_vars, "prob"))) |>
        pivot_longer(
            cols = all_of(all_vars),
            names_to = "variable",
            values_to = "x"
        )

    p <- ggplot(plot_df, aes(x = x, y = prob)) +
        geom_point(alpha = 0.2, size = 0.8, colour = "steelblue") +
        facet_wrap(~variable, scales = "free_x") +
        ylim(-0.05, 1.05) +
        labs(
            title = "Marginal Probabilities — Spline Basis Expansion",
            x     = "Variable value",
            y     = "P(X)"
        ) +
        theme_bw()

    if (!is.null(save_path)) {
        ggsave(save_path, p, width = 10, height = 6, dpi = 300)
        cat("Plot saved to", save_path, "\n")
    } else {
        print(p)
    }
}


plot_gradient_norms.CubicSplineBasisExpansion <- function(
    model,
    X,
    metadata,
    save_path = NULL) {
    all_vars <- c(metadata$query_cols, metadata$evidence_cols)
    norms <- gradient_norm.CubicSplineBasisExpansion(model, X)

    plot_df <- as.data.frame(X)
    colnames(plot_df) <- all_vars
    plot_df$norm <- norms

    plot_df_long <- plot_df |>
        pivot_longer(
            cols = all_of(all_vars),
            names_to = "variable",
            values_to = "x"
        )

    p <- ggplot(plot_df_long, aes(x = x, y = norm)) +
        geom_point(alpha = 0.2, size = 0.8, colour = "darkorange") +
        facet_wrap(~variable, scales = "free_x") +
        labs(
            title = "Gradient Norms — Spline Basis Expansion",
            x     = "Variable value",
            y     = expression(group("||", nabla * P(X), "||")[2])
        ) +
        theme_bw()

    if (!is.null(save_path)) {
        ggsave(save_path, p, width = 10, height = 6, dpi = 300)
        cat("Plot saved to", save_path, "\n")
    } else {
        print(p)
    }
}


plot_interaction_heatmap.CubicSplineBasisExpansion <- function(
    model,
    df,
    var_i,
    var_j,
    metadata,
    n_grid = 50L,
    save_path = NULL) {
    all_vars <- c(metadata$query_cols, metadata$evidence_cols)
    col_means <- colMeans(df[, all_vars])

    xi_seq <- seq(model$x_low, model$x_high, length.out = n_grid)
    xj_seq <- seq(model$x_low, model$x_high, length.out = n_grid)
    grid <- expand.grid(xi = xi_seq, xj = xj_seq)

    X_grid <- matrix(
        rep(col_means, each = nrow(grid)),
        nrow = nrow(grid),
        ncol = length(all_vars),
        dimnames = list(NULL, all_vars)
    )
    X_grid[, var_i] <- grid$xi
    X_grid[, var_j] <- grid$xj

    grid$prob <- predict_proba.CubicSplineBasisExpansion(model, X_grid)

    p <- ggplot(grid, aes(x = xi, y = xj, fill = prob)) +
        geom_tile() +
        scale_fill_distiller(
            palette = "RdYlBu", limits = c(0, 1),
            name = "P(X)"
        ) +
        labs(
            title = sprintf("P(X) | %s, %s — others at mean", var_i, var_j),
            x     = var_i,
            y     = var_j
        ) +
        theme_bw()

    if (!is.null(save_path)) {
        ggsave(save_path, p, width = 6, height = 5, dpi = 300)
        cat("Plot saved to", save_path, "\n")
    } else {
        print(p)
    }
}


# ═════════════════════════════════════════════
# print method
# ═════════════════════════════════════════════

print.CubicSplineBasisExpansion <- function(model, ...) {
    status <- if (model$is_fitted) "fitted" else "not fitted"
    cat(sprintf(
        "CubicSplineBasisExpansion(n_knots=%d, degree=%d, interactions=%s, status=%s)\n",
        model$n_knots, model$degree,
        model$include_interactions, status
    ))
}
