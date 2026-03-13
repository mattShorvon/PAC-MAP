# ── 1. Source your script ──────────────────────────────────────────────
source("lipschitz_aware_pacmap/spline_basis_expansion.R")

# ── 2. Create a tiny, fully controlled example ────────────────────────
# 2 variables, 1 interior knot, degree 1 (linear) -- easy to compute by hand
model <- new_cubic_spline_expansion(
    n_knots      = 3L, # 1 interior knot -> n_basis = 1 + 1 + 1 = 3 per var
    degree       = 1L, # linear splines: easy to verify by hand
    weight_scale = 1.0,
    random_seed  = 42L
)

# Fix X so you know exactly what the inputs are
X <- matrix(
    c(
        0.25, 0.75, # sample 1: var1=0.25, var2=0.75
        0.75, 0.25
    ), # sample 2: var1=0.50, var2=0.50
    nrow = 2, ncol = 2
)

model <- fit.CubicSplineBasisExpansion(model, X)

# ── 3. Inspect knots ───────────────────────────────────────────────────
cat("\n--- Knots per variable ---\n")
for (i in seq_len(model$n_vars)) {
    cat(sprintf(
        "  Var %d interior knots : %s\n", i,
        paste(round(model$knots_per_var[[i]], 4), collapse = ", ")
    ))
    cat(sprintf(
        "  Var %d boundary knots : %s\n", i,
        paste(round(model$boundary_knots_per_var[[i]], 4), collapse = ", ")
    ))
}

# ── 4. Override weights with known fixed values ────────────────────────
# This is the key step for hand-verification: you choose the weights
model$weights <- c(
    1.0, 0.0, 1.0, # weights for var 1 basis functions
    0.0, 1.0, 0.0
) # weights for var 2 basis functions

cat("\n--- Weights ---\n")
print(model$weights)

# ── 5. Inspect the basis matrix (Phi) for your test points ────────────
Phi <- transform.CubicSplineBasisExpansion(model, X)
cat("\n--- Basis matrix Phi (n_samples x n_features) ---\n")
print(round(Phi, 6))

# Verify rows sum to 1 per variable block (partition of unity)
K <- model$n_basis_per_var
cat("\n--- Row sums per variable block (should all be 1.0) ---\n")
for (i in seq_len(model$n_vars)) {
    cols <- ((i - 1) * K + 1):(i * K)
    cat(sprintf(
        "  Var %d: %s\n", i,
        paste(round(rowSums(Phi[, cols, drop = FALSE]), 6), collapse = ", ")
    ))
}

# ── 6. Compute output and compare to hand calculation ─────────────────
# output = Phi %*% w  (your hand calculation should match this)
output <- predict_proba.CubicSplineBasisExpansion(model, X)
cat("\n--- Output probabilities ---\n")
print(round(output, 6))

# ── 7. Verify manually: Phi %*% weights step by step ──────────────────
cat("\n--- Manual dot product check ---\n")
for (s in seq_len(nrow(X))) {
    manual <- sum(Phi[s, ] * model$weights)
    cat(sprintf(
        "  Sample %d: Phi[%d,] . w = %.6f  |  predict = %.6f  |  match = %s\n",
        s, s, manual, output[s], isTRUE(all.equal(manual, output[s]))
    ))
}


for (i in seq_len(model$n_vars)) {
    cat(sprintf(
        "\n--- Piecewise polynomial coefficients: variable %d (%s) ---\n",
        i, model$variable_names[i]
    ))

    breakpoints <- c(
        model$boundary_knots_per_var[[i]][1],
        model$knots_per_var[[i]],
        model$boundary_knots_per_var[[i]][2]
    )

    cat(sprintf(
        "  Breakpoints: %s\n",
        paste(round(breakpoints, 4), collapse = " | ")
    ))

    for (k in seq_len(model$n_basis_per_var)) {
        cat(sprintf("\n  phi_%d(x%d):\n", k - 1, i))

        for (j in seq_len(length(breakpoints) - 1)) {
            x_int <- seq(breakpoints[j], breakpoints[j + 1], length.out = 100)

            phi_int <- bSpline(
                x              = x_int,
                knots          = model$knots_per_var[[i]],
                degree         = model$degree,
                intercept      = TRUE,
                Boundary.knots = model$boundary_knots_per_var[[i]]
            )

            y_int <- phi_int[, k]
            cf <- coef(lm(y_int ~ poly(x_int, degree = model$degree, raw = TRUE)))

            term_strs <- mapply(function(coef, power) {
                if (power == 0) {
                    sprintf("%.6f", coef)
                } else {
                    sprintf("%.6f * x^%d", coef, power)
                }
            }, cf, seq(0, model$degree))

            cat(sprintf(
                "    x in [%.4f, %.4f]: %s\n",
                breakpoints[j], breakpoints[j + 1],
                paste(term_strs, collapse = " + ")
            ))
        }
    }
}
