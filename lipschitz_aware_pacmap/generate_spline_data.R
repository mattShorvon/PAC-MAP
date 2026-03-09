source("lipschitz_aware_pacmap/spline_basis_expansion.R")

output_dir <- "lipschitz_aware_pacmap/data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Initialise model
model <- new_cubic_spline_expansion(
    n_knots              = 3L,
    degree               = 3L,
    include_interactions = TRUE,
    weight_scale         = 1.0,
    sparsity             = 0.3,
    x_low                = 0.0,
    x_high               = 1.0,
    random_seed          = 42L
)

# Generate data (fits model internally)
result <- generate_data.CubicSplineBasisExpansion(
    model,
    n_samples       = 5000L,
    n_query_vars    = 2L,
    n_evidence_vars = 1L
)
model <- result$model
df <- result$df
metadata <- result$metadata

print(model)
cat("\nFirst 5 rows:\n")
print(head(df, 5))

cat("\nFeature summary (first 10 rows):\n")
print(head(feature_summary.CubicSplineBasisExpansion(model), 10))

# Save data and weights
write.csv(df, file.path(output_dir, "spline_data.csv"), row.names = FALSE)
saveRDS(model$weights, file.path(output_dir, "spline_weights.rds"))
cat("\nData saved to", file.path(output_dir, "spline_data.csv"), "\n")
cat("Weights saved to", file.path(output_dir, "spline_weights.rds"), "\n")

# Lipschitz constant (exact gradients, approximate maximiser)
X_mat <- as.matrix(df[, c(metadata$query_cols, metadata$evidence_cols)])
lip <- lipschitz_constant.CubicSplineBasisExpansion(model, n_grid_points = 10000L)

# Analytical upper bound (no grid search)
L_upper <- lipschitz_upper_bound.CubicSplineBasisExpansion(model)

# Plots
plot_marginal_probs.CubicSplineBasisExpansion(
    model, df, metadata,
    save_path = file.path(output_dir, "marginal_probs.png")
)

plot_gradient_norms.CubicSplineBasisExpansion(
    model, X_mat, metadata,
    save_path = file.path(output_dir, "gradient_norms.png")
)

plot_interaction_heatmap.CubicSplineBasisExpansion(
    model, df,
    var_i = "Q1",
    var_j = "E1",
    metadata = metadata,
    save_path = file.path(output_dir, "heatmap_Q1_E1.png")
)
