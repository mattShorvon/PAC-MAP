library(data.table)
library(ggplot2)
library(patchwork)

# Load tracking data
dt <- fread("results/pac_map_tracking.csv")

# Plot p_hat
g1 <- ggplot(dt, aes(x = m, y = p_hat)) +
    geom_line(color = "steelblue", linewidth = 1.2) +
    labs(
        x = "Samples (m)",
        y = expression(hat(p)),
        title = "Best Probability Estimate"
    ) +
    theme_bw()

# Plot p_tick
g2 <- ggplot(dt, aes(x = m, y = p_tick)) +
    geom_line(color = "red", linewidth = 1.2) +
    labs(
        x = "Samples (m)",
        y = expression(tilde(p)),
        title = "Remaining Probability Mass"
    ) +
    theme_bw()

# Plot M
g3 <- ggplot(dt, aes(x = m, y = M)) +
    geom_line(color = "darkgreen", linewidth = 1.2) +
    scale_y_log10() +
    labs(
        x = "Samples (m)",
        y = "M (Required Samples)",
        title = "Sample Requirement"
    ) +
    theme_bw()

# Combine plots
combined <- g1 | g2 | g3

print(combined)
ggsave("results/pac_map_convergence.pdf", combined,
    width = 15, height = 5, dpi = 300
)
