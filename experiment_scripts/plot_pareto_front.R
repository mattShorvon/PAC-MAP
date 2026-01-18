library(data.table)
library(ggplot2)
library(patchwork)

# Define datasets and probabilities
datasets <- list(
    list(name = "baudio", prob = 4.426858642517541e-05, color = "black"),
    list(name = "ocr_letters", prob = 2.521704522188303e-05, color = "black"),
    list(name = "msweb", prob = 8.088554272768827e-06, color = "black")
)

m <- 100000
plots <- list()

for (i in seq_along(datasets)) {
    ds <- datasets[[i]]

    # Calculate epsilon and delta
    epsilon <- seq(0, 1 - ds$prob - 1e-10, length.out = 1000)
    delta <- (1 - (ds$prob / (1 - epsilon)))^m
    dt <- data.table(epsilon = epsilon, delta = delta)

    # Create plot
    plots[[i]] <- ggplot(dt, aes(x = epsilon, y = delta)) +
        geom_line(color = ds$color, linewidth = 2) +
        labs(
            x = if (i == 2) expression("Error Tolerance " * epsilon) else NULL,
            y = if (i == 1) expression("Failure Probability " * delta) else NULL,
            title = ds$name,
            subtitle = bquote(hat(p) == .(sprintf("%.2e", ds$prob)))
        ) +
        theme_bw() +
        theme(
            axis.text = element_text(size = 22),
            axis.title = element_text(size = 24),
            plot.title = element_text(size = 28, hjust = 0.5, face = "bold"),
            plot.subtitle = element_text(size = 24, hjust = 0.5)
        )
}

# Combine plots
combined <- plots[[1]] | plots[[2]] | plots[[3]]

combined <- combined +
    plot_annotation(
        theme = theme(
            plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
            plot.subtitle = element_text(size = 12, hjust = 0.5)
        )
    )

plot(combined)

# Save
ggsave("results/pareto_fronts.pdf", combined,
    width = 15, height = 5.5, dpi = 300
)
