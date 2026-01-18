library(data.table)
library(ggplot2)
library(ggsci)

# Load data
df <- fread("amp+pacmap_h_results.csv")
df <- df[Date == "07-01-2026 16:07:09", ]
df[, Date := NULL]
df[, Query := NULL]
colnames(df)[2] <- "Amp_Prob"
colnames(df)[3] <- "Pac_Map_Prob"

# Prep data
df[, lambda := log(Pac_Map_Prob / Amp_Prob)]
df[, mu := mean(lambda), by = .(Dataset)]
df[, se := sd(lambda) / sqrt(10), by = .(Dataset)]

# Reduce
tmp <- unique(df[, .(Dataset, mu, se)])
setorder(tmp, -mu)

# Convert Dataset to ordered factor to preserve sort order in plot
tmp[, Dataset := factor(Dataset, levels = Dataset)]

# Plot
pd <- position_dodge(width = 0.9)
g <- ggplot(tmp, aes(Dataset, mu)) +
    geom_bar(
        color = "grey", position = pd, stat = "identity",
        alpha = 0.75
    ) +
    geom_errorbar(aes(ymin = mu - se, ymax = mu + se),
        position = pd,
        width = 0.4, linewidth = 0.75
    ) +
    scale_fill_npg() +
    labs(y = "Log likelihood ratio") +
    theme_bw() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16), # ← x-axis title
        axis.title.y = element_text(size = 16)
    )

plot(g)
ggsave(
    "results/amp_pacmaph_comparison.pdf",
    g,
    width = 8, height = 10, dpi = 300
)

# Trying out plotting percentage increase instead of log likelihood ratio

# Load data
df <- fread("amp+pacmap_h_results.csv")
df <- df[Date == "07-01-2026 16:07:09", ]
df[, Date := NULL]
df[, Query := NULL]
colnames(df)[2] <- "Amp_Prob"
colnames(df)[3] <- "Pac_Map_Prob"

# Prep data
df[, lambda := ((Pac_Map_Prob - Amp_Prob) / Amp_Prob) * 100]
df[, mu := mean(lambda), by = .(Dataset)]
df[, se := sd(lambda) / sqrt(10), by = .(Dataset)]

# Reduce
tmp <- unique(df[, .(Dataset, mu, se)])
setorder(tmp, -mu)

# Convert Dataset to ordered factor to preserve sort order in plot
tmp[, Dataset := factor(Dataset, levels = Dataset)]

# Plot
pd <- position_dodge(width = 0.9)
g <- ggplot(tmp, aes(Dataset, mu)) +
    geom_bar(
        color = "grey", position = pd, stat = "identity",
        alpha = 0.75
    ) +
    geom_errorbar(aes(ymin = mu - se, ymax = mu + se),
        position = pd,
        width = 0.4, linewidth = 0.75
    ) +
    scale_fill_npg() +
    labs(y = "Percentage Increase (%)") +
    theme_bw() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16), # ← x-axis title
        axis.title.y = element_text(size = 16)
    )

plot(g)
ggsave(
    "results/amp_pacmaph_comparison_pct_increase.pdf",
    g,
    width = 8, height = 6, dpi = 300
)

# Using log2
df <- fread("amp+pacmap_h_results.csv")
df <- df[Date == "07-01-2026 16:07:09", ]
df[, Date := NULL]
df[, Query := NULL]
colnames(df)[2] <- "Amp_Prob"
colnames(df)[3] <- "Pac_Map_Prob"

# Prep data
df[, lambda := log2(Pac_Map_Prob / Amp_Prob)]
df[, mu := mean(lambda), by = .(Dataset)]
df[, se := sd(lambda) / sqrt(10), by = .(Dataset)]

# Reduce
tmp <- unique(df[, .(Dataset, mu, se)])
setorder(tmp, -mu)

# Convert Dataset to ordered factor to preserve sort order in plot
tmp[, Dataset := factor(Dataset, levels = Dataset)]

# Plot
pd <- position_dodge(width = 0.9)
g <- ggplot(tmp, aes(Dataset, mu)) +
    geom_bar(
        color = "grey", position = pd, stat = "identity",
        alpha = 0.75
    ) +
    geom_errorbar(aes(ymin = mu - se, ymax = mu + se),
        position = pd,
        width = 0.4, linewidth = 0.75
    ) +
    scale_fill_npg() +
    labs(y = "Log likelihood ratio") +
    theme_bw() +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 16), # ← x-axis title
        axis.title.y = element_text(size = 16)
    )

plot(g)
ggsave(
    "results/amp_pacmaph_comparison.pdf",
    g,
    width = 8, height = 10, dpi = 300
)
