library(data.table)
library(ggplot2)
library(ggsci)

# Load data
df <- fread("amp+pacmap_h_results.csv")
df[, Query_Index := .GRP, by = .(Dataset, Query)]
df[, Date := NULL]
df[, Query := NULL]
colnames(df)[2] <- "Amp_Prob"
colnames(df)[3] <- "Pac_Map_Prob"

# Prep data
df[, lambda := .SD[, log(Pac_Map_Prob / Amp_Prob), by = .(Query_Index, Dataset)]]
df[, lambda := log(.SD[Method == "PAC_MAP", Prob] / Prob), by = .(Query_Index, Dataset)]
df[, mu := mean(lambda), by = .(Dataset)]
df[, se := sd(lambda) / sqrt(10), by = .(Dataset)]

# Reduce
tmp <- unique(df[Method != "PAC_MAP", .(Dataset, Method, mu, se)])
tmp[Method == "Max Product", Method := "MP"]
tmp[Method == "ArgMax Product", Method := "AMP"]
tmp[Method == "Hybrid Belief-Propagation", Method := "HBP"]
tmp[Method == "Max Search", Method := "MS"]
to_drop <- df[Prob == 1, unique(Dataset)]
tmp <- tmp[!Dataset %in% to_drop]

# Plot
pd <- position_dodge(width = 0.9)
g <- ggplot(df, aes(Dataset, mu)) +
    geom_bar(
        color = "black", position = pd, stat = "identity",
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
ggsave("results/amp_pacmaph_comparison.pdf", g, width = 8, height = 12, dpi = 300)
