##### Simulate convergence behavior #####

# Load libraries
library(data.table)
library(ggplot2)
library(ggsci)
library(cowplot)

# Simulate PMF with fixed MAP
set.seed(123)
n_states <- 64
p <- rexp(n_states)
p <- p / sum(p)
p_star <- p[which.max(p)]

# Monitor hat_p, check_p
hats <- double()
checks <- double()
fails <- double()

# Run PAC-MAP
epsilon <- 0.01
delta <- 0.01
fail_prob <- 1
numerator <- (1 - epsilon) * log(1 / delta)
m <- 0L
M <- Inf
s <- double()
hat_p <- 1 / n_states
check_p <- 0
while (m < M) {
  m <- m + 1L
  p_new <- sample(p, 1)
  s <- unique(c(s, p_new))
  if (p_new > hat_p) {
    hat_p <- p_new
  }
  check_p <- 1 - sum(s)
  hats <- c(hats, hat_p)
  checks <- c(checks, check_p)
  new_fail <- (1 - hat_p / (1 - epsilon))^m
  fails <- c(fails, new_fail)
  M <- numerator / hat_p
}

# Plot results
df <- data.table(
  m = 1:length(hats), Upper = checks, Lower = hats, Fail = fails
)
df1 <- melt(df, id.vars = 'm', measure.vars = c('Upper', 'Lower'), 
            variable.name = 'Bound', value.name = 'Probability')
t1 <- data.table('Probability' = p_star + 0.02, 
                 'txt' = 'p')
g1 <- ggplot(df1, aes(m, Probability, color = Bound)) + 
  geom_line(linewidth = 1) +
  geom_hline(yintercept = p_star, linetype = 'dashed') + 
  scale_color_d3() +
  annotate('text', x = 0, y = p_star, 
           label = paste0("italic(p)^'*' ==", round(p_star, 3)), parse = TRUE, 
           hjust = -0.25, vjust = -0.75, size = 5) +
  labs(x = expression(paste("Number of samples ", italic(m)))) +
  theme_bw() +
  theme(legend.position = c(0.95, 0.95),
        legend.justification = c(1, 1),
        axis.title = element_text(size = 14),      
        axis.text = element_text(size = 12),       
        legend.text = element_text(size = 12),     
        legend.title = element_text(size = 14))

g2 <-ggplot(df, aes(m, Fail)) + 
  geom_line(linewidth = 1) + 
  geom_hline(yintercept = 0.01, linetype = 'dashed') + 
  annotate('text', x = 0, y = delta,
           label = paste0('delta ==', delta), parse = TRUE,
           hjust = -0.25, vjust = -0.75, size = 5) +
  labs(x = expression(paste("Number of samples ", italic(m))),
       y = 'Maximum Failure Probability') +
  theme_bw() + 
  theme(axis.title = element_text(size = 14),      
        axis.text = element_text(size = 12),       
        legend.text = element_text(size = 12),    
        legend.title = element_text(size = 14))

# Put it all together
g <- plot_grid(g1, g2, labels = c("A", "B"))
ggsave('~/downloads/convergence.pdf', height = 5, width = 13)