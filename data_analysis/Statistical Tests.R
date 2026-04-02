# Statistical Tests.R

library(tidyverse)
library(rstatix)
library(irr)

# Helper: compute effect size r = |Z| / sqrt(N) for Wilcoxon / Mann-Whitney
# Z is approximated from the two-tailed p-value: Z = qnorm(p/2)
effect_size_r <- function(p_value, n_total) {
  Z <- abs(qnorm(p_value / 2))
  return(Z / sqrt(n_total))
}

# Mapping configuration
region_map <- data.frame(
  region_code = c("a", "b", "c", "d", "e", "f", "g"),
  continent = c("North America", "South America", "Europe", "Africa", "Middle East", "Asia", "Australia")
)

load_cluster <- function(filename) {
  if (!file.exists(filename)) return(NULL)
  read.csv(filename) %>%
    # We keep the NAs here to handle them specifically in the tests
    filter(Securitization_Text != "not applicable" | Securitization_Visual != "not applicable")
}

# RQ1: Global South (GS) vs. Global North (GN)
cat("RQ1: EXTENT OF SECURITIZATION (GLOBAL SOUTH VS. NORTH)\n")

# Use 'cluster_1' files for joint analysis of extent
gs_data <- load_cluster("cluster_1_a_gs_hy_labeled.csv") %>% mutate(Group = "Global South")
gn_data <- load_cluster("cluster_1_a_gn_hy_labeled.csv") %>% mutate(Group = "Global North")
rq1_data <- bind_rows(gs_data, gn_data)

if(!is.null(rq1_data) && nrow(rq1_data) > 0) {
  cat("--- Mean Intensity (Handling NAs) ---\n")
  rq1_stats <- rq1_data %>% 
    group_by(Group) %>% 
    summarise(N = n(),
              Text_M = mean(Score_Text, na.rm = TRUE), 
              Vis_M  = mean(Score_Visual, na.rm = TRUE),
              .groups = "drop")
  print(as.data.frame(rq1_stats))
  
  cat("\n--- Mann-Whitney U Test: Text ---\n")
  mw_text <- rstatix::wilcox_test(rq1_data, Score_Text ~ Group)
  print(mw_text)
  
  # Effect size for the text comparison
  p_text <- mw_text$p
  n_total <- nrow(rq1_data)
  r_text <- effect_size_r(p_text, n_total)
  cat(sprintf("Effect size r = %.3f (|Z|/√N)\n", r_text))
}

# RQ2: China (ZH) vs. US (US) by Region
cat("RQ2: CHINA (ZH) VS. USA (US) BY REGION\n")

# Use 'cluster_2' files for specific outlet comparison
rq2_data <- map_df(region_map$region_code, function(code) {
  us <- load_cluster(paste0("cluster_2_", code, "_us_hy_labeled.csv"))
  zh <- load_cluster(paste0("cluster_2_", code, "_zh_hy_labeled.csv"))
  bind_rows(
    if(!is.null(us)) us %>% mutate(Outlet = "US", code = code),
    if(!is.null(zh)) zh %>% mutate(Outlet = "China", code = code)
  )
}) %>% left_join(region_map, by = c("code" = "region_code"))

cat("--- Exact Scores per Continent, Modality, and Country ---\n")
regional_scores <- rq2_data %>%
  group_by(continent, Outlet) %>%
  summarise(Text_M = mean(Score_Text, na.rm = TRUE), 
            Vis_M  = mean(Score_Visual, na.rm = TRUE), 
            .groups = "drop")
print(as.data.frame(regional_scores))

cat("\n--- Regional Significance (Text) with Holm Adjustment ---\n")
# Holm correction controls family-wise error rate at α = 0.05
holm_results <- rq2_data %>% 
  group_by(continent) %>% 
  rstatix::wilcox_test(Score_Text ~ Outlet) %>% 
  rstatix::adjust_pvalue(method = "holm") %>% 
  rstatix::add_significance()
print(holm_results)

# Add effect sizes for each region (using original unadjusted p)
cat("\n--- Effect sizes (r) per region (Mann-Whitney) ---\n")
region_effect_sizes <- rq2_data %>%
  group_by(continent) %>%
  summarise(
    n_total = n(),
    p_val = wilcox.test(Score_Text ~ Outlet, exact = FALSE)$p.value,
    r = effect_size_r(p_val, n_total),
    .groups = "drop"
  )
print(region_effect_sizes)

# RQ3: Textual vs. Visual Differences
cat("RQ3: MODALITY DIFFERENCES (TEXT VS. VISUAL)\n")

all_files <- list.files(pattern = ".*_labeled\\.csv")
# The paired subsample includes only articles where BOTH text and visual scores
# are present (non-NA). This ensures a direct within‑article comparison.
paired_data <- map_df(all_files, load_cluster) %>%
  filter(!is.na(Score_Text) & !is.na(Score_Visual))

if(nrow(paired_data) > 0) {
  paired_long <- paired_data %>%
    mutate(id = row_number()) %>%
    pivot_longer(cols = c(Score_Text, Score_Visual), names_to = "Modality", values_to = "Score")
  
  cat("--- Paired Wilcoxon Test ---\n")
  paired_test <- rstatix::wilcox_test(paired_long, Score ~ Modality, paired = TRUE)
  print(paired_test)
  
  # Effect size for paired test (r = |Z|/√N, with N = number of pairs)
  n_pairs <- nrow(paired_data)
  p_paired <- paired_test$p
  r_paired <- effect_size_r(p_paired, n_pairs)
  cat(sprintf("Effect size r = %.3f (|Z|/√N)\n", r_paired))
  
  cat("\n--- Spearman Correlation ---\n")
  rho <- cor(paired_data$Score_Text, paired_data$Score_Visual, method = "spearman")
  cat(sprintf("Spearman's ρ = %.3f\n", rho))
  
  cat("\n--- Cohen's Kappa (categorical agreement) ---\n")
  kappa_result <- irr::kappa2(paired_data %>% select(Securitization_Text, Securitization_Visual))
  print(kappa_result)
}
