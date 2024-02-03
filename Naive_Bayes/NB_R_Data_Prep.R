# LOAD LIBRARIES

library(tidyverse)
library(corrplot)

# --------------------------------------------------------------------------------- #
# LOAD GAMES DATASET

gbg_nb <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# ---------------------------------------------------------------------------------- #
## SELECT VARIABLES TO USE FOR FULL DECISION TREE MODELING

gbg_nb_full <- gbg_nb %>%
  select(total_result, game_type, weekday, location, total_line, away_rest, home_rest, 
         div_game, roof, surface, temp, wind, total_team_elo, total_qb_elo,
         home_total_offense_rank, home_total_defense_rank, away_total_defense_rank,
         away_total_offense_rank, off_def_diff)

# ------------------------------------------------------------------------------- #
## CHECK FOR CORRELATION
# Naive Bayes assumes independence so we don't want correlation between variables

gbg_nb_num <- gbg_nb_full %>%
  keep(is.numeric)

col4 <- colorRampPalette(c("Black", "darkgrey", "#CFB87C"))

corrplot(cor(gbg_nb_num), method="ellipse", col=col4(100),
         addCoef.col = "black", tl.col="black", tl.cex=0.6, cl.cex=0.75,
         number.cex = 0.5)

# ------------------------------------------------------------------------------- #
## WRITE FULL PREPPED DATA TO .csv 
#  first decision tree model will include all of the above variables
write.csv(gbg_nb_full, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Naive_Bayes/prepped_data/gbg_nb_full_r.csv',
          row.names = FALSE)


