# LOAD LIBRARIES

library(tidyverse)

# --------------------------------------------------------------------------------- #
# LOAD GAMES DATASET

gbg_dt <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# ---------------------------------------------------------------------------------- #
## SELECT VARIABLES TO USE FOR FULL DECISION TREE MODELING

gbg_dt_full <- gbg_dt %>%
  select(total_result, game_type, weekday, location, total_line, away_rest, home_rest, 
         div_game, roof, surface, temp, wind, total_team_elo, total_qb_elo,
         home_total_offense_rank, home_total_defense_rank, away_total_defense_rank,
         away_total_offense_rank, off_def_diff)

# ------------------------------------------------------------------------------- #
## WRITE FULL PREPPED DATA TO .csv 
#  first decision tree model will include all of the above variables
write.csv(gbg_dt_full, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/gbg_dt_full_r.csv',
          row.names = FALSE)


