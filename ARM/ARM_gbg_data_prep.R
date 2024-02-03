library(dplyr)

# Read in games dataset
gbg_cl <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# Select relevant columns for Association Rule Mining
gbg_cl <- gbg_cl %>% select(season, week, total_result,weekday, location, game_type, roof, surface, 
                            referee, total_line, div_game, temp, wind, 
                            total_team_elo, total_qb_elo, home_total_offense_rank,
                            home_total_defense_rank, away_total_offense_rank,
                            away_total_defense_rank)

# ---------------------------------------------------------------------------- #
# Ranks the total_team_elo and total_qb_elo of games based on where they stand the current week
# Ranks are 1-16 where 16 is the best and 1 is the worst
seasons <- unique(gbg_cl$season)

gbg_cl$total_team_elo_rank <- 0
gbg_cl$total_qb_elo_rank <- 0

for (season_num in seasons){
  weeks <- unique(gbg_cl$week[gbg_cl$season == season_num])
  
  for (week_num in weeks) {
    # total_team_elo ranks
    team_list <- gbg_cl$total_team_elo[gbg_cl$season == season_num & gbg_cl$week == week_num]
    rank_list <- rank(team_list, ties.method = 'max')
    gbg_cl$total_team_elo_rank[gbg_cl$season == season_num & gbg_cl$week == week_num] <- rank_list
    
    
    # total_qb_elo_ranks
    qb_list <- gbg_cl$total_qb_elo[gbg_cl$season == season_num & gbg_cl$week == week_num]
    rank_list <- rank(qb_list, ties.method = 'max')
    gbg_cl$total_qb_elo_rank[gbg_cl$season == season_num & gbg_cl$week == week_num] <- rank_list
  }
}

# ---------------------------------------------------------------------------- #
# BINNING

# Bins the necessary columns
gbg_cl <- gbg_cl %>%
  mutate(div_game = ifelse(div_game == 1, 'div_game', 'non_div_game'),
         total_line_bin = cut(total_line, labels=c('Under 35', '35-39', '40-50',
                                                   '51-55', 'Over 55'), 
                              breaks=c(0, 34, 39, 50, 55, 100)),
         temp_bin = cut(temp, labels=c('Below 0', '0-10', '11-20', '21-32',
                                       '33-45', '46-60', '61-72', '73-84', 'Over 84'),
                        breaks=c(-50, -1, 10, 20, 32, 45, 60, 72, 84, 200)),
         wind_bin = cut(wind, labels=c('No Wind', 'Light Wind', 'Moderate Wind',
                                       'Strong Wind', 'Gale', 'Strong Gale'), 
                        breaks=c(-50, 0, 12, 24, 31, 46, 150)),
         total_team_elo_bin = cut(total_team_elo_rank, labels=c('1-5', '6-10', 
                                                           '11+'),
                                  breaks=c(0, 5, 10, 20)),
         total_qb_elo_bin = cut(total_qb_elo_rank, labels=c('1-5', '6-10',
                                                             '11+'),
                                  breaks = c(0, 5, 10 , 20)),
         total_qb_elo_pos = ifelse(total_qb_elo < 0, 'negative_qb_elo', 
                                   'positive_qb_elo'),
         home_total_offense_rank_bin = cut(home_total_offense_rank, labels=c('1-5', '6-10',
                                                                         '11-15', '16-20',
                                                                         '21-25', '25+'),
                                       breaks=c(0, 5, 10, 15, 20, 25, 35)),
         away_total_offense_rank_bin = cut(away_total_offense_rank, labels=c('1-5', '6-10',
                                                                         '11-15', '16-20',
                                                                         '21-25', '25+'),
                                       breaks=c(0, 5, 10, 15, 20, 25, 35)),
         home_total_defense_rank_bin = cut(home_total_defense_rank, labels=c('1-5', '6-10',
                                                                         '11-15', '16-20',
                                                                         '21-25', '25+'),
                                       breaks=c(0, 5, 10, 15, 20, 25, 35)),
         away_total_defense_rank_bin = cut(away_total_defense_rank, labels=c('1-5', '6-10',
                                                                         '11-15', '16-20',
                                                                         '21-25', '25+'),
                                       breaks=c(0, 5, 10, 15, 20, 25, 35)))


# For the binned columns pastes on words to indicate what column the rank is from
# There will be no columns names when ARM is performed so this is necessary
gbg_cl$total_line_bin <- paste('Total of ', gbg_cl$total_line_bin)
gbg_cl$temp_bin <- paste(gbg_cl$temp_bin, ' degrees')
gbg_cl$total_team_elo_bin <- paste('Total Team ELO Rank of ', gbg_cl$total_team_elo_bin)
gbg_cl$total_qb_elo_bin <- paste('Total QB ELO Rank of ', gbg_cl$total_qb_elo_bin)
gbg_cl$home_total_offense_rank_bin <- paste('Total Home Offense Rank of ', gbg_cl$home_total_offense_rank_bin)
gbg_cl$away_total_offense_rank_bin <- paste('Total Away Offense Rank of ', gbg_cl$away_total_offense_rank_bin)
gbg_cl$home_total_defense_rank_bin <- paste('Total Home Defense Rank of ', gbg_cl$home_total_defense_rank_bin)
gbg_cl$away_total_defense_rank_bin <- paste('Total Away Defense Rank of ', gbg_cl$away_total_defense_rank_bin)

# --------------------------------------------------------------------------- #
# Remove Unneeded columns
gbg_cl <- gbg_cl %>% select(-c('season', 'week', 'total_line', 'temp', 'wind',
                               'total_team_elo', 'total_qb_elo', 
                               'home_total_offense_rank', 'away_total_offense_rank',
                               'home_total_defense_rank', 'away_total_defense_rank',
                               'total_team_elo_rank', 'total_qb_elo_rank')) 


# Write prepped data to .csv file
# Doesn't include row names
# After writing, remove the first row from the .csv file to get rid of column names
write.csv(gbg_cl, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/ARM/prepped_data/gbg_prepped_for_ARM.csv',
          row.names=FALSE)

