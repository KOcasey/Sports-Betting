import pickle

model = pickle.load(open('saved_models/rf_model.pkl','rb')) #read mode

def make_prediction(total_line, temp, wind, total_qb_elo, team_elo_diff, qb_elo_diff, avg_home_total_yards, avg_away_total_yards, avg_home_total_yards_against, avg_away_total_yards_against, div_game, roof_type, surface_type):
  roof_closed = 0
  roof_dome = 0
  roof_open = 0
  roof_outdoors = 0
  if roof_type.lower() == 'dome':
      roof_dome = 1
  elif roof_type.lower() == 'outdoors':
      roof_outdoors = 1
  elif roof_type.lower() == 'open':
      roof_open = 1
  elif roof_type.lower() == 'closed':
      roof_closed = 1

  surface_astroplay = 0
  surface_astroturf = 0
  surface_dessograss = 0
  surface_fieldturf = 0
  surface_grass = 0
  surface_matrixturf = 0
  surface_sportturf = 0
  if surface_type.lower() == 'astroplay':
      surface_astroplay = 1
  elif surface_type.lower() == 'astroturf':
      surface_astroturf = 1
  elif surface_type.lower() == 'dessograss':
      surface_dessograss = 1
  elif surface_type.lower() == 'fieldturf':
      surface_fieldturf = 1
  elif surface_type.lower() == 'grass':
      surface_grass = 1
  elif surface_type.lower() == 'matrixturf':
      surface_matrixturf = 1
  elif surface_type.lower() == 'sportturf':
      surface_sportturf = 1

  #get prediction
  input_cols = [[total_line, temp, wind, total_qb_elo, team_elo_diff, qb_elo_diff, avg_home_total_yards, avg_away_total_yards, avg_home_total_yards_against,
                  avg_away_total_yards_against, div_game, roof_closed, roof_dome, roof_open, roof_outdoors, surface_astroplay, surface_astroturf, surface_dessograss,
                  surface_fieldturf, surface_grass, surface_matrixturf, surface_sportturf]]
  prediction = model.predict(input_cols)
  if prediction == 0:
      prediction = 'Over'
  else:
      prediction = 'Under'

  return prediction