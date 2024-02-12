from flask import Flask, render_template, request, jsonify
from helper_functions import make_prediction

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')
    #return jsonify({'hello': 'from new edit template api auto-deployed with GitHub actions!'}), 200

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        total_line = float(request.form["total_line"])
        temp = float(request.form["temp"])
        wind = float(request.form["wind"])
        total_qb_elo = float(request.form["total_qb_elo"])
        team_elo_diff = float(request.form["team_elo_diff"])
        qb_elo_diff = float(request.form["qb_elo_diff"])
        avg_home_total_yards = float(request.form["avg_home_total_yards"])
        avg_away_total_yards = float(request.form["avg_away_total_yards"])
        avg_home_total_yards_against = float(request.form["avg_home_total_yards_against"])
        avg_away_total_yards_against = float(request.form["avg_away_total_yards_against"])
        div_game = int(request.form["div_game"])
        roof_type = str(request.form["roof_type"])
        surface_type = str(request.form["surface_type"])

        prediction = make_prediction(total_line, temp, wind, total_qb_elo, team_elo_diff, qb_elo_diff, avg_home_total_yards, avg_away_total_yards, avg_home_total_yards_against, avg_away_total_yards_against, div_game, roof_type, surface_type)
        return render_template("index.html", prediction_text='The Predicted Total Result is: ' + str(prediction))

if __name__ == "__main__":
    print('Hi')
    app.run(debug=True, port=5000)