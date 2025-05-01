from flask import Flask, request, render_template, jsonify
import logging
import os
import google.cloud.storage
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from scipy.stats import poisson
from math import ceil, floor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Logging initialized at startup - Cloud Run Predictor app")

app = Flask(__name__)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = google.cloud.storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        logger.info(f"Checking if {source_blob_name} exists in gs://{bucket_name}/")
        if not blob.exists():
            raise FileNotFoundError(f"{source_blob_name} not found in gs://{bucket_name}/")
        logger.info(f"Downloading {source_blob_name} to {destination_file_name}")
        blob.download_to_filename(destination_file_name)
        logger.info(f"Successfully downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        logger.error(f"Failed to download {source_blob_name}: {str(e)}")
        raise

class TeamScorePredictor:
    def __init__(self):
        self.team_stats = {}
        self.conference_stats = {}
        self.conference_power_trends = {}
        self.head_to_head = {}
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.team_encoder = {}
        self.max_mean = 15.0
        self.min_mean = 0.0
        self.conf_power_impact = None

    def preprocess_data(self, dataset):
        dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y%m%d')
        dataset = dataset.sort_values('Date')

        all_teams = set(dataset['HomeTeam']).union(set(dataset['AwayTeam']))
        self.team_encoder = {team: i for i, team in enumerate(all_teams)}

        inter_conf_games = dataset[dataset['HomeConference'] != dataset['AwayConference']]
        conf_power_diffs = inter_conf_games['HomeConfPower'] - inter_conf_games['AwayConfPower']
        score_diffs = inter_conf_games['HomeMean'] - inter_conf_games['AwayMean']
        if len(inter_conf_games) > 0:
            slope, _ = np.polyfit(conf_power_diffs, score_diffs, 1)
            self.conf_power_impact = slope
        else:
            self.conf_power_impact = 1.0

        for index, row in dataset.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_id = f"{home_team}_vs_{away_team}"

            if home_team not in self.team_stats:
                self.team_stats[home_team] = {
                    'scores': [], 'means': [], 'conference': row['HomeConference'],
                    'wins': 0, 'losses': 0, 'dates': [], 'home_scores': [], 'away_scores': [],
                    'conf_scores': {}, 'win_prob': []
                }
            if away_team not in self.team_stats:
                self.team_stats[away_team] = {
                    'scores': [], 'means': [], 'conference': row['AwayConference'],
                    'wins': 0, 'losses': 0, 'dates': [], 'home_scores': [], 'away_scores': [],
                    'conf_scores': {}, 'win_prob': []
                }
            if match_id not in self.head_to_head:
                self.head_to_head[match_id] = {'home_scores': [], 'away_scores': []}

            if row['HomeConference'] not in self.conference_stats:
                self.conference_stats[row['HomeConference']] = []
            if row['AwayConference'] not in self.conference_stats:
                self.conference_stats[row['AwayConference']] = []
            if row['HomeConference'] not in self.conference_power_trends:
                self.conference_power_trends[row['HomeConference']] = {'power': [], 'dates': []}
            if row['AwayConference'] not in self.conference_power_trends:
                self.conference_power_trends[row['AwayConference']] = {'power': [], 'dates': []}

            if row['AwayConference'] not in self.team_stats[home_team]['conf_scores']:
                self.team_stats[home_team]['conf_scores'][row['AwayConference']] = []
            if row['HomeConference'] not in self.team_stats[away_team]['conf_scores']:
                self.team_stats[away_team]['conf_scores'][row['HomeConference']] = []

            self.team_stats[home_team]['scores'].append(row['HomeActualScore'])
            self.team_stats[away_team]['scores'].append(row['AwayActualScore'])
            self.team_stats[home_team]['means'].append(row['HomeMean'])
            self.team_stats[away_team]['means'].append(row['AwayMean'])
            self.team_stats[home_team]['wins'] += row['HomeWins']
            self.team_stats[home_team]['losses'] += row['HomeLoses']
            self.team_stats[away_team]['wins'] += row['AwayWins']
            self.team_stats[away_team]['losses'] += row['AwayLoses']
            self.team_stats[home_team]['dates'].append(row['Date'])
            self.team_stats[away_team]['dates'].append(row['Date'])
            self.team_stats[home_team]['home_scores'].append(row['HomeActualScore'])
            self.team_stats[away_team]['away_scores'].append(row['HomeActualScore'])
            self.team_stats[home_team]['conf_scores'][row['AwayConference']].append(row['HomeActualScore'])
            self.team_stats[away_team]['conf_scores'][row['HomeConference']].append(row['AwayActualScore'])
            self.team_stats[home_team]['win_prob'].append(row['HomeWinProbability'])
            self.team_stats[away_team]['win_prob'].append(row['AwayWinProbability'])

            self.head_to_head[match_id]['home_scores'].append(row['HomeActualScore'])
            self.head_to_head[match_id]['away_scores'].append(row['AwayActualScore'])

            self.conference_stats[row['HomeConference']].append(row['HomeConfPower'])
            self.conference_stats[row['AwayConference']].append(row['AwayConfPower'])
            self.conference_power_trends[row['HomeConference']]['power'].append(row['HomeConfPower'])
            self.conference_power_trends[row['AwayConference']]['power'].append(row['AwayConfPower'])
            self.conference_power_trends[row['HomeConference']]['dates'].append(row['Date'])
            self.conference_power_trends[row['AwayConference']]['dates'].append(row['Date'])

        for team in self.team_stats:
            self.team_stats[team]['avg_score'] = np.mean(self.team_stats[team]['scores'])
            self.team_stats[team]['avg_mean'] = np.mean(self.team_stats[team]['means'])
            self.team_stats[team]['win_rate'] = (self.team_stats[team]['wins'] /
                                                (self.team_stats[team]['wins'] + self.team_stats[team]['losses'] + 1e-5))
            self.team_stats[team]['avg_win_prob'] = np.mean(self.team_stats[team]['win_prob']) / 100
            self.team_stats[team]['recent_scores'] = self.team_stats[team]['scores'][-5:] if len(self.team_stats[team]['scores']) >= 5 else self.team_stats[team]['scores']
            self.team_stats[team]['recent_means'] = self.team_stats[team]['means'][-5:] if len(self.team_stats[team]['means']) >= 5 else self.team_stats[team]['means']
            self.team_stats[team]['avg_recent_score'] = np.mean(self.team_stats[team]['recent_scores'])
            self.team_stats[team]['avg_recent_mean'] = np.mean(self.team_stats[team]['recent_means'])
            self.team_stats[team]['home_avg'] = np.mean(self.team_stats[team]['home_scores']) if self.team_stats[team]['home_scores'] else 0
            self.team_stats[team]['away_avg'] = np.mean(self.team_stats[team]['away_scores']) if self.team_stats[team]['away_scores'] else 0
            for conf in self.team_stats[team]['conf_scores']:
                self.team_stats[team]['conf_scores'][conf] = np.mean(self.team_stats[team]['conf_scores'][conf]) if self.team_stats[team]['conf_scores'][conf] else 0
            if len(self.team_stats[team]['dates']) > 1:
                x = np.array([(d - self.team_stats[team]['dates'][0]).days for d in self.team_stats[team]['dates']])
                y = np.array(self.team_stats[team]['means'])
                self.team_stats[team]['trend'] = np.polyfit(x, y, 1)[0]
            else:
                self.team_stats[team]['trend'] = 0

        for conf in self.conference_stats:
            self.conference_stats[conf] = np.mean(self.conference_stats[conf])

        for conf in self.conference_power_trends:
            if len(self.conference_power_trends[conf]['dates']) > 1:
                x = np.array([(d - self.conference_power_trends[conf]['dates'][0]).days for d in self.conference_power_trends[conf]['dates']])
                x = x - x[0]
                y = np.array(self.conference_power_trends[conf]['power'])
                if len(x) > 1 and np.std(x) > 0:
                    self.conference_power_trends[conf]['trend'] = np.polyfit(x, y, 1)[0]
                else:
                    self.conference_power_trends[conf]['trend'] = 0
                self.conference_power_trends[conf]['std'] = np.std(self.conference_power_trends[conf]['power'])
            else:
                self.conference_power_trends[conf]['trend'] = 0
                self.conference_power_trends[conf]['std'] = 0

        for match_id in self.head_to_head:
            self.head_to_head[match_id]['avg_home'] = np.mean(self.head_to_head[match_id]['home_scores'])
            self.head_to_head[match_id]['avg_away'] = np.mean(self.head_to_head[match_id]['away_scores'])

    def create_features(self, home_team, away_team):
        if home_team not in self.team_stats or away_team not in self.team_stats:
            missing_team = home_team if home_team not in self.team_stats else away_team
            logger.info(f"Team '{missing_team}' not found in the dataset. Cannot proceed with prediction.")
            return None

        match_id = f"{home_team}_vs_{away_team}"
        home_conf = self.team_stats[home_team]['conference']
        away_conf = self.team_stats[away_team]['conference']
        home_conf_avg = self.team_stats[home_team]['conf_scores'].get(away_conf, 0)
        away_conf_avg = self.team_stats[away_team]['conf_scores'].get(home_conf, 0)

        conf_power_diff = self.conference_stats[home_conf] - self.conference_stats[away_conf]
        team_strength_diff = self.team_stats[home_team]['avg_score'] - self.team_stats[away_team]['avg_score']
        recent_form_diff = self.team_stats[home_team]['avg_recent_mean'] - self.team_stats[away_team]['avg_recent_mean']

        features = [
            self.team_stats[home_team]['avg_score'],
            self.team_stats[away_team]['avg_score'],
            self.team_stats[home_team]['avg_mean'],
            self.team_stats[away_team]['avg_mean'],
            self.conference_stats[home_conf],
            self.conference_stats[away_conf],
            self.team_stats[home_team]['win_rate'],
            self.team_stats[away_team]['win_rate'],
            self.team_stats[home_team]['avg_recent_score'],
            self.team_stats[away_team]['avg_recent_score'],
            self.team_stats[home_team]['avg_recent_mean'],
            self.team_stats[away_team]['avg_recent_mean'],
            self.team_stats[home_team]['trend'],
            self.team_stats[away_team]['trend'],
            self.head_to_head[match_id]['avg_home'] if match_id in self.head_to_head else self.team_stats[home_team]['home_avg'],
            self.head_to_head[match_id]['avg_away'] if match_id in self.head_to_head else self.team_stats[away_team]['away_avg'],
            self.team_stats[home_team]['home_avg'],
            self.team_stats[away_team]['away_avg'],
            home_conf_avg,
            away_conf_avg,
            conf_power_diff,
            team_strength_diff,
            recent_form_diff,
            self.team_encoder[home_team],
            self.team_encoder[away_team]
        ]
        return np.array(features).reshape(1, -1)

    def calculate_moneyline_probabilities(self, home_mean, away_mean):
        epsilon = 1e-5
        home_mean = max(home_mean, 0) + epsilon
        away_mean = max(away_mean, 0) + epsilon
        exponent = 1.83
        home_prob = (home_mean ** exponent) / (home_mean ** exponent + away_mean ** exponent)
        away_prob = 1 - home_prob
        return home_prob, away_prob

    def convert_probability_to_decimal_odds(self, prob):
        if prob <= 0 or prob >= 1:
            raise ValueError("Probability must be between 0 and 1")
        return round(1 / prob, 2)

    def determine_total_line(self, total_mean):
        if abs(total_mean - round(total_mean)) < 1e-10:
            return float(round(total_mean) - 0.5)
        nearest_integer = round(total_mean)
        if total_mean >= nearest_integer:
            return float(nearest_integer + 0.5)
        return float(nearest_integer - 0.5)

    def calculate_over_under_probabilities(self, total_mean):
        total_line = self.determine_total_line(total_mean)
        k_max = int(total_line - 0.5)
        under_prob = poisson.cdf(k_max, total_mean)
        over_prob = 1 - under_prob
        over_prob_percent = over_prob * 100
        under_prob_percent = under_prob * 100
        over_decimal_odds = self.convert_probability_to_decimal_odds(over_prob)
        under_decimal_odds = self.convert_probability_to_decimal_odds(under_prob)
        return total_line, over_prob_percent, under_prob_percent, over_decimal_odds, under_decimal_odds

    def predict(self, home_team, away_team):
        features = self.create_features(home_team, away_team)
        if features is None:
            return None

        features_scaled = features.copy()
        features_scaled[:, :23] = self.scaler.transform(features[:, :23])

        with tf.device('/GPU:0'):
            prediction_scaled = self.model.predict(features_scaled, verbose=0)
            prediction = self.target_scaler.inverse_transform(prediction_scaled)

        home_conf = self.team_stats[home_team]['conference']
        away_conf = self.team_stats[away_team]['conference']
        home_conf_power = self.conference_stats[home_conf]
        away_conf_power = self.conference_stats[away_conf]
        home_win_prob = self.team_stats[home_team]['avg_win_prob']
        away_win_prob = self.team_stats[away_team]['avg_win_prob']

        if home_conf != away_conf:
            conf_power_diff = home_conf_power - away_conf_power
            score_adjustment = conf_power_diff * self.conf_power_impact
            win_prob_factor = (home_win_prob - away_win_prob + 1) / 2
            adjusted_home = prediction[0][0] + (score_adjustment * win_prob_factor)
            adjusted_away = prediction[0][1] - (score_adjustment * win_prob_factor)

            adjusted_home = max(self.min_mean, min(self.max_mean, adjusted_home))
            adjusted_away = max(self.min_mean, min(self.max_mean, adjusted_away))
        else:
            adjusted_home = prediction[0][0]
            adjusted_away = prediction[0][1]

        home_moneyline, away_moneyline = self.calculate_moneyline_probabilities(adjusted_home, adjusted_away)
        home_decimal_odds = self.convert_probability_to_decimal_odds(home_moneyline)
        away_decimal_odds = self.convert_probability_to_decimal_odds(away_moneyline)
        total_mean = adjusted_home + adjusted_away
        total_line, over_prob_percent, under_prob_percent, over_decimal_odds, under_decimal_odds = self.calculate_over_under_probabilities(total_mean)

        return {
            'PredictedMeans': {
                'HomeMean': float(adjusted_home),
                'AwayMean': float(adjusted_away)
            },
            'MoneyLine': {
                'HomeWinProbability': float(home_moneyline),
                'AwayWinProbability': float(away_moneyline),
                'HomeDecimalOdds': float(home_decimal_odds),
                'AwayDecimalOdds': float(away_decimal_odds)
            },
            'OverUnder': {
                'TotalLine': float(total_line),
                'OverProbabilityPercent': float(over_prob_percent),
                'UnderProbabilityPercent': float(under_prob_percent),
                'OverDecimalOdds': float(over_decimal_odds),
                'UnderDecimalOdds': float(under_decimal_odds)
            }
        }

# Initialize predictor at startup
predictor = None
BUCKET_NAME = "mean-predictor"
MODEL_PATH = '/tmp/team_score_predictor_model.keras'
DATA_PATH = '/tmp/massey_all_data.csv'
SCALER_PATH = '/tmp/scaler.pkl'
TARGET_SCALER_PATH = '/tmp/target_scaler.pkl'

logger.info("Starting predictor initialization")
try:
    if not os.path.exists(MODEL_PATH):
        download_blob(BUCKET_NAME, 'team_score_predictor_model.keras', MODEL_PATH)
    if not os.path.exists(DATA_PATH):
        download_blob(BUCKET_NAME, 'massey_all_data.csv', DATA_PATH)
    if not os.path.exists(SCALER_PATH):
        download_blob(BUCKET_NAME, 'scaler.pkl', SCALER_PATH)
    if not os.path.exists(TARGET_SCALER_PATH):
        download_blob(BUCKET_NAME, 'target_scaler.pkl', TARGET_SCALER_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH} after download")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH} after download")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH} after download")
    if not os.path.exists(TARGET_SCALER_PATH):
        raise FileNotFoundError(f"Target scaler file not found at {TARGET_SCALER_PATH} after download")

    predictor = TeamScorePredictor()
    logger.info("Reading CSV")
    dataset = pd.read_csv(DATA_PATH)
    predictor.preprocess_data(dataset)
    logger.info("Loading model")
    predictor.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    predictor.model.compile(loss='mae', metrics=['mae'])
    logger.info("Loading scalers")
    with open(SCALER_PATH, 'rb') as f:
        predictor.scaler = pickle.load(f)
    with open(TARGET_SCALER_PATH, 'rb') as f:
        predictor.target_scaler = pickle.load(f)
    logger.info("Predictor initialized successfully")
except Exception as e:
    logger.error(f"Startup failed: {str(e)}")
    raise SystemExit(f"Startup failed: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def predict_score():
    try:
        logger.info(f"Request received: {request.method}")
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        logger.info(f"Checking template at: {template_path}")
        
        if not os.path.exists(template_path):
            logger.error(f"Template not found at: {template_path}")
            return "Template not found, but app is alive!", 200

        if request.method == 'POST':
            home_team = request.form.get('home_team')
            away_team = request.form.get('away_team')
            logger.info(f"POST request - Home: {home_team}, Away: {away_team}")
            
            # Check if request is AJAX (via content type or custom header)
            is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/x-www-form-urlencoded'
            
            if home_team and away_team:
                result = predictor.predict(home_team, away_team)
                if result is None:
                    error_msg = f"Prediction failed. One or both teams not found in the dataset."
                    logger.warning(f"Prediction failed for {home_team} vs {away_team}")
                    if is_ajax:
                        return jsonify({"error": error_msg})
                    else:
                        return render_template('index.html', error=error_msg, 
                                             home_team=home_team, away_team=away_team)
                
                logger.info(f"Prediction result: {result}")
                # Format prediction to match index.html expectations
                prediction = {
                    'PredictedMeans': {
                        'home_mean': f"{result['PredictedMeans']['HomeMean']:.2f}",
                        'away_mean': f"{result['PredictedMeans']['AwayMean']:.2f}"
                    },
                    'WinProbabilities': {
                        'home_win_probability': f"{result['MoneyLine']['HomeWinProbability'] * 100:.2f}",
                        'away_win_probability': f"{result['MoneyLine']['AwayWinProbability'] * 100:.2f}"
                    },
                    'MoneyLine': {
                        'home_decimal_odds': f"{result['MoneyLine']['HomeDecimalOdds']:.2f}",
                        'away_decimal_odds': f"{result['MoneyLine']['AwayDecimalOdds']:.2f}"
                    },
                    'OverUnder': {
                        'total_line': f"{result['OverUnder']['TotalLine']:.1f}",
                        'over_probability_percent': f"{result['OverUnder']['OverProbabilityPercent']:.2f}",
                        'under_probability_percent': f"{result['OverUnder']['UnderProbabilityPercent']:.2f}",
                        'over_decimal_odds': f"{result['OverUnder']['OverDecimalOdds']:.2f}",
                        'under_decimal_odds': f"{result['OverUnder']['UnderDecimalOdds']:.2f}"
                    }
                }
                
                if is_ajax:
                    return jsonify({"prediction": prediction})
                else:
                    return render_template('index.html', 
                                         prediction=prediction,
                                         home_team=home_team, 
                                         away_team=away_team)
            else:
                error_msg = "Please provide both home and away team names"
                logger.warning("Missing team names in POST request")
                if is_ajax:
                    return jsonify({"error": error_msg})
                else:
                    return render_template('index.html', error=error_msg)
                    
        logger.info("Serving GET request")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Request handling failed: {str(e)}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"error": f"Server error: {str(e)}"}), 500
        return f"Server error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)