import joblib
import pandas as pd

# Load saved objects
model_qb = joblib.load('qb_model.joblib')
model_skill = joblib.load('skill_model.joblib')
helper_data = joblib.load('helper_data.joblib')

tier_map = helper_data['tier_map']
num_cols = helper_data['num_cols']
qb_features = helper_data['qb_features']
skill_features = helper_data['skill_features']

def format_name(name):
    parts = name.split()
    if len(parts) < 2:
        return name
    return f"{parts[0][0]}.{parts[-1]}"

def predict_fantasy_points(player_name, opponent_team, home_away, player_data_df):
    short_name = format_name(player_name)
    if short_name not in player_data_df['player_short_name'].values:
        return f"Player '{player_name}' not found in dataset."

    player_pos = player_data_df[player_data_df['player_short_name'] == short_name]['position'].iloc[0]

    tiers = player_data_df[player_data_df['opponent_team'] == opponent_team]['tier'].map(tier_map).dropna().unique()
    if len(tiers) == 0:
        return f"Opponent team '{opponent_team}' not found or tier missing."
    opponent_defense_tier = tiers[0]

    is_home = 1 if home_away.lower() == 'home' else 0

    player_data = player_data_df[player_data_df['player_short_name'] == short_name].sort_values('week').iloc[-1]

    if player_pos == 'QB':
        qb_features_base = [c + '_season_avg' for c in num_cols]
        feat_vals = [player_data[col] for col in qb_features_base]
        X_input = feat_vals + [opponent_defense_tier, is_home]
        X_df = pd.DataFrame([X_input], columns=qb_features)
        pred_fp = model_qb.predict(X_df)[0]
    else:
        skill_feats_base = [c + '_season_avg' for c in num_cols if c not in ['passing_yards', 'passing_tds', 'interceptions']]
        feat_vals = [player_data[col] for col in skill_feats_base]
        X_input = feat_vals + [opponent_defense_tier, is_home]
        X_df = pd.DataFrame([X_input], columns=skill_features)
        pred_fp = model_skill.predict(X_df)[0]

    return f"Predicted fantasy points for {player_name} vs {opponent_team} {home_away.title()}: {pred_fp:.2f}"
