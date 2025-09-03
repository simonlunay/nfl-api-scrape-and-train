import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Load the data
df = pd.read_json('nfl_skill_players_2023_weekly_with_opponent.json')

# Map defense tiers to numeric values
tier_map = {'top': 3, 'mid': 2, 'bottom': 1}
df['opponent_defense_tier'] = df['tier'].map(tier_map)

# Ensure 'is_home' exists
df['is_home'] = df.get('is_home', 0)

# Format player name to short form (e.g., A.Rodgers)
def format_name(name):
    parts = name.split()
    if len(parts) < 2:
        return name
    return f"{parts[0][0]}.{parts[-1]}"

df['player_short_name'] = df['player_name'].apply(format_name)

# Define target stats
num_cols = ['passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds',
            'receptions', 'receiving_yards', 'receiving_tds',
            'fumbles_lost']

# Create season average columns
to_avg = df[['player_short_name', 'week'] + num_cols].copy()
for col in num_cols:
    df[col + '_season_avg'] = df.groupby('player_short_name')[col].transform('mean')

# Define features for each position
qb_features = [c + '_season_avg' for c in num_cols] + ['opponent_defense_tier', 'is_home']
skill_features = [c + '_season_avg' for c in num_cols if c not in ['passing_yards', 'passing_tds', 'interceptions']] + ['opponent_defense_tier', 'is_home']

# Define output stats
qb_target_stats = num_cols
skill_target_stats = [s for s in num_cols if s not in ['passing_yards', 'passing_tds', 'interceptions']]

# Prepare data
qb_df = df[df['position'] == 'QB'].dropna(subset=qb_features + qb_target_stats)
skill_df = df[df['position'].isin(['RB', 'WR', 'TE'])].dropna(subset=skill_features + skill_target_stats)

# Split features and targets
X_qb = qb_df[qb_features]
y_qb = qb_df[qb_target_stats]
X_skill = skill_df[skill_features]
y_skill = skill_df[skill_target_stats]

# Train models
model_qb = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model_qb.fit(X_qb, y_qb)

model_skill = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42))
model_skill.fit(X_skill, y_skill)

# Save models and helper data
joblib.dump(model_qb, 'qb_multi_model.joblib')
joblib.dump(model_skill, 'skill_multi_model.joblib',compress=5)

helper_data = {
    'tier_map': tier_map,
    'qb_features': qb_features,
    'skill_features': skill_features,
    'qb_target_stats': qb_target_stats,
    'skill_target_stats': skill_target_stats,
}
joblib.dump(helper_data, 'helper_data.joblib')

print("Models and helper data saved!")
