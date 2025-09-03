import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load data
df = pd.read_json('nfl_skill_players_2023_weekly_with_opponent.json')

# 2. Clean stats
stat_cols = [
    'passing_yards', 'passing_tds', 'interceptions',
    'rushing_yards', 'rushing_tds',
    'receptions', 'receiving_yards', 'receiving_tds',
    'fumbles_lost'
]
for col in stat_cols:
    df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

# 3. Calculate fantasy points
df['fantasy_points'] = (
    df['passing_yards'] / 25 +
    df['passing_tds'] * 4 +
    df['interceptions'] * -2 +
    df['rushing_yards'] / 10 +
    df['rushing_tds'] * 6 +
    df['receptions'] * 1 +
    df['receiving_yards'] / 10 +
    df['receiving_tds'] * 6 +
    df['fumbles_lost'] * -2
)

# 4. Season averages per player
season_avg = df.groupby(['player_name', 'position']).agg({
    'passing_yards': 'mean',
    'passing_tds': 'mean',
    'interceptions': 'mean',
    'rushing_yards': 'mean',
    'rushing_tds': 'mean',
    'receptions': 'mean',
    'receiving_yards': 'mean',
    'receiving_tds': 'mean',
    'fumbles_lost': 'mean',
    'fantasy_points': 'mean'
}).reset_index()

season_avg = season_avg.rename(columns={
    'passing_yards': 'passing_yards_season_avg',
    'passing_tds': 'passing_tds_season_avg',
    'interceptions': 'interceptions_season_avg',
    'rushing_yards': 'rushing_yards_season_avg',
    'rushing_tds': 'rushing_tds_season_avg',
    'receptions': 'receptions_season_avg',
    'receiving_yards': 'receiving_yards_season_avg',
    'receiving_tds': 'receiving_tds_season_avg',
    'fumbles_lost': 'fumbles_lost_season_avg',
    'fantasy_points': 'fantasy_points_season_avg'
})

df = df.merge(season_avg, on=['player_name', 'position'], how='left')

# 5. Map teams to defense tiers (example from your data)
team_to_tier = {
    'KC': 'top', 'BAL': 'top', 'SF': 'top', 'LAC': 'top', 'DAL': 'top',
    'PIT': 'top', 'JAX': 'top', 'BUF': 'top', 'LAR': 'top', 'NE': 'top',
    'TEN': 'mid', 'IND': 'mid', 'LV': 'mid', 'PHI': 'mid', 'ARI': 'mid',
    'DEN': 'mid', 'MIN': 'mid', 'SEA': 'mid', 'GB': 'mid', 'MIA': 'mid',
    'CIN': 'bottom', 'TB': 'bottom', 'ATL': 'bottom', 'NYG': 'bottom', 'NO': 'bottom',
    'CAR': 'bottom', 'HOU': 'bottom', 'WAS': 'bottom', 'CLE': 'bottom', 'NYJ': 'bottom',
    'CHI': 'bottom', 'DET': 'bottom'
}

# Map tiers to numeric values
tier_map = {'top': 2, 'mid': 1, 'bottom': 0}

# 6. Add 'opponent_defense_tier' column by mapping opponent_team to tier
df['opponent_defense_tier'] = df['opponent_team'].map(team_to_tier)

# 7. Encode defense tier numerically
df['opponent_defense_tier_num'] = df['opponent_defense_tier'].map(tier_map)

# 8. Encode home/away
df['is_home'] = df['home_away'].apply(lambda x: 1 if x == 'home' else 0)

# 9. Features
features = [
    'passing_yards_season_avg',
    'passing_tds_season_avg',
    'interceptions_season_avg',
    'rushing_yards_season_avg',
    'rushing_tds_season_avg',
    'receptions_season_avg',
    'receiving_yards_season_avg',
    'receiving_tds_season_avg',
    'fumbles_lost_season_avg',
    'opponent_defense_tier_num',
    'is_home'
]

model_df = df.dropna(subset=features + ['fantasy_points'])

X = model_df[features]
y = model_df['fantasy_points']

# 10. Train/test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f'Model MAE: {mae:.2f} fantasy points')

# 11. Prediction function with opponent team name and home/away
def predict_fantasy_points(player_name, opponent_team, home_away):
    player_stats = season_avg[season_avg['player_name'] == player_name]
    if player_stats.empty:
        return f"No data found for player {player_name}"

    player_stats = player_stats.iloc[0]

    # Map team to tier
    tier_str = team_to_tier.get(opponent_team.upper())
    if tier_str is None:
        return f"Unknown opponent team '{opponent_team}'"

    tier_num = tier_map[tier_str]
    is_home = 1 if home_away.lower() == 'home' else 0

    feat_vec = pd.DataFrame([{
        'passing_yards_season_avg': player_stats['passing_yards_season_avg'],
        'passing_tds_season_avg': player_stats['passing_tds_season_avg'],
        'interceptions_season_avg': player_stats['interceptions_season_avg'],
        'rushing_yards_season_avg': player_stats['rushing_yards_season_avg'],
        'rushing_tds_season_avg': player_stats['rushing_tds_season_avg'],
        'receptions_season_avg': player_stats['receptions_season_avg'],
        'receiving_yards_season_avg': player_stats['receiving_yards_season_avg'],
        'receiving_tds_season_avg': player_stats['receiving_tds_season_avg'],
        'fumbles_lost_season_avg': player_stats['fumbles_lost_season_avg'],
        'opponent_defense_tier_num': tier_num,
        'is_home': is_home
    }])

    prediction = model.predict(feat_vec)[0]
    return f"Predicted fantasy points for {player_name} vs {opponent_team.upper()} ({home_away}): {prediction:.2f}"

# 12. Examples
print(predict_fantasy_points("A.Rodgers", "KC", "home"))
print(predict_fantasy_points("M.Lewis", "NYG", "away"))
