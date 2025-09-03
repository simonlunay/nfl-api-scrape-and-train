# NFL Skill Player Fantasy Predictor

A machine learning project that predicts fantasy points and individual stat lines for NFL skill players (QB, RB, WR, TE). The model predicts stats such as **passing yards, passing TDs, interceptions, rushing yards, rushing TDs, receptions, receiving yards, receiving TDs, fumbles lost**, and calculates **fantasy points** for upcoming games.  

This project uses **pandas**, **scikit-learn**, and weekly NFL data to preprocess stats, create features, train models, and make predictions.

---

## Project Overview

The workflow consists of four main scripts:

1. **`train_nfl_multioutput_models.py`**  
   - Loads weekly player stats with opponent info (`nfl_skill_players_2023_weekly_with_opponent.json`).  
   - Maps opponent defenses to tiers (`top`, `mid`, `bottom`).  
   - Calculates season averages for all relevant stats per player.  
   - Encodes categorical features such as opponent defense tier and home/away.  
   - Splits data by position: QB vs skill players (RB, WR, TE).  
   - Trains **Random Forest MultiOutputRegressor** models separately for QBs and skill players.  
   - Saves trained models (`qb_multi_model.joblib` and `skill_multi_model.joblib`) and helper data (`helper_data.joblib`).

2. **`train_nfl_fantasy_model_single_output.py`**  
   - Loads the same weekly player dataset.  
   - Cleans and converts all stat columns to numeric.  
   - Calculates fantasy points per game using standard scoring rules.  
   - Computes season averages per player and encodes opponent defense tiers numerically.  
   - Trains a **Random Forest Regressor** to predict fantasy points as a single output.  
   - Evaluates model performance with **Mean Absolute Error (MAE)**.  
   - Provides a function to predict fantasy points for a player given opponent and home/away.

3. **`predict_fantasy_points.py`**  
   - Loads the saved QB and skill player multi-output models along with helper data.  
   - Formats player names to a short version (e.g., `A.Rodgers`) for consistency.  
   - Uses the most recent game stats and opponent defense tier to create input features.  
   - Predicts fantasy points for QBs or skill players using the corresponding model.  
   - Returns a formatted string with the predicted fantasy points.

4. **`helper_data.json / helper_data.joblib`**  
   - Stores mappings for defense tiers, feature columns, target stats, and numeric column order.  
   - Ensures consistency when preparing new player data for prediction.

---

## Features & Targets

### Features:
- Player season averages for all relevant stats (passing, rushing, receiving, turnovers)
- Opponent defense tier (numeric)
- Home/Away indicator

### Targets:
- For multi-output models: individual stat lines (passing yards, passing TDs, interceptions, rushing yards, rushing TDs, receptions, receiving yards, receiving TDs, fumbles lost)  
- For single-output model: fantasy points

---

## Usage

I used this in my fantasy sport predictor.
