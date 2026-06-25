# Utility Usage Prediction Tool

A console-based Machine Learning application that predicts a household's
monthly electricity usage (in units) and estimated bill amount, based on
factors like household size, home area, AC usage, number of appliances,
season, and previous month's consumption.

Built as **Project 1** for the CodeVedX AI/ML Internship Program.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Dataset](#dataset)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Problem Statement

Households often have no early warning of what their electricity bill will
look like until the meter reading / bill actually arrives. Factors like
how long the AC ran, how many appliances are in use, the season, and home
size all affect consumption, but most people don't have an easy way to
estimate this in advance.

This tool addresses that by letting a user input these factors and get an
ML-based prediction of expected unit consumption and bill amount, while
also functioning as a simple record-keeping system for past usage data.

## Features

- Menu-driven console interface (no GUI dependencies, runs anywhere Python runs)
- Add new household usage records to the dataset (CSV-backed)
- View and update existing records
- Train a regression model on demand (Random Forest or Linear Regression)
- Predict usage + estimated bill for a new household profile
- View saved model performance metrics (MAE, RMSE, R²)
- Input validation on every field with clear error messages
- Exception handling throughout — the app does not crash on bad input or missing files

## Project Structure

```
CodeVedX/
├── .vscode/                        # VS Code launch/test/settings config
│   ├── launch.json
│   ├── settings.json
│   └── extensions.json
├── .env                            # sets PYTHONPATH=./src for imports & tests
├── data/
│   ├── generate_dataset.py        # one-off script used to create the dataset
│   └── utility_usage_data.csv     # 600-row synthetic household dataset
├── models/
│   ├── utility_model.pkl          # trained model (created after first training run)
│   ├── encoders.pkl               # saved label encoders for categorical fields
│   └── metrics.json               # last training run's evaluation metrics
├── src/
│   ├── config.py                  # file paths & constants
│   ├── exceptions.py              # custom exception classes
│   ├── validators.py              # input validation helpers
│   ├── data_handler.py            # CSV read/write/update logic
│   ├── ml_pipeline.py             # preprocessing, training, evaluation, prediction
│   └── console_app.py             # main entry point / menu UI
├── tests/
│   ├── conftest.py
│   ├── test_validators.py
│   ├── test_data_handler.py
│   └── test_ml_pipeline.py
├── docs/                          # report, presentation, diagrams (see /docs)
├── screenshots/                  # placeholder folder for demo screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

### Why this structure

Each file in `src/` has a single responsibility — config in one place,
validation logic separate from the CSV logic, ML logic separate from the
console UI. This keeps any one file small and easy to review, and matches
how the internship task plan asked for modular programming with separate
files where appropriate.

## Tech Stack

| Component        | Choice                          |
|-------------------|----------------------------------|
| Language          | Python 3.10+                    |
| ML Library        | scikit-learn (RandomForestRegressor, LinearRegression) |
| Data handling     | Built-in `csv` module (no pandas dependency for the core app) |
| Model persistence | `pickle`                        |
| Testing           | `pytest`                        |
| Interface         | Console (menu-driven)           |

> Note: the dataset is intentionally handled with Python's built-in `csv`
> module rather than pandas, to keep the runtime dependency footprint small
> for a console tool. `numpy` is used internally by scikit-learn.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/CodeVedX.git
cd CodeVedX
```

2. (Recommended) create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) regenerate the dataset from scratch:
```bash
python data/generate_dataset.py
```
This step is optional — `data/utility_usage_data.csv` is already included
in the repo with 600 sample records.

## Opening in VS Code

This project includes a `.vscode/` folder with ready-made configuration:

1. Open the **`CodeVedX`** folder itself in VS Code (`File > Open Folder...`) —
   not a parent folder — so the settings below apply correctly.
2. Install the recommended extensions when prompted (Python + Pylance +
   debugpy), or manually install **ms-python.python**.
3. Select a Python interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter*
   → choose your virtual environment (or system Python 3.10+) after running
   step 3 of Installation above.
4. **Run the app**: open the *Run and Debug* panel (`Ctrl+Shift+D`) and choose
   **"Run Console App"** from the dropdown, then press the green play button.
   This runs `src/console_app.py` with the correct working directory already
   configured.
5. **Run the tests**: open the *Testing* panel (the flask icon in the sidebar)
   — VS Code will auto-discover all tests in `tests/` via the bundled pytest
   settings. Click the play button at the top to run them all, or run
   individually per test.
6. Alternatively, use the integrated terminal directly:
   ```bash
   python src/console_app.py        # run the app
   python -m pytest tests/ -v       # run the tests
   ```

> The included `.env` file sets `PYTHONPATH=./src` so both the test runner
> and Pylance's import resolution understand `from config import ...`-style
> imports used throughout the `src/` modules, without needing a package
> install or `sys.path` hacks.

## Usage Guide

Run the application from the project root:

```bash
python src/console_app.py
```

You'll see a menu like this:

```
==================================================
   UTILITY USAGE PREDICTION TOOL  (CodeVedX AI/ML)
==================================================
  1. View usage records
  2. Add a new usage record
  3. Update an existing record
  4. Train / retrain prediction model
  5. Predict usage for a household
  6. View last model performance
  0. Exit
==================================================
```

### Typical first run

1. Choose **option 4** to train the model for the first time (this reads
   `data/utility_usage_data.csv` and saves a trained model to `models/`).
2. Choose **option 5** to predict usage for a new household — you'll be
   asked for household size, home area, home type, season, average
   temperature, AC usage hours, number of appliances, and previous month's
   units.
3. Choose **option 2** any time you want to log an actual real-world
   reading into the dataset for future retraining.

### Example session (option 5 — Predict)

```
> Household size (1-10): 4
> Home area in sqft (200-6000): 1500
> Home type ['Apartment', 'Independent House', 'Villa']: Apartment
> Season ['Summer', 'Winter', 'Monsoon', 'Spring']: Summer
> Average temperature in Celsius (-5 to 50): 37
> Daily AC usage hours (0-24): 6
> Number of major appliances (1-30): 10
> Previous month's units consumed (0-2000): 300

Predicted electricity usage: ~231.6 units this month.
Estimated bill amount: ~Rs. 1255.4
```

## Dataset

`data/utility_usage_data.csv` contains 600 synthetic but realistic household
records. It was generated with `data/generate_dataset.py` using a formula
that combines household size, area, AC usage, and season with random noise
— so the relationships are realistic (not a perfectly clean linear formula),
similar to what you'd see in a real utility billing dataset.

| Column             | Type    | Description                                              |
|---------------------|---------|------------------------------------------------------------|
| `record_id`         | int     | Unique row identifier                                     |
| `household_size`    | int     | Number of people living in the household (1-7)            |
| `home_area_sqft`    | int     | Home area in square feet                                   |
| `home_type`         | string  | Apartment / Independent House / Villa                      |
| `season`            | string  | Summer / Winter / Monsoon / Spring                          |
| `avg_temp_c`        | float   | Average outdoor temperature for that month (°C)            |
| `ac_usage_hours`    | float   | Average daily air-conditioner usage (hours)                |
| `num_appliances`    | int     | Count of major household appliances in use                  |
| `prev_month_units`  | float   | Units consumed in the previous billing month                |
| `units_consumed`    | float   | **Target** — actual electricity units consumed this month   |
| `bill_amount`       | float   | Actual bill amount (₹), derived from `units_consumed` via slab rates |

New records added through the console app (option 2) are appended to this
same file, so the dataset can grow over time as real usage gets logged.

## Machine Learning Approach

1. **Preprocessing** — numeric fields are cast to float; categorical fields
   (`home_type`, `season`) are label-encoded.
2. **Feature engineering** — two derived features are added:
   - `area_per_person` = `home_area_sqft / household_size`
   - `appliance_density` = `num_appliances / household_size`
   These capture how "spread out" the consumption drivers are per person,
   which turned out to help more than the raw counts alone during testing.
3. **Train/test split** — 80/20 split, `random_state=42` for reproducibility.
4. **Model selection** — the console app supports two model types:
   - `RandomForestRegressor` (default — handles non-linear interactions
     between AC usage and temperature well)
   - `LinearRegression` (simpler baseline, included for comparison)
5. **Evaluation metrics** — MAE, RMSE, and R² are computed on the held-out
   test set after every training run and saved to `models/metrics.json`.
6. **Persistence** — the trained model and fitted label encoders are
   pickled to `models/utility_model.pkl` and `models/encoders.pkl` so
   predictions don't require retraining every time the app starts.

## Model Performance

Latest training run (Random Forest, 480 train / 120 test rows):

| Metric    | Value   |
|-----------|---------|
| MAE       | ~17.1 units |
| RMSE      | ~21.2 units |
| R² Score  | ~0.88   |

Linear Regression was also tested for comparison and scored slightly
better on this dataset (R² ~0.89), which makes sense since the underlying
formula used to generate the synthetic data is mostly additive. Random
Forest is kept as the default since it's more robust to noisier real-world
data and doesn't assume a linear relationship.

These numbers will vary slightly each time the dataset is regenerated or
the model is retrained, since there's randomness in both the synthetic
data and the train/test split.

## Testing

Unit tests are written with `pytest` and cover:
- Input validators (range checks, type checks, invalid categories)
- Dataset read/write/update logic (using a temporary copy of the CSV so
  tests never modify the real data file)
- ML pipeline (training produces expected metrics, predictions are sane,
  unseen categories don't crash the pipeline)

Run all tests from the project root:

```bash
python -m pytest tests/ -v
```

Expected output: all tests pass (30 tests at time of writing).

### Sample test scenarios

| Scenario                                   | Expected Result                          |
|---------------------------------------------|-------------------------------------------|
| Household size = 4 (valid)                  | Accepted                                   |
| Household size = "abc"                      | `InvalidInputError` — must be a whole number |
| Household size = 0                          | `InvalidInputError` — out of range (1-10)  |
| Home type = "Treehouse"                      | `InvalidInputError` — not a valid option   |
| Predict before training a model              | `ModelNotTrainedError` with a clear message |
| Dataset file missing                         | `DatasetError` with the expected file path |
| Unseen category at prediction time           | Falls back gracefully, does not crash      |

## Screenshots

> Capture these while demoing the app and place them in `/screenshots`:
1. Main menu on launch
2. "View usage records" output table
3. "Train model" output showing MAE / RMSE / R²
4. "Predict usage" — input prompts + final prediction output
5. A terminal run of `pytest tests/ -v` showing all tests passing

## Future Enhancements

- Add a Flask-based web UI as an alternative to the console interface
- Add data visualization (matplotlib/seaborn charts of usage trends over time)
- Support multiple regions with different electricity slab rates
- Add a "what-if" comparison mode (e.g., compare bill with/without AC)
- Hyperparameter tuning via `GridSearchCV` for the Random Forest model
- Containerize with Docker for easier deployment
- Migrate storage from CSV to a lightweight database (SQLite) as the dataset grows

## Author

Developed as part of the **CodeVedX Artificial Intelligence & Machine
Learning (AI/ML) Internship Program** — Project 1: Utility Usage
Prediction Tool.

Repository name (per task instructions): `CODEVEDX`
