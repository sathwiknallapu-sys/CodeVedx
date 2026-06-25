# Student Performance Prediction System

A machine learning system that predicts a student's final academic marks and performance category (At Risk / Average / Good / Excellent) from behavioral and academic indicators such as attendance, study habits, and past performance.

Built as **Project 2** of the CodeVedX AI/ML Internship Program.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Machine Learning Approach](#machine-learning-approach)
- [Model Performance](#model-performance)
- [Testing](#testing)
- [Screenshots to Capture](#screenshots-to-capture)
- [Future Enhancements](#future-enhancements)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Educational institutions often identify struggling students too late — usually only after final exam results are published, when intervention is no longer possible. Teachers and academic coordinators rarely have a quantitative, early-warning tool that combines multiple weak signals (attendance, study habits, assignment completion, sleep, support systems) into one actionable prediction.

This project builds a system that takes a student's mid-semester indicators and predicts:
1. Their likely **final marks** (a continuous score out of 100), and
2. Their **performance category**, so 'At Risk' students can be flagged for early academic support.

## Objectives

- Build an end-to-end ML pipeline: data generation → cleaning → feature engineering → model training → evaluation → deployment in a usable interface.
- Compare multiple algorithms for both regression and classification, and automatically select the best-performing one.
- Engineer meaningful derived features (not just feed raw columns into a model) to demonstrate genuine ML reasoning.
- Provide a robust, user-friendly console interface with input validation and batch-processing support.
- Document the system to a professional, presentable standard.

## Scope

**In scope:** synthetic-but-realistic tabular dataset, regression + classification models, console UI, batch CSV prediction, prediction history logging, automated tests, EDA visualizations.

**Out of scope:** live integration with a school's student information system, a web/mobile front-end (a console interface was chosen — see *Project Architecture*), real student PII (synthetic data is used deliberately, for privacy reasons explained in `src/generate_dataset.py`).

## Expected Output

Given inputs like attendance %, daily study hours, previous semester marks, assignments completed, extracurricular hours, sleep hours, parental support level, and internet access — the system outputs:

```
Predicted Final Marks   : 77.8 / 100
Performance Category    : Good
Confidence              : 77.0%

Category Probability Breakdown:
  Good      :  77.0%  ###############
  Excellent :  13.2%  ##
  Average   :   9.7%  #
  At Risk   :   0.0%
```

---

## Project Architecture

### System Design

The system follows a classic **modular ML pipeline** architecture, where each stage is an independently runnable, independently testable Python module:

```
 Raw CSV Data
      |
      v
[data_preprocessing.py]  --> cleans, validates, imputes missing values
      |
      v
[feature_engineering.py] --> derives study_attendance_ratio, effort_index
      |
      v
[model_training.py]      --> trains + compares models, saves best ones
      |
      v
   models/*.pkl  (persisted regressor, classifier, scaler, encoder)
      |
      v
[predictor.py]            --> loads models, validates input, predicts
      |
      v
[main.py]                 --> console UI (single / batch prediction, history, model summary)
```

### Workflow Diagram Description

1. **Input stage** — A user enters a student's data through the console menu (or supplies a CSV for batch mode).
2. **Validation stage** — Every field is checked for type and valid range before it ever reaches a model. Invalid input is rejected with a specific, actionable error message (not a stack trace).
3. **Feature engineering stage** — The same transformation used during training (`engineer_features()`) is applied identically at prediction time, eliminating training/serving skew.
4. **Scaling stage** — Features are standardized using the *same* `StandardScaler` fitted during training (loaded from disk), ensuring consistent input distribution for the models.
5. **Prediction stage** — Two models run in parallel: a regressor outputs exact marks, a classifier outputs category + confidence probabilities for all four classes.
6. **Output stage** — Results are displayed with a visual probability bar chart in the console, and logged to `prediction_history.csv` for later review.

### Folder Structure

```
student-performance-prediction/
├── data/
│   ├── student_performance.csv          # raw synthetic dataset
│   ├── student_performance_clean.csv    # cleaned dataset (generated)
│   └── prediction_history.csv           # log of predictions made via the UI (generated)
├── models/
│   ├── marks_regressor.pkl              # best regression model (generated)
│   ├── performance_classifier.pkl       # best classification model (generated)
│   ├── feature_scaler.pkl               # fitted StandardScaler (generated)
│   ├── label_encoder.pkl                # fitted LabelEncoder (generated)
│   └── metadata.json                    # model comparison metrics (generated)
├── src/
│   ├── config.py                        # central paths & constants
│   ├── exceptions.py                    # custom exception classes
│   ├── generate_dataset.py              # synthetic dataset generator
│   ├── data_preprocessing.py            # cleaning, validation, EDA summary
│   ├── feature_engineering.py           # derived feature creation
│   ├── model_training.py                # trains & compares ML models
│   ├── visualize.py                     # generates EDA charts (docs/charts/)
│   ├── predictor.py                     # loads models, serves predictions
│   └── main.py                          # console UI entry point
├── tests/
│   └── test_project.py                  # automated unit tests (17 tests)
├── docs/
│   ├── charts/                          # generated EDA visualizations
│   └── project_report.docx              # full project report
├── screenshots/                         # for demo screenshots (see below)
├── requirements.txt
├── .gitignore
└── README.md
```

### Required Libraries

| Library | Purpose |
|---|---|
| `pandas` | data loading, cleaning, manipulation |
| `numpy` | numerical operations |
| `scikit-learn` | ML models, scaling, encoding, metrics, train/test split |
| `matplotlib` / `seaborn` | EDA visualizations |
| `joblib` | model serialization (save/load `.pkl` files) |

Install everything with:
```bash
pip install -r requirements.txt
```

---

## Dataset

Real student records cannot be ethically published (privacy/FERPA-style concerns), so this project uses a **synthetic dataset that mimics realistic statistical relationships** — generated by `src/generate_dataset.py`. The target (`final_marks`) is built as a weighted, noisy function of the input features, so the relationships a model "discovers" mirror genuine educational research findings (e.g. study time and attendance matter more than extracurricular load).

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `student_id` | string | Unique student identifier (e.g. `STU1000`) |
| `attendance_percentage` | float (0–100) | Percentage of classes attended |
| `study_hours_per_day` | float (0–24) | Average self-study hours per day |
| `previous_sem_marks` | float (0–100) | Marks scored in the previous semester |
| `assignments_completed` | int (0–10) | Number of assignments completed out of 10 |
| `extracurricular_hours` | float | Hours per week spent on extracurricular activities |
| `sleep_hours` | float (0–24) | Average sleep hours per night |
| `parental_support` | int (1–5) | Self-reported parental support level (1=low, 5=high) |
| `internet_access` | int (0/1) | Whether the student has internet access at home |
| `final_marks` | float (0–100) | **Target (regression)** — final semester marks |
| `performance_category` | string | **Target (classification)** — derived from `final_marks`: At Risk (<50), Average (50–69), Good (70–84), Excellent (85+) |

The raw dataset (500 records) intentionally includes ~3% missing values in three columns to let the cleaning pipeline demonstrate real-world data-quality handling.

### Regenerating the Dataset

```bash
cd src
python generate_dataset.py
```

---

## Installation Guide

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CODEVEDX.git
cd CODEVEDX/student-performance-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset
cd src
python generate_dataset.py

# 5. Run the full training pipeline (cleans data, engineers features, trains & saves models)
python model_training.py

# 6. (Optional) Generate EDA charts
python visualize.py

# 7. Launch the console application
python main.py
```

---

## Usage Guide

Running `python main.py` opens an interactive menu:

```
MAIN MENU
  1. Predict performance for a new student
  2. Batch predict from a CSV file
  3. View prediction history
  4. View model performance summary
  5. Exit
```

### Option 1 — Single Prediction
You'll be prompted for each feature one at a time, with input validation. Invalid entries (wrong type or out-of-range) trigger a re-prompt instead of crashing the program.

### Option 2 — Batch Prediction
Point the system at a CSV file containing the required feature columns (a `student_id` column is optional). Each row is predicted independently; invalid rows are skipped with a clear reason, and a summary count is shown at the end.

### Option 3 — Prediction History
Displays the most recent predictions logged to `data/prediction_history.csv`.

### Option 4 — Model Performance Summary
Displays the saved metrics (MAE, RMSE, R² for regression; Accuracy, Precision, Recall, F1 for classification) for every model that was compared during training.

---

## Machine Learning Approach

### Data Preprocessing
- **Missing value imputation:** median imputation (robust to outliers/skew) for numeric columns.
- **Duplicate removal:** based on `student_id`.
- **Range validation:** every numeric column is checked against a physically sensible range (e.g. attendance must be 0–100) before training or prediction; violations raise a custom `DataValidationError`.

### Feature Engineering
Two derived features were created specifically because raw columns alone underrepresent how these factors interact in real life:

- **`study_attendance_ratio`** = `study_hours_per_day × attendance_percentage / 100` — captures combined engagement (a student who studies a lot but skips class, or attends but never studies, scores lower on this than one who does both).
- **`effort_index`** = a weighted 0–10 composite of normalized study hours (45%), attendance (30%), and assignment completion (25%) — a single "overall effort" signal.

In testing, **both engineered features correlated more strongly with final marks (0.59 and 0.64) than any single raw feature (best raw feature: 0.55)** — and a Random Forest feature-importance analysis confirmed `effort_index` as the single most predictive feature in the dataset. This is direct evidence the feature engineering step adds real signal, not just complexity.

### Model Selection
Two models were trained and compared for each task; the better performer (by validation metric) is automatically saved:

| Task | Candidate Models | Selection Metric |
|---|---|---|
| Regression (exact marks) | Linear Regression, Random Forest Regressor | R² (higher is better) |
| Classification (category) | Logistic Regression, Random Forest Classifier | Weighted F1-score |

### Model Training & Evaluation
- 80/20 train-test split (`random_state=42` for reproducibility).
- Features standardized with `StandardScaler` (fitted on training data only, to avoid data leakage).
- Classification labels encoded with `LabelEncoder`.

### Model Saving/Loading
All artifacts (`regressor`, `classifier`, `scaler`, `label_encoder`) are persisted with `joblib` into `models/`, plus a `metadata.json` summarizing every metric for transparency. `predictor.py` loads these once at startup and reuses them for all subsequent predictions — no retraining needed at inference time.

---

## Model Performance

*(Actual numbers from the most recent training run — see `models/metadata.json` for the live values.)*

**Regression (predicting exact final marks):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (selected) | 5.51 | 7.12 | 0.567 |
| Random Forest Regressor | 5.83 | 7.66 | 0.498 |

**Classification (predicting performance category):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (selected) | 0.730 | 0.727 | 0.730 | 0.707 |
| Random Forest Classifier | 0.730 | 0.719 | 0.730 | 0.698 |

These are deliberately realistic (not artificially perfect) numbers — the dataset includes genuine noise, exactly as real-world student behavior data would, so an R² near 1.0 or 99% accuracy would actually be a red flag for overfitting or data leakage rather than a sign of a better project.

---

## Testing

Run the automated test suite (17 tests covering preprocessing, feature engineering, and prediction):

```bash
cd tests
python test_project.py
# or, if pytest is installed:
python -m pytest test_project.py -v
```

### Test Coverage
- Missing value imputation correctness (median-based)
- Data range validation (accepts valid, rejects out-of-range)
- Duplicate record removal
- Missing-file error handling
- Feature engineering formula correctness and bounds
- End-to-end prediction returns all expected output fields
- Predicted marks always fall within [0, 100]
- Predicted category is always one of the four valid labels
- Input validation rejects missing fields, out-of-range values, and wrong types
- **Sanity/regression test:** a high-effort student profile always predicts higher marks than a low-effort one (guards against silent model degradation on retraining)

### Sample Manual Test Cases

| Input Scenario | Expected Behavior |
|---|---|
| All valid, high-effort student (90% attendance, 5 study hrs, 85 prev marks) | Predicts "Good" or "Excellent" with high confidence |
| All valid, low-effort student (45% attendance, 0.5 study hrs, 35 prev marks) | Predicts "At Risk", shows academic support note |
| `attendance_percentage = 150` | Rejected: "outside valid range [0, 100]" |
| `study_hours_per_day = "abc"` | Rejected: "must be numeric" |
| Missing field in input dict | Rejected: "Missing required field" |
| CSV batch with 1 bad row among 4 | 3 processed successfully, 1 skipped with reason, summary printed |
| Missing CSV file path | Graceful "File not found" message, no crash |
| Models not yet trained, app launched | Graceful error directing user to run `model_training.py` first |

---

## Screenshots to Capture

For the LinkedIn demo video and submission, capture:
1. The main menu on launch.
2. A single prediction walkthrough (input → result with probability bars).
3. The "At Risk" academic support note triggering for a low-effort profile.
4. Batch prediction processing a CSV with a mix of valid/invalid rows.
5. The model performance summary screen.
6. One or two EDA charts from `docs/charts/` (e.g. `feature_importance.png`, `study_hours_vs_marks.png`).
7. The terminal running `python test_project.py` showing all tests passing.

---

## Future Enhancements

- Migrate the console UI to a Flask web dashboard with charts rendered in-browser.
- Add a time-series view to track a student's predicted trajectory across multiple checkpoints in a semester.
- Support multi-class SHAP-based explainability per individual prediction ("why was this student flagged At Risk?").
- Add authentication and per-teacher dashboards for real classroom deployment.
- Replace synthetic data with anonymized real institutional data (with proper consent/IRB approval) for production use.
- Hyperparameter tuning via `GridSearchCV`/`RandomizedSearchCV` for further accuracy gains.

---

## Tech Stack

`Python 3` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `joblib` · `unittest`

---

## Author

Built as part of the **CodeVedX AI/ML Internship Program** — Project 2: Student Performance Prediction System.

## License

This project is submitted for educational purposes as part of an internship program.
