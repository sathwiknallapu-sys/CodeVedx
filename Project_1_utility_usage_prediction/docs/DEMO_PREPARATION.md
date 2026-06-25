# Demo Preparation Guide

This covers what to say and show for the LinkedIn demo video (1-3 minutes,
per the internship submission instructions) and for any live Q&A.

---

## 1. Two-Minute Presentation Script

> **[0:00 - 0:15] Hook + Introduction**
> "Hi, I'm [Your Name], and this is my first AI/ML internship project for
> CodeVedX — a Utility Usage Prediction Tool. It's a console-based machine
> learning app that predicts a household's monthly electricity usage and
> estimated bill, based on things like household size, AC usage, and season."

> **[0:15 - 0:35] The Problem**
> "Most people only find out their electricity usage after the bill arrives.
> There's no easy way to estimate it ahead of time, even though it depends
> on predictable factors — how many people live there, how much the AC runs,
> how hot it's been. So I built a tool that takes those inputs and gives an
> ML-based prediction."

> **[0:35 - 1:10] How It Works (show the terminal)**
> "The app has a simple menu — you can log new usage records, train the
> model, and predict usage for a new household. Let me show option 5."
> *(Type `5`, fill in sample values, show the prediction output.)*
> "Behind the scenes, this is a Random Forest Regressor trained on a 600-row
> dataset, with engineered features like area-per-person and appliance
> density that turned out to matter more than the raw counts alone."

> **[1:10 - 1:35] Results**
> "On held-out test data, the model gets an R² of about 0.88 — meaning it
> explains roughly 88% of the variation in actual usage. I also compared it
> against a simpler Linear Regression baseline for context."
> *(Show option 6 — last model performance.)*

> **[1:35 - 1:55] Engineering Quality**
> "The whole project is modular — separate files for configuration,
> validation, data handling, and the ML pipeline — and it's backed by 30
> automated unit tests covering both normal use and edge cases like invalid
> input or a missing model file."

> **[1:55 - 2:00] Close**
> "That's Project 1 — thanks for watching! Full code is on my GitHub, linked
> below. Open to feedback."

---

## 2. Demo Walkthrough Checklist

Record in this order for a clean, logical flow:

1. **Show the menu** (`python src/console_app.py`) — point out all 6 options.
2. **Option 1 — View records** — show a handful of real rows from the dataset.
3. **Option 4 — Train model** — run training live, narrate the MAE/RMSE/R² as they print.
4. **Option 5 — Predict usage** — enter realistic sample values, show the prediction + estimated bill.
5. **Option 6 — View last model performance** — confirms the saved metrics match.
6. *(Optional, if time allows)* **Option 2 — Add a record** — show a new entry being logged.
7. **Briefly show the code structure** (`ls src/` or your editor's file tree) to demonstrate modularity.
8. **Briefly show `pytest tests/ -v` passing** — strong proof of correctness for viewers with a technical background.

Keep total recording under 3 minutes — trim dead air between typed inputs in editing if needed.

---

## 3. Likely Questions & Suggested Answers

**Q: Why did you use a Random Forest instead of a neural network?**
A: For a tabular dataset of this size (600 rows, 8 features), a Random
Forest is a more appropriate choice — it handles non-linear interactions
well, doesn't need large amounts of data to generalize, and is faster to
train and easier to interpret than a neural network would be here.

**Q: Where did the dataset come from? Is it real data?**
A: It's a synthetic dataset I generated programmatically, since no real
consumption dataset was provided in the task brief. I deliberately added
random noise and realistic feature interactions (like AC usage scaling with
temperature) so the prediction problem isn't trivially easy.

**Q: How do you know the model isn't overfitting?**
A: I evaluate strictly on a held-out 20% test split that the model never
sees during training, and the R² of ~0.88 is a believable, moderate score
rather than a suspiciously perfect 0.99+, which would be a red flag for
data leakage.

**Q: What happens if a user enters invalid input, like text instead of a number?**
A: The validators module catches it and raises a custom `InvalidInputError`
with a clear message, and the console app catches that exception and
re-prompts rather than crashing.

**Q: What would you change if this dataset were 10x larger or real-world?**
A: I'd consider hyperparameter tuning via GridSearchCV, possibly migrating
from CSV to SQLite for performance, and re-validating that the engineered
features (area-per-person, appliance density) still hold up — real-world
data is noisier and might reveal different important features.

**Q: Why CSV instead of a database?**
A: For a console tool of this scope, CSV keeps the dependency footprint
small and the data human-readable/inspectable. I noted SQLite migration as
a future enhancement if the dataset grows significantly.

**Q: Did you consider deploying this as a web app?**
A: Yes — a Flask-based UI is the first future enhancement I'd build, and
the existing modular structure (`ml_pipeline.py`, `data_handler.py`) was
designed so it could be reused directly behind a Flask API without much
rewriting.

**Q: How did you validate the feature engineering choices (area_per_person, appliance_density)?**
A: I compared model performance with and without these derived features
during development, and confirmed directionally that they improved fit —
this matched the general pattern from household energy literature that
per-capita features tend to carry more signal than raw counts.

---

## 4. Tips for Recording

- Use a clean, fixed-width terminal font and increase font size before recording — small terminal text is hard to read on LinkedIn video.
- Practice the typed inputs once beforehand so the demo segment isn't slow or fumbling.
- If recording the screen, hide unrelated desktop clutter, notifications, and other open windows.
- Add captions or on-screen text for the key metric (R² ≈ 0.88) since some viewers will watch without sound.
- End with a clear call-to-action: GitHub link in the video description, `#codevedx #internship #AIML` in the post.
