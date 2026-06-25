"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset for the Student Performance
Prediction System.

Why synthetic data?
A real student dataset usually cannot be shared publicly because of
privacy (FERPA / institutional policy). For an internship-grade project
it is standard and acceptable practice to generate a synthetic dataset
that mimics real-world statistical relationships (e.g. more study hours
and higher attendance generally lead to better marks, but with natural
noise and some exceptions, just like real life).

Run:
    python generate_dataset.py

Output:
    ../data/student_performance.csv
"""

import numpy as np
import pandas as pd
import os

# Reproducibility - same "random" data every time the script is run
np.random.seed(42)

N_STUDENTS = 500


def generate_student_data(n=N_STUDENTS):
    """Builds a DataFrame of synthetic but realistic student records."""

    student_ids = [f"STU{1000 + i}" for i in range(n)]

    # --- Core behavioural / academic features -----------------------
    attendance_percentage = np.clip(np.random.normal(78, 12, n), 35, 100)
    study_hours_per_day = np.clip(np.random.normal(3.2, 1.4, n), 0, 9)
    previous_sem_marks = np.clip(np.random.normal(65, 14, n), 20, 100)

    # Assignments completed out of 10
    assignments_completed = np.clip(
        np.random.normal(7.5, 2.0, n), 0, 10
    ).round().astype(int)

    # Extra-curricular involvement: hours per week (mild negative effect
    # past a certain point, mild positive effect on well-roundedness)
    extracurricular_hours = np.clip(np.random.exponential(2.5, n), 0, 15)

    # Sleep hours per night
    sleep_hours = np.clip(np.random.normal(6.7, 1.1, n), 3, 10)

    # Parental support score, 1 (low) - 5 (high), categorical-ish
    parental_support = np.random.choice(
        [1, 2, 3, 4, 5], size=n, p=[0.05, 0.12, 0.28, 0.33, 0.22]
    )

    # Internet access at home: affects ability to study online resources
    internet_access = np.random.choice([0, 1], size=n, p=[0.18, 0.82])

    # --- Target construction ------------------------------------------
    # Build final marks as a weighted, noisy function of the features
    # above. Weights reflect plausible real-world influence.
    noise = np.random.normal(0, 7, n)

    final_marks = (
        0.25 * attendance_percentage
        + 5.0 * study_hours_per_day
        + 0.32 * previous_sem_marks
        + 1.5 * assignments_completed
        + 1.2 * parental_support
        + 1.0 * sleep_hours
        - 0.4 * np.maximum(extracurricular_hours - 6, 0)  # too much = penalty
        + 1.8 * internet_access
        + noise
        - 18  # centering constant so the mean lands around 65-70
    )

    final_marks = np.clip(final_marks, 0, 100)

    # --- Convert numeric marks into performance categories ------------
    def categorize(mark):
        if mark >= 85:
            return "Excellent"
        elif mark >= 70:
            return "Good"
        elif mark >= 50:
            return "Average"
        else:
            return "At Risk"

    performance_category = [categorize(m) for m in final_marks]

    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance_percentage": attendance_percentage.round(1),
        "study_hours_per_day": study_hours_per_day.round(1),
        "previous_sem_marks": previous_sem_marks.round(1),
        "assignments_completed": assignments_completed,
        "extracurricular_hours": extracurricular_hours.round(1),
        "sleep_hours": sleep_hours.round(1),
        "parental_support": parental_support,
        "internet_access": internet_access,
        "final_marks": final_marks.round(1),
        "performance_category": performance_category,
    })

    # Inject a small amount of realistic missingness (real-world data
    # is never perfectly clean — this also lets us demonstrate the
    # data-cleaning step the internship explicitly asks for)
    for col in ["attendance_percentage", "study_hours_per_day", "sleep_hours"]:
        missing_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_student_data()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "student_performance.csv")

    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['performance_category'].value_counts()}")
