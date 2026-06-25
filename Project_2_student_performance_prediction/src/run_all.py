"""
run_all.py
-----------
Single-command entry point that runs the entire project pipeline
from scratch:

    Step 1: Generate the dataset
    Step 2: Clean and validate the data
    Step 3: Generate EDA charts
    Step 4: Train and compare models, save the best ones
    Step 5: Run the evaluation report
    Step 6: Run the automated test suite
    Step 7: Launch the console UI

Usage:
    cd src
    python run_all.py            # run everything then open the UI
    python run_all.py --no-ui    # run pipeline steps only (useful in CI/CD)
    python run_all.py --skip-train   # skip dataset gen + training if already done

This script is the recommended way for a first-time user to get
the project running end to end without having to know which module
to run first.
"""

import os
import sys
import time
import argparse
import subprocess
import unittest

# Ensure imports resolve from src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config


def banner(text):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def step(number, description):
    print(f"\n[Step {number}] {description}")
    print("-" * 50)


def success(msg):
    print(f"  [OK]  {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def run_step_1_generate_dataset():
    """Generates the synthetic student dataset."""
    step(1, "Generating synthetic student dataset")
    from generate_dataset import generate_student_data
    import pandas as pd

    if os.path.exists(config.RAW_DATA_PATH):
        print("  Dataset already exists. Regenerating...")

    df = generate_student_data()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    success(f"Generated {len(df)} records -> {config.RAW_DATA_PATH}")
    success(f"Class distribution: {df['performance_category'].value_counts().to_dict()}")


def run_step_2_clean_data():
    """Cleans and validates the raw dataset."""
    step(2, "Cleaning and validating data")
    from data_preprocessing import clean_pipeline
    df = clean_pipeline()
    success(f"Clean dataset saved -> {config.CLEAN_DATA_PATH}")
    success(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns, zero missing values")


def run_step_3_generate_charts():
    """Generates EDA visualisation charts."""
    step(3, "Generating EDA charts")
    from visualize import generate_all_charts
    generate_all_charts()
    charts_dir = os.path.join(config.BASE_DIR, "docs", "charts")
    chart_files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
    success(f"Generated {len(chart_files)} chart(s) -> docs/charts/")


def run_step_4_train_models():
    """Trains and evaluates all candidate models."""
    step(4, "Training and comparing ML models")
    from model_training import run_training_pipeline
    metadata = run_training_pipeline()
    best_reg = metadata["regression"]["best_model"]
    best_clf = metadata["classification"]["best_model"]
    reg_r2 = metadata["regression"]["metrics"][best_reg]["r2"]
    clf_f1 = metadata["classification"]["metrics"][best_clf]["f1"]
    success(f"Best regressor   : {best_reg} (R² = {reg_r2})")
    success(f"Best classifier  : {best_clf} (F1 = {clf_f1})")
    success("All artifacts saved to models/")


def run_step_5_evaluate():
    """Generates the detailed evaluation report."""
    step(5, "Generating evaluation report")
    from evaluate import generate_report
    generate_report()


def run_step_6_tests():
    """Runs the automated test suite."""
    step(6, "Running automated test suite")
    test_dir = os.path.join(config.BASE_DIR, "tests")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    if result.wasSuccessful():
        success(f"All {result.testsRun} tests passed.")
        return True
    else:
        fail(f"{len(result.failures)} failure(s), {len(result.errors)} error(s) in {result.testsRun} tests.")
        for test, traceback in result.failures + result.errors:
            print(f"\n  FAILED: {test}")
            print(f"  {traceback.splitlines()[-1]}")
        return False


def run_step_7_launch_ui():
    """Launches the interactive console UI."""
    step(7, "Launching console UI")
    print("  Starting the Student Performance Prediction System...")
    print("  (Ctrl+C to exit)\n")
    time.sleep(0.8)
    import main
    main.main()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Student Performance Prediction System - Full Pipeline Runner"
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run pipeline steps only, skip launching the interactive UI",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip dataset generation and model training (assume already done)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the automated test suite",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    banner("STUDENT PERFORMANCE PREDICTION SYSTEM - FULL PIPELINE")
    print(f"  Project: CodeVedX AI/ML Internship - Project 2")
    print(f"  Base directory: {config.BASE_DIR}")

    start = time.time()

    if not args.skip_train:
        run_step_1_generate_dataset()
        run_step_2_clean_data()
        run_step_3_generate_charts()
        run_step_4_train_models()
    else:
        print("\n  [--skip-train] Skipping dataset generation and model training.")

    run_step_5_evaluate()

    tests_passed = True
    if not args.skip_tests:
        tests_passed = run_step_6_tests()
    else:
        print("\n  [--skip-tests] Skipping test suite.")

    elapsed = time.time() - start
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")

    if not tests_passed:
        print("  WARNING: Some tests failed. Review output above before submitting.\n")
    else:
        print("  All pipeline steps completed successfully.")
        print("  Ready for GitHub push and LinkedIn demo.\n")

    if not args.no_ui:
        run_step_7_launch_ui()
    else:
        print("  [--no-ui] Skipping interactive UI launch.")
        print(f"  To start the UI manually: cd src && python main.py")
        print(f"  To start the API:          cd src && python app.py\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. Goodbye!")
        sys.exit(0)
