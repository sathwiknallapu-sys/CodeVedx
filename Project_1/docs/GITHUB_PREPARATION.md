# GitHub Preparation Guide

This document covers how to publish this project to the `CODEVEDX` repository,
per the internship task instructions.

## 1. Repository Setup

```bash
# from inside the CodeVedX project folder
git init
git add .
git commit -m "Initial commit: project scaffold, dataset, and README"
git branch -M main
git remote add origin https://github.com/<your-username>/CODEVEDX.git
git push -u origin main
```

> The task brief asks for the repository to be named **CODEVEDX** and to contain
> all internship tasks (not just this one) — if you're adding this as Project 1
> inside an existing CODEVEDX repo, place these files under a subfolder such as
> `project-1-utility-usage-prediction/` instead of the repo root.

## 2. Suggested Repository Structure (multi-project repo)

```
CODEVEDX/
├── project-1-utility-usage-prediction/   <- this project
│   ├── data/
│   ├── docs/
│   ├── models/
│   ├── src/
│   ├── tests/
│   ├── README.md
│   └── requirements.txt
├── project-2-student-performance/
├── project-3-fake-news-detection/
├── project-4-helpdesk-chatbot/
├── project-5-recommendation-engine/
└── README.md   <- top-level repo overview linking to each project
```

## 3. Commit Message Conventions

Use clear, conventional-style commit messages so the history reads well to a reviewer:

| Prefix      | When to use                                  | Example                                              |
|-------------|-----------------------------------------------|-------------------------------------------------------|
| `feat:`     | New feature or capability                     | `feat: add Random Forest training pipeline`           |
| `fix:`      | Bug fix                                       | `fix: include record_id in CSV writer fieldnames`     |
| `docs:`     | Documentation only                            | `docs: add README usage guide and dataset table`       |
| `test:`     | Adding or updating tests                      | `test: add unit tests for data_handler module`         |
| `refactor:` | Code change with no behavior change           | `refactor: split validation logic into validators.py`  |
| `chore:`    | Tooling, config, dependency changes           | `chore: add requirements.txt and .gitignore`           |

### Suggested commit sequence for this project

```bash
git commit -m "chore: scaffold project folder structure"
git commit -m "feat: add synthetic dataset generator and initial CSV"
git commit -m "feat: implement config, exceptions, and validators modules"
git commit -m "feat: implement CSV-based data handler with CRUD operations"
git commit -m "feat: implement ML pipeline (preprocessing, training, prediction)"
git commit -m "feat: implement console application with menu-driven UI"
git commit -m "fix: include record_id in REQUIRED_COLUMNS for CSV writer"
git commit -m "test: add unit tests for validators, data handler, and ML pipeline"
git commit -m "docs: add README, project report, and presentation"
```

Committing in small, logical chunks like this (rather than one giant commit) is
generally what reviewers expect to see and makes the project history easy to follow.

## 4. Repository Description

> A console-based Machine Learning application that predicts household
> electricity usage and estimated bill amount from inputs like household
> size, AC usage, appliance count, and season. Built as Project 1 for the
> CodeVedX AI/ML Internship Program.

## 5. Suggested Topics / Tags

Add these as GitHub repository topics for discoverability:

```
machine-learning, python, scikit-learn, regression, random-forest,
console-application, csv, predictive-modeling, internship-project,
codevedx, ai-ml, utility-prediction
```

## 6. README Badges (optional polish)

If you want to add badges at the top of the README, these are appropriate
(no CI pipeline is included in this project, so stick to static badges):

```markdown
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
```

## 7. Before You Push — Checklist

- [ ] `data/utility_usage_data.csv` is present and not empty
- [ ] `models/*.pkl` exist after running training at least once (or note that they're gitignored and regenerated locally)
- [ ] `README.md` is complete and renders correctly on GitHub (check tables/code blocks)
- [ ] `requirements.txt` lists all dependencies
- [ ] `python -m pytest tests/ -v` passes locally before pushing
- [ ] No personal file paths, API keys, or credentials are committed
- [ ] `.gitignore` excludes `__pycache__/`, `.pytest_cache/`, and any virtual environment folder
