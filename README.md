# HKU Student Housing Preference Analysis

Predicting whether a student lives in HKU dorms (halls/residential colleges) vs. off-campus housing, based on a self-collected survey, using logistic regression, CART, and random forest.

> Originally built as a course project for IIMT2641 (Introduction to Business Analytics), Oct 2025.

## Summary

- **Data**: 162 responses to a self-designed Google Forms survey (160 after removing incomplete rows), collected from HKU students.
- **Target**: whether the respondent currently lives in an HKU dorm (binary).
- **Predictors**: student status (local/mainland/international), year of study, faculty, monthly housing budget, scholarship status, and Likert-scale (1–5) importance ratings for price, commute, privacy, study environment, social life, facilities, and rules/flexibility.
- **Models**: logistic regression, CART (decision tree), and random forest.
- **Result**: random forest and CART reach up to ~72–75% accuracy on a single held-out test split; a more robust 5-fold cross-validation puts average accuracy closer to **60–65%**, with logistic regression underperforming the tree-based models. The strongest predictors across models are **price sensitivity, commute time, privacy, and year of study**.

## Why the range instead of one number

A single train/test split can make a model look better or worse than it really is, especially with ~160 rows. This repo reports both:
1. The original single-split result from the course submission (`scripts/`, R).
2. A 5-fold cross-validated re-check (`analysis/`, Python) to see how stable that number actually is.

Being upfront about that gap is the point of including both — a single lucky split isn't a claim worth making on its own.

## Repo structure

```
data/                Raw survey export (anonymous, no PII)
scripts/              Original R analysis (data cleaning, logistic regression, CART, random forest)
analysis/             Python re-implementation with proper cross-validation, for a sanity check
docs/                 Survey instrument, and a proposed sustainability extension (not yet collected)
```

## Limitations

- Small sample (n=160) limits generalizability.
- No sustainability-related variables were collected in this survey round — see `docs/sustainability_extension_proposal.md` for two additional Likert-scale questions drafted for a future data collection round, not yet fielded.
- Missing predictors likely to matter: distance from home, family income, prior dorm experience.
- Class balance is close to even (81 dorm / 79 off-campus), which helps, but the modest predictor set caps how much signal is available.

## How to run

**R** (`scripts/`): requires `caTools`, `caret`, `rpart`, `rpart.plot`, `randomForest`, `ROCR`, `pROC`, `car`.

**Python** (`analysis/`): `pip install pandas numpy scikit-learn`, then `python analysis/python_reanalysis_honest_cv.py`.
