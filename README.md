# ML Classification & Clustering Pipeline

[![CI Pipeline](https://github.com/chopragauri/ml-classification-clustering-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/chopragauri/ml-classification-clustering-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)

An end-to-end machine learning pipeline covering classification, ensemble learning, and clustering — applied to the [Country Socioeconomic Dataset](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) to build a country development intelligence system.

---

## Architecture

```
Country-data.csv (167 countries × 9 features)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  src/features.py  — Feature Engineering           │
│  • Composite development score (weighted norm)    │
│  • Binary target derivation (is_developed)        │
│  • Train/test split + StandardScaler              │
└───────────────────┬───────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌─────────────────────────────┐
│  src/train.py    │  │  src/cluster.py              │
│  11 Classifiers  │  │  4 Clustering Algorithms     │
│  GridSearchCV    │  │  Silhouette / DB / CH eval   │
│  ROC-AUC tuning  │  │                              │
└────────┬─────────┘  └──────────────┬──────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐       ┌──────────────────────────┐
│ results/        │       │ plots/                   │
│ metrics.json    │       │ 32 visualisations        │
└─────────────────┘       └──────────────────────────┘
```

---

## What This Project Does

- Engineers a **binary classification target** (`is_developed`) from raw socioeconomic indicators using a composite development score
- Trains and tunes **11 classifiers** with GridSearchCV and cross-validation
- Applies **4 clustering algorithms** to segment countries by economic and health patterns
- Produces **32 visualisations** — confusion matrices, ROC curves, feature importance charts, dendrograms, PCA scatter plots, and radar charts

---

## Project Structure

```
ml-classification-clustering-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI — import checks, schema validation, leaderboard print
├── src/
│   ├── features.py                   # Feature engineering & preprocessing
│   ├── train.py                      # Classification training & evaluation
│   └── cluster.py                    # Clustering algorithms & metrics
├── results/
│   └── metrics.json                  # Saved leaderboard & clustering scores
├── plots/                            # Generated visualisations
├── Customer_Intelligence_System.ipynb
├── config.yaml                       # Centralised hyperparameter grids
└── requirements.txt
```

---

## Algorithms

| Category | Models |
|---|---|
| **Classification** | Logistic Regression · Naive Bayes · KNN · SVM · Decision Tree |
| **Ensemble** | Random Forest · AdaBoost · Gradient Boosting · XGBoost · LightGBM · Stacking |
| **Clustering** | K-Means · K-Medoids · DBSCAN · Hierarchical (Ward) |
| **Evaluation** | Confusion Matrix · ROC-AUC · F1 · Silhouette · Davies-Bouldin · Calinski-Harabasz |

---

## Dataset

- **Source:** [Kaggle — Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)
- **Size:** 167 countries × 9 features
- **Features:** `child_mort` · `exports` · `health` · `imports` · `income` · `inflation` · `life_expec` · `total_fer` · `gdpp`

---

## Results

### Classification Leaderboard

| Rank | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| 🥇 | **Gradient Boosting** | 0.853 | 0.872 | **0.997** |
| 2 | Naive Bayes | 0.853 | 0.872 | 0.993 |
| 3 | Random Forest | **0.882** | **0.895** | 0.990 |
| 4 | Logistic Regression | 0.824 | 0.850 | 0.990 |
| 5 | XGBoost | 0.853 | 0.872 | 0.986 |
| 6 | SVM | 0.794 | 0.829 | 0.983 |
| 7 | AdaBoost | 0.794 | 0.821 | 0.979 |
| 8 | Stacking | 0.765 | 0.810 | 0.976 |
| 9 | LightGBM | 0.765 | 0.810 | 0.972 |
| 10 | KNN | 0.794 | 0.829 | 0.962 |
| 11 | Decision Tree | 0.735 | 0.780 | 0.735 |

### Clustering Results

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| **DBSCAN** | 3 | **0.441** | 0.876 | 48.7 |
| K-Medoids | 5 | 0.300 | **0.869** | **55.1** |
| K-Means | 5 | 0.299 | 0.872 | 54.3 |
| Hierarchical | 5 | 0.219 | 1.301 | 41.2 |

---

## Key Findings

- `child_mort` and `life_expec` are the **strongest predictors** of development status across all models
- Countries form **5 natural clusters** ranging from least-developed (Sub-Saharan Africa) to high-income (Western Europe, North America)
- **Gradient Boosting** achieves the best ROC-AUC of **0.997**
- **DBSCAN** achieves the best Silhouette score (0.441) and identifies economic outliers: Luxembourg, Singapore, Qatar, United States

---

## How to Run

### Using the modular src/ pipeline

```bash
pip install -r requirements.txt

python - <<'EOF'
from src.features import load_config, load_and_engineer
from src.train import train_all_models, build_leaderboard
from src.cluster import run_all_clusters, evaluate_clusters

cfg = load_config("config.yaml")
X_train, X_test, y_train, y_test, scaler = load_and_engineer("Country-data.csv", cfg)

results = train_all_models(X_train, y_train, X_test, y_test, cfg)
print(build_leaderboard(results))

cluster_labels = run_all_clusters(X_train, cfg)
print(evaluate_clusters(X_train, cluster_labels))
EOF
```

### Google Colab (full notebook)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook` → select `Customer_Intelligence_System.ipynb`
3. `Runtime → Run all`
4. Upload `Country-data.csv` from [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) when prompted

### Local (notebook)
```bash
pip install -r requirements.txt
jupyter notebook Customer_Intelligence_System.ipynb
```

---

## CI / Continuous Integration

Every push to `main` triggers the [CI workflow](.github/workflows/ci.yml) which:

1. **Installs** all dependencies from `requirements.txt`
2. **Validates** that all `src/` modules import cleanly
3. **Checks** `config.yaml` schema — required top-level keys, hyperparameter sections
4. **Validates** `results/metrics.json` — all 11 classifiers and 4 clustering algorithms present
5. **Prints** the full leaderboard sorted by ROC-AUC

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `scikit-learn` | Base classifiers, GridSearchCV, preprocessing, clustering |
| `XGBoost` | Gradient boosted trees |
| `LightGBM` | Fast gradient boosting |
| `scikit-learn-extra` | K-Medoids (PAM) |
| `matplotlib` / `seaborn` | Visualisations |
| `PyYAML` | Config management |
| GitHub Actions | CI pipeline |
