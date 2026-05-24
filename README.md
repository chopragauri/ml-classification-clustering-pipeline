# ML Classification & Clustering Pipeline

An end-to-end machine learning pipeline covering classification, ensemble learning, and clustering — applied to the [Country Socioeconomic Dataset](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) to build a country development intelligence system.

---

## What This Project Does

- Engineers a **binary classification target** (`is_developed`) from raw socioeconomic indicators using a composite development score
- Trains and tunes **11 classifiers** with GridSearchCV and cross-validation
- Applies **4 clustering algorithms** to segment countries by economic and health patterns
- Produces **32 visualisations** — confusion matrices, ROC curves, feature importance charts, dendrograms, PCA scatter plots, and radar charts

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

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting** | 0.853 | 0.872 | **0.997** |
| Naive Bayes | 0.853 | 0.872 | 0.993 |
| Random Forest | 0.882 | 0.895 | 0.990 |
| Logistic Regression | 0.824 | 0.850 | 0.990 |
| XGBoost | 0.853 | 0.872 | 0.986 |
| SVM | 0.794 | 0.829 | 0.983 |
| AdaBoost | 0.794 | 0.821 | 0.979 |
| Stacking | 0.765 | 0.810 | 0.976 |
| LightGBM | 0.765 | 0.810 | 0.972 |
| KNN | 0.794 | 0.829 | 0.962 |
| Decision Tree | 0.735 | 0.780 | 0.735 |

### Clustering Results

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|---|
| K-Means | 5 | 0.299 | 0.872 |
| K-Medoids | 5 | 0.300 | 0.869 |
| DBSCAN | 3 | 0.441 | 0.876 |
| Hierarchical | 5 | 0.219 | 1.301 |

---

## Key Findings

- `child_mort` and `life_expec` are the **strongest predictors** of development status across all models
- Countries form **5 natural clusters** ranging from least-developed (Sub-Saharan Africa) to high-income (Western Europe, North America)
- **Gradient Boosting** achieves the best ROC-AUC of **0.997**
- DBSCAN identifies economic outliers: Luxembourg, Singapore, Qatar, United States

---

## How to Run

### Google Colab
1. Open [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook` → select `Customer_Intelligence_System.ipynb`
3. `Runtime → Run all`
4. Upload `Country-data.csv` from [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) when prompted

### Local
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm scikit-learn-extra scipy
jupyter notebook Customer_Intelligence_System.ipynb
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
