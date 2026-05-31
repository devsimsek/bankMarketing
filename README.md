# Bank Marketing Campaign - Term Deposit Prediction

This repository contains our group's solution to the Topic 4 micro-project for the THA Data Analytics course. Used to predict whether a client will subscribe to a term deposit using demographic, financial, campaign, and macroeconomic features. Built for a Portuguese bank's direct marketing optimization.

## Team Members

- [Metin Simsek](https://smsk.dev)
- Zheyu Yin
- Kirill Radiushin
- Diego Martins da Silva
- Farhan Mansoor
- Sonia Woks Keuleu

## Quick Start

```bash
pip install -r requirements.txt
# Navigate to visual studio code and execute all cells or use jupyter notebook to run the notebook.
```

Open `Presentation.html` for the 13-slide presentation summarizing our findings.

> We have utilised Claude's beautiful Presentation generation feature for the theming and design of the presentation.

## Data

Two datasets from [Moro et al., 2014] (UCI ML Repository):

| File                            | Rows   | Features | Notes                                                                                                              |
| ------------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------ |
| `data/bank-additional-full.csv` | 41,188 | 21       | **Used for modeling.** Includes 5 economic indicators from Banco de Portugal + `day_of_week`. Semicolon-separated. |
| `data/bank-full.csv`            | 45,211 | 17       | Original version. Kept for reference only. No economic indicators.                                                 |

### Economic Indicators

All five sourced from Banco de Portugal, reflecting the 2008–2010 financial crisis recovery:

| Feature          | Description                    | Frequency |
| ---------------- | ------------------------------ | --------- |
| `emp.var.rate`   | Employment variation rate      | Quarterly |
| `cons.price.idx` | Consumer price index           | Monthly   |
| `cons.conf.idx`  | Consumer confidence index      | Monthly   |
| `euribor3m`      | Euribor 3-month interbank rate | Daily     |
| `nr.employed`    | Number of employees            | Quarterly |

### Target

`y` - term deposit subscription (yes / no). Severe class imbalance: **88.7% no, 11.3% yes**.

## Notebook Outline

| Section                                | Content                                                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Imports & Setup                     | All libraries, plot style                                                                                                            |
| 2. Data Loading                        | Load `bank-additional-full.csv`                                                                                                      |
| 3. EDA & Cleanup                       | Target distribution, numerical/categorical analysis, correlation, duration analysis, economic indicator time-series & decile scatter |
| 4. Preprocessing & Feature Engineering | "unknown" as category, binary encoding, OHE, StandardScaler, 4 composite features, SMOTE, 80/20 stratified split                     |
| 5. Model Selection                     | Logistic Regression, Random Forest, XGBoost - 5-fold CV on SMOTE-balanced data                                                       |
| 6. Evaluation                          | Test set (original distribution): confusion matrices, ROC curves, classification reports                                             |
| 6.1 Feature Importance                 | RF + XGBoost top-20, SHAP beeswarm & waterfall, PR curves, lift/gains, calibration, threshold tuning                                 |
| 6.2 Duration-Inclusive Model           | Reference model with data leakage - theoretical upper bound                                                                          |
| 7–10                                   | Discussion, business recommendations, conclusion                                                                                     |

## Key Results

### Cross-Validation (SMOTE-balanced, 5-fold)

| Model               | ROC-AUC   | F1        | Precision | Recall    |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.800     | 0.707     | 0.806     | 0.630     |
| Random Forest       | 0.959     | 0.898     | 0.914     | 0.882     |
| **XGBoost**         | **0.972** | **0.929** | **0.954** | **0.905** |

### Test Set (Original Imbalanced Distribution - 8,238 samples)

| Model               | Accuracy  | Precision | Recall    | F1        | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 0.827     | 0.355     | 0.651     | 0.459     | 0.798     |
| Random Forest       | 0.886     | 0.493     | 0.574     | 0.531     | 0.806     |
| **XGBoost**         | **0.897** | **0.561** | **0.412** | **0.475** | **0.798** |

### Duration-Inclusive Reference

XGBoost with `duration`: ROC-AUC 0.952, F1 0.662.

The gap from 0.952 -> 0.798 quantifies the cost of a production-realistic model.

### Ablation Study: Impact of Economic Indicators

XGBoost trained **without** the 5 Banco de Portugal indicators vs the full model:

| Metric    | No Econ | With Econ | Delta    |
| --------- | ------- | --------- | -------- |
| CV AUC   | 0.9455 | 0.9723   | +0.0268 |
| CV F1    | 0.8862 | 0.9286   | +0.0424 |
| Test ROC-AUC | 0.7725 | 0.7981 | +0.0256 |
| Test F1  | 0.4552 | 0.4771   | +0.0219 |

Economic indicators provide a clear lift across both CV and test metrics, confirming their value for production deployment.

### Threshold Tuning (FN:FP = 4:1)

| Threshold          | Precision | Recall    | F1    |
| ------------------ | --------- | --------- | ----- |
| Default (0.50)     | 0.568     | 0.407     | 0.474 |
| **Optimal (0.31)** | **0.501** | **0.575** | -     |

## Model Pipeline

```
Raw Data (41,188 \* 21)
  -> Preserve "unknown" as category
  -> Binary encode (default, housing, loan)
  -> One-Hot Encode (7 categorical features, drop='first')
  -> Feature Engineering (4 composite features)
  -> StandardScaler (13 numerical features)
  -> SMOTE oversampling (58,476 balanced samples)
  -> XGBoost (200 trees, max_depth=6, lr=0.05)
  -> 53 features after encoding (37 OHE + 13 numerical + 3 binary)
```

## Engineered Features

| Feature               | Formula                           | Rationale                                                             |
| --------------------- | --------------------------------- | --------------------------------------------------------------------- |
| `pdays_was_contacted` | `pdays != 999`                    | Decouples the 999 sentinel (never contacted) from numeric scale       |
| `engagement_score`    | `(previous + 1) / (campaign + 1)` | RFM-inspired - engagement relative to contact frequency               |
| `loan_burden`         | `housing_bin + loan_bin`          | Total active loans (0, 1, 2)                                          |
| `euribor_cpi_spread`  | `euribor3m − CPI / 100`           | Real interest rate proxy - positive spread = deposits more attractive |

All four appear in top-20 feature importance rankings.

## Novel Visualizations

Six techniques beyond standard EDA:

1. **SHAP beeswarm summary** - global feature attribution with direction & magnitude
2. **SHAP waterfall** - single-prediction explanation for interpretability
3. **Precision-Recall curves** - more informative than ROC for 11.3% positive class
4. **Lift & cumulative gains chart** - "contact top 20% -> capture X% of subscribers"
5. **Calibration curves** - reliability diagram (15 bins)
6. **Decision threshold tuning** - cost-sensitive sweep with FN:FP = 4:1 ratio

## Key Technical Decisions

| Decision                                | Why                                                                   |
| --------------------------------------- | --------------------------------------------------------------------- |
| "unknown" as category                   | Not missing at random - carries predictive signal                     |
| SMOTE only, no class_weight             | Double correction overcorrects - confirmed empirically                |
| Duration excluded from production model | Post-hoc feature - unknown at prediction time (data leakage)          |
| OneHotEncoder `drop='first'`            | Avoids dummy variable trap                                            |
| Tree-based models                       | Handle multicollinearity (euribor3m ↔ nr.employed ρ ≈ 0.94) naturally |
| XGBoost recommended                     | Best CV ranking (AUC 0.972), highest test precision (0.561)           |

## Environment Notes

Tested with: Python 3.14, pandas 3.0, scikit-learn 1.8, xgboost 3.2, shap 0.51.

## References

- Moro, S., Cortez, P., & Rita, P. (2014). _A Data-Driven Approach to Predict the Success of Bank Telemarketing._ Decision Support Systems.
- [UCI ML Repository: Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing)
