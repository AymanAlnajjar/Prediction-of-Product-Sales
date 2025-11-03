
# Prediction Of Product Sales
## Forecasting retail item sales from product & outlet attributes

**Author**: Ayman Alnajjar

---

### Business problem:

Retailers need to estimate future sales for each product–outlet combination to plan inventory, optimize shelf space, and forecast revenue. The goal of this project is to build a regression model that predicts `Item_Outlet_Sales` using product characteristics (e.g., MRP, visibility, fat content, category) and outlet attributes (e.g., size, location tier, type). This repository showcases my end‑to‑end data science workflow: data understanding, cleaning, EDA, feature encoding, model training/tuning, and evaluation.

---

### Data:

- **Source:** Public BigMart‑style retail dataset (loaded via a Google Drive file link inside the notebook).
- **Observations:** ~8,523 rows
- **Features:** ~12 explanatory columns after dropping ID and year fields for modeling
- **Target:** `Item_Outlet_Sales` (continuous)

Key notes from data quality checks:
- Missing values in **Item_Weight** and **Outlet_Size**.
- Inconsistent labels in **Item_Fat_Content** (e.g., `LF` → `Low Fat`, etc.).
- No duplicate rows detected.

---

## Methods

- **Data preparation**
  - Resolve categorical inconsistencies (e.g., normalize `Item_Fat_Content` labels).
  - Impute missing values: numeric with mean; categorical with most frequent.
  - Encode categoricals:
    - **Ordinal** for `Outlet_Size` (`Small` < `Medium` < `High`) and `Outlet_Location_Type` (`Tier 1` < `Tier 2` < `Tier 3`).
    - **One‑Hot Encoding** for remaining nominal features (with `handle_unknown="ignore"`).
  - **Scaling** numeric features with `StandardScaler` (especially for linear models).
  - Split data using `train_test_split` with a fixed `random_state` for reproducibility.

- **Models evaluated**
  - **Linear Regression** as a baseline.
  - **RandomForestRegressor** (default).
  - **RandomForestRegressor** with **GridSearchCV** tuning (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).

- **Evaluation**
  - Custom utility prints **MAE**, **MSE**, **RMSE**, **R²** for train and test sets.
  - Emphasis on **RMSE** and **R²** to quantify typical error magnitude and explained variance.

---

## Results

### Example visualizations (place any exported images into a `figures/` subfolder and update paths below)

#### Visual 1 — Feature distributions
![sample image](figures/project1_sample_image.png)

> Example: Histograms/boxplots to understand numeric/categorical distributions and detect outliers.

#### Visual 2 — Correlation heatmap
![correlation heatmap](figures/correlation_heatmap.png)

> Example: Heatmap to quickly scan linear relationships between numeric features and the target.

---

## Model

**Best performing approach:** Tuned **RandomForestRegressor** via GridSearchCV.

**Held‑out test metrics:**
- **MAE:** ~741.05
- **RMSE:** ~1,065.89
- **R²:** ~0.589

**Interpretation for the business problem:**
- An RMSE of ~1,066 sales units means the typical prediction error magnitude is roughly 1.1k units per item–outlet record on the test set.
- An R² of ~0.59 indicates the model explains about 59% of the variance in sales. This is a meaningful improvement over the linear baseline and provides a practical signal for planning decisions (stocking, promotions, revenue forecasting), while leaving room for improvement with richer features.

> Baseline **Linear Regression** (for comparison) achieved approximately:
> - Test **RMSE:** ~1,094
> - Test **R²:** ~0.566

---

## Recommendations

- Use the Random Forest model as the current baseline for demand forecasting tasks where tabular product/outlet attributes are available.
- Couple predictions with safety stock rules to buffer error (e.g., set reorder points at forecast + k·RMSE).
- Re‑train monthly/quarterly as new sales data arrives to prevent performance drift.

---

## Limitations & Next Steps

- **Data richness:** No temporal features (seasonality, holidays, trends) or price elasticity features (promotions/discounts) are modeled yet.
- **Granularity:** The dataset aggregates at item–outlet level but does not include time series per SKU–store; adding that would likely improve accuracy.
- **Explainability:** Tree‑based feature importance and permutation importance can be added to identify top drivers; partial dependence/SHAP for deeper insights.
- **Next steps:**
  - Add **time‑aware** features (month/quarter/holiday), lagged sales, and promo flags.
  - Explore **Gradient Boosting** models (XGBoost/LightGBM/CatBoost) with careful tuning and cross‑validation.
  - Perform **hyperparameter sweeps** with randomized search for broader coverage.
  - Build a **simple API** or batch scoring pipeline and set up monitoring for real‑world use.

---

## Reproducibility

- Environment: Python 3.x, `pandas`, `scikit‑learn`, `numpy`, `matplotlib`.
- Everything runs from the single notebook: **`Prediction_of_Product_Sale.ipynb`**.
- Steps:
  1. Install dependencies: `pip install -r requirements.txt` (or install the libraries above).
  2. Open the notebook and run cells top‑to‑bottom. The dataset is fetched via a Google Drive file link in the notebook.
  3. Export figures to the `figures/` folder if you want them embedded in this README.

---

## Repository Structure (suggested)

```
.
├── Prediction_of_Product_Sale.ipynb   # full analysis & modeling workflow
├── figures/                           # exported charts/images (optional)
│   ├── project1_sample_image.png
│   └── correlation_heatmap.png
├── README.md                          # this file
└── requirements.txt                   # optional, freeze your environment
```

---

## Acknowledgements

This project is inspired by public retail sales datasets (BigMart‑style) commonly used for regression modeling practice.
