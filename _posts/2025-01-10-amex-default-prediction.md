---
layout: post
title: "AmEx Default Prediction: End-to-End Risk Modeling With the AmEx Metric"
image: "/posts/classification-title-img.png"
tags: [Credit Risk, Machine Learning, Gradient Boosting, Feature Engineering, Optuna, LightGBM, Python]
---

# Table of Contents
- [00. Project Overview](#overview)
- [01. Business Context & Problem](#business)
- [02. Data Cleaning & Preprocessing](#preprocessing)
- [03. Feature Engineering & Aggregations](#features)
- [04. Modeling & Custom AmEx Metric](#modeling)
- [05. Evaluation & Results](#results)
- [06. Experimentation & Tracking (Databricks + MLflow)](#experimentation)
- [07. Links](#links)

---

## <a name="overview"></a>00. Project Overview

Interview summary: I built an end-to-end credit default risk model for the American Express Kaggle competition. The core challenges were heavy class imbalance, a custom evaluation metric (AmEx metric), and large-scale sequential tabular data that benefits from strong feature aggregation and regularization. I used LightGBM with carefully engineered aggregations, tuned via Optuna, and validated with a time-aware split. 

**Kaggle:** [AmEx Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)

At a glance
- Objective: rank customers by default risk according to the AmEx metric
- Approach: per-customer aggregations + LightGBM optimized with custom AmEx metric
- Experimentation: Databricks + MLflow for tracking and model registry
- Highlights: scalable feature build (PySpark), clear CV design, readable code

---

## <a name="business"></a>01. Business Context & Problem

Goal: Predict whether a customer will default in the next period using historical monthly statements. Business impact is twofold: reduce credit losses (better risk ranking) and keep approval friction low (precision at the top deciles).

Constraints: Class imbalance, cost of false negatives (missed risk), and need for stable ranking across cohorts.

---

## <a name="preprocessing"></a>02. Data Cleaning & Preprocessing

Steps:
- Consistent dtypes and downcasting to reduce memory footprint
- Handle missing values via group-wise imputation or sentinel encoding
- Remove obvious data errors and enforce monotonic time order per customer

```python
import pandas as pd
import numpy as np

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

# Ensure types and order
for c in train.select_dtypes(include=["int64"]).columns:
    train[c] = pd.to_numeric(train[c], downcast="integer")
for c in train.select_dtypes(include=["float64"]).columns:
    train[c] = pd.to_numeric(train[c], downcast="float")

train = train.sort_values(["customer_ID","S_2"])  # S_2 is the statement date
```

---

## <a name="features"></a>03. Feature Engineering & Aggregations

I built per-customer aggregations over the sequential statements to capture credit behavior trends.

```python
aggs = {
    "B_1": ["mean","std","min","max","last"],
    "P_2": ["mean","std","last"],
    "D_39": ["mean","last"],
    # ... add other numeric columns
}

def last(x):
    return x.iloc[-1]

def build_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = pd.to_datetime(df["S_2"]).dt.to_period("M").astype(str)
    grouped = df.groupby("customer_ID")
    feats = grouped.agg(aggs)
    feats.columns = [f"{c}_{stat}" for c, stat in feats.columns]
    feats.reset_index(inplace=True)
    # Recency deltas
    for c in ["B_1","P_2","D_39"]:
        if f"{c}_last" in feats.columns and f"{c}_mean" in feats.columns:
            feats[f"{c}_last_delta_mean"] = feats[f"{c}_last"] - feats[f"{c}_mean"]
    return feats

train_feats = build_customer_aggregates(train)
```

PySpark (Databricks) equivalent

```python
from pyspark.sql import functions as F
from pyspark.sql import Window

df = spark.read.parquet("/mnt/amex/train.parquet")  # columns: customer_ID, S_2, numeric features

# Ensure ordering per customer for recency aware calcs
w = Window.partitionBy("customer_ID").orderBy(F.col("S_2").cast("timestamp")).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

agg_df = df.groupBy("customer_ID").agg(
    F.mean("B_1").alias("B_1_mean"), F.stddev("B_1").alias("B_1_std"), F.min("B_1").alias("B_1_min"), F.max("B_1").alias("B_1_max"), F.last("B_1", ignorenulls=True).alias("B_1_last"),
    F.mean("P_2").alias("P_2_mean"), F.stddev("P_2").alias("P_2_std"), F.last("P_2", ignorenulls=True).alias("P_2_last"),
    F.mean("D_39").alias("D_39_mean"), F.last("D_39", ignorenulls=True).alias("D_39_last")
)

# Recency deltas: last - mean
agg_df = agg_df.withColumn("B_1_last_delta_mean", F.col("B_1_last") - F.col("B_1_mean")) \
               .withColumn("P_2_last_delta_mean", F.col("P_2_last") - F.col("P_2_mean")) \
               .withColumn("D_39_last_delta_mean", F.col("D_39_last") - F.col("D_39_mean"))

# Optional volatility/missingness counts
vol_df = df.groupBy("customer_ID").agg(
    F.stddev("B_1").alias("B_1_vol"),
    F.sum(F.when(F.col("B_1").isNull(), 1).otherwise(0)).alias("B_1_null_cnt")
)

train_feats_spark = agg_df.join(vol_df, on="customer_ID", how="left")
```

Robust ratio via PySpark UDF (example)

```python
from pyspark.sql.types import DoubleType

@F.udf(returnType=DoubleType())
def safe_divide(numer, denom):
    try:
        d = float(denom)
        return float(numer) / d if d not in (0.0, None) else 0.0
    except Exception:
        return 0.0

df = df.withColumn("utilization_ratio", safe_divide(F.col("B_1"), F.col("P_2")))
```

---

## <a name="modeling"></a>04. Modeling & Custom AmEx Metric

I used LightGBM with the official AmEx metric for model selection. The AmEx metric rewards correct ranking at top deciles and penalizes false positives less aggressively than AUC.

```python
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split

# Custom AmEx metric implementation (vectorized)
def amex_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos
    order = np.argsort(-y_pred)
    y_true = y_true[order]

    # Top 4% capture
    top_four = int(0.04 * y_true.shape[0])
    top_four_capture = y_true[:top_four].sum() / (n_pos + 1e-6)

    # Weighted Gini
    weights = np.where(y_true == 0, 20.0, 1.0)
    cum_weights = np.cumsum(weights)
    cum_pos = np.cumsum(y_true * weights)
    lorentz = cum_pos / (cum_pos[-1] + 1e-6)
    gini = (lorentz - cum_weights / cum_weights[-1]).sum()

    # Normalized by perfect model
    perfect_order = np.argsort(-y_true)
    y_true_sorted = y_true[perfect_order]
    weights_sorted = weights[perfect_order]
    cum_weights_p = np.cumsum(weights_sorted)
    cum_pos_p = np.cumsum(y_true_sorted * weights_sorted)
    lorentz_p = cum_pos_p / (cum_pos_p[-1] + 1e-6)
    gini_max = (lorentz_p - cum_weights_p / cum_weights_p[-1]).sum() + 1e-6

    normalized_gini = gini / gini_max
    return 0.5 * (normalized_gini + top_four_capture)

def lgb_amex_eval(y_pred, dataset):
    y_true = dataset.get_label()
    return "amex", amex_metric(y_true, y_pred), True

feature_cols = [c for c in train_feats.columns if c != "customer_ID"]
X_train, X_val = train_feats.iloc[:-20000], train_feats.iloc[-20000:]
y_train, y_val = train.loc[train_feats.index[:-20000], "target"], train.loc[train_feats.index[-20000:], "target"]

train_set = lgb.Dataset(X_train[feature_cols], label=y_train)
valid_set = lgb.Dataset(X_val[feature_cols], label=y_val)

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 512),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 2000),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "verbose": -1,
    }
    model = lgb.train(params, train_set, valid_sets=[valid_set],
                      feval=lgb_amex_eval, num_boost_round=8000,
                      early_stopping_rounds=400, verbose_eval=200)
    y_pred = model.predict(X_val[feature_cols])
    return 1.0 - amex_metric(y_val.values, y_pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
model = lgb.train({**best_params, "objective":"binary", "metric":"None", "verbose":-1},
                  train_set, valid_sets=[valid_set], feval=lgb_amex_eval,
                  num_boost_round=study.best_trial.number + 400,
                  early_stopping_rounds=400, verbose_eval=200)
```

---

## <a name="results"></a>05. Evaluation & Results

Interview angle: I tracked the AmEx metric on validation folds and monitored stability across deciles. The final model achieved a strong AmEx score, with feature importance dominated by rolling means/last values and utilization deltas.

---

## Related Work: What Winners Did

- Boosted trees dominated with rich per-customer aggregations (mean/std/min/max/last, recency deltas, utilization ratios)
- Time-aware customer-level CV; optimize validation proxy for AmEx metric; threshold tuning
- Ensembling across seeds/models; rank averaging; light stacking
- DL sequences (GRU/LSTM/Transformer) explored; competitive when ensembled but usually matched by strong GBMs

## <a name="experimentation"></a>06. Experimentation & Tracking (Databricks + MLflow)

I ran training experiments on Databricks and tracked them with MLflow—logging parameters, AmEx metric, feature lists, and model artifacts. Model versions are registered for promotion after validation.

```python
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

with mlflow.start_run(run_name="lgb-amex-v1"):
    mlflow.log_params({
        "learning_rate": best_params.get("learning_rate", 0.03),
        "num_leaves": best_params.get("num_leaves", 256),
        "feature_fraction": best_params.get("feature_fraction", 0.8),
        "bagging_fraction": best_params.get("bagging_fraction", 0.8)
    })
    model = lgb.train({**best_params, "objective":"binary", "metric":"None"},
                      train_set, valid_sets=[valid_set], feval=lgb_amex_eval,
                      num_boost_round=2000, early_stopping_rounds=300, verbose_eval=200)
    y_val_pred = model.predict(X_val[feature_cols])
    mlflow.log_metric("amex_val", amex_metric(y_val.values, y_val_pred))
    mlflow.lightgbm.log_model(model, artifact_path="model")
```

Alternative: enable autologging

```python
import mlflow
import mlflow.lightgbm

mlflow.lightgbm.autolog(log_models=True)
# train your LightGBM model as usual; params/metrics/artifacts will auto-log
```

---

## <a name="links"></a>07. Links

- Kaggle competition: [AmEx Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)


