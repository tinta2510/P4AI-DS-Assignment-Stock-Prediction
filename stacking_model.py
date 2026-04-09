import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Utility functions copied/adapted from main.ipynb so this file can be
# run independently.
# ---------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def encode_labels(y_train, y_other=None):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    if y_other is None:
        return y_train_enc, le

    y_other_enc = le.transform(y_other)
    return y_train_enc, y_other_enc, le


class PurgedTimeSeriesSplit:
    """
    Same splitter idea as in main.ipynb.

    Splits on unique dates, purges the embargo window before each validation block,
    and keeps chronological ordering intact.
    """

    def __init__(self, n_splits=5, embargo_days=10, min_train_size=252):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size

    def split(self, df: pd.DataFrame):
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        unique_dates = pd.Index(sorted(df["date"].dropna().unique()))

        if len(unique_dates) < self.n_splits + 1:
            raise ValueError("Not enough unique dates for the requested number of splits.")

        fold_sizes = np.full(self.n_splits, len(unique_dates) // self.n_splits, dtype=int)
        fold_sizes[: len(unique_dates) % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size
            val_dates = unique_dates[start:stop]
            current = stop

            if len(val_dates) == 0:
                continue

            val_start = pd.Timestamp(val_dates.min())
            embargo_cutoff = val_start - pd.Timedelta(days=self.embargo_days)
            train_dates = unique_dates[unique_dates < embargo_cutoff]

            if len(train_dates) < self.min_train_size:
                continue

            train_idx = df.index[df["date"].isin(train_dates)].to_numpy()
            val_idx = df.index[df["date"].isin(val_dates)].to_numpy()

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx


RANDOM_STATE = 42


def build_model(family_name: str, params: Dict):
    if family_name == "xgb":
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

    if family_name == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )),
        ])

    if family_name == "svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, **params)),
        ])

    if family_name == "mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                early_stopping=True,
                max_iter=300,
                random_state=RANDOM_STATE,
                **params,
            )),
        ])

    raise ValueError(f"Unknown family: {family_name}")


@dataclass
class StackingArtifacts:
    base_models: Dict[str, object]
    meta_model: object
    label_encoder: LabelEncoder
    base_model_names: List[str]
    class_order: List[int]
    feature_cols: List[str]


class TimeSeriesStackingClassifier:
    """
    Two-level stacking for your triple-barrier classification notebook.

    Workflow:
    1) Build OOF predicted probabilities from the best version of each family.
    2) Train a multinomial logistic regression meta-model on those OOF features.
    3) Refit all base models on the full train/validation set.
    4) Predict on the holdout test using meta-features from full-fit base models.

    This avoids leakage better than fitting a stacker directly on in-sample base
    predictions.
    """

    def __init__(
        self,
        feature_cols: List[str],
        cv_splits: int = 5,
        embargo_days: int = 10,
        min_train_size: int = 252 * 2,
        meta_C: float = 1.0,
    ):
        self.feature_cols = feature_cols
        self.cv_splits = cv_splits
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
        self.meta_C = meta_C
        self.artifacts_: Optional[StackingArtifacts] = None

    @staticmethod
    def load_best_base_configs(cv_results_path: str) -> Dict[str, Dict]:
        cv = pd.read_csv(cv_results_path)
        metric_cols = [
            "accuracy", "balanced_accuracy", "macro_f1",
            "weighted_f1", "macro_precision", "macro_recall",
        ]
        summary = (
            cv.groupby(["family", "version"], as_index=False)[metric_cols]
            .mean()
            .sort_values(["balanced_accuracy", "macro_f1", "family"], ascending=[False, False, True])
        )
        best_versions = summary.groupby("family", as_index=False).first()

        configs = {}
        for _, row in best_versions.iterrows():
            family = row["family"]
            version = int(row["version"])
            params_row = cv[(cv["family"] == family) & (cv["version"] == version)].iloc[0]
            configs[family] = {
                "version": version,
                "params": json.loads(params_row["params_json"]),
            }
        return configs

    def _fit_single_model(self, family: str, params: Dict, X_train: pd.DataFrame, y_train: pd.Series):
        clf = build_model(family, params)
        if family == "xgb":
            y_train_enc, le = encode_labels(y_train)
            clf.fit(X_train, y_train_enc)
            return clf, le
        clf.fit(X_train, y_train)
        return clf, None

    def _predict_proba_single_model(self, family: str, clf, le: Optional[LabelEncoder], X: pd.DataFrame, class_order: List[int]):
        if family == "xgb":
            proba = clf.predict_proba(X)
            model_classes = list(le.classes_)
        else:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X)
                model_classes = list(clf.classes_)
            else:
                # fallback for models without predict_proba
                pred = clf.predict(X)
                model_classes = sorted(pd.Series(pred).unique().tolist())
                proba = np.zeros((len(X), len(model_classes)), dtype=float)
                for j, cls in enumerate(model_classes):
                    proba[:, j] = (pred == cls).astype(float)

        out = np.zeros((len(X), len(class_order)), dtype=float)
        class_to_idx = {cls: j for j, cls in enumerate(model_classes)}
        for k, cls in enumerate(class_order):
            if cls in class_to_idx:
                out[:, k] = proba[:, class_to_idx[cls]]
        return out

    def _build_oof_meta_features(self, trainval_df: pd.DataFrame, base_configs: Dict[str, Dict]):
        df = trainval_df.sort_values(["date", "symbol"]).reset_index(drop=True).copy()
        y = df["tb_label"]
        class_order = sorted(y.unique().tolist())

        splitter = PurgedTimeSeriesSplit(
            n_splits=self.cv_splits,
            embargo_days=self.embargo_days,
            min_train_size=self.min_train_size,
        )

        base_names = list(base_configs.keys())
        oof = np.full((len(df), len(base_names) * len(class_order)), np.nan, dtype=float)
        valid_rows = np.zeros(len(df), dtype=bool)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            X_train = train_df[self.feature_cols]
            y_train = train_df["tb_label"]
            X_val = val_df[self.feature_cols]

            col_offset = 0
            for family in base_names:
                params = base_configs[family]["params"]
                clf, le = self._fit_single_model(family, params, X_train, y_train)
                proba = self._predict_proba_single_model(family, clf, le, X_val, class_order)
                width = len(class_order)
                oof[val_idx, col_offset: col_offset + width] = proba
                col_offset += width

            valid_rows[val_idx] = True
            print(
                f"[stacking] fold {fold} | train={len(train_df):,} | val={len(val_df):,} | "
                f"train_end={train_df['date'].max().date()} | "
                f"val={val_df['date'].min().date()}->{val_df['date'].max().date()}"
            )

        if not valid_rows.any():
            raise RuntimeError("No OOF rows were produced. Relax min_train_size or reduce n_splits.")

        meta_cols = [f"{family}_p_{cls}" for family in base_names for cls in class_order]
        meta_X = pd.DataFrame(oof, columns=meta_cols, index=df.index)
        meta_df = pd.concat([df[["date", "symbol", "tb_label"]], meta_X], axis=1)
        meta_df = meta_df.loc[valid_rows].dropna().reset_index(drop=True)
        return meta_df, base_names, class_order

    def fit(self, trainval_df: pd.DataFrame, cv_results_path: str):
        base_configs = self.load_best_base_configs(cv_results_path)
        meta_train_df, base_names, class_order = self._build_oof_meta_features(trainval_df, base_configs)

        meta_X = meta_train_df.drop(columns=["date", "symbol", "tb_label"])
        meta_y = meta_train_df["tb_label"]
        meta_y_enc, meta_le = encode_labels(meta_y)

        meta_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=self.meta_C,
                max_iter=2000,
                multi_class="multinomial",
                random_state=RANDOM_STATE,
            )),
        ])
        meta_model.fit(meta_X, meta_y_enc)

        full_base_models = {}
        X_full = trainval_df[self.feature_cols]
        y_full = trainval_df["tb_label"]
        for family, cfg in base_configs.items():
            clf, le = self._fit_single_model(family, cfg["params"], X_full, y_full)
            full_base_models[family] = {"model": clf, "label_encoder": le}

        self.artifacts_ = StackingArtifacts(
            base_models=full_base_models,
            meta_model=meta_model,
            label_encoder=meta_le,
            base_model_names=base_names,
            class_order=class_order,
            feature_cols=self.feature_cols,
        )
        return self

    def _make_meta_features_from_full_base_models(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.artifacts_ is None:
            raise RuntimeError("Call fit(...) first.")

        X = df[self.artifacts_.feature_cols]
        parts = []
        for family in self.artifacts_.base_model_names:
            entry = self.artifacts_.base_models[family]
            proba = self._predict_proba_single_model(
                family=family,
                clf=entry["model"],
                le=entry["label_encoder"],
                X=X,
                class_order=self.artifacts_.class_order,
            )
            cols = [f"{family}_p_{cls}" for cls in self.artifacts_.class_order]
            parts.append(pd.DataFrame(proba, columns=cols, index=df.index))
        return pd.concat(parts, axis=1)

    def predict(self, df: pd.DataFrame):
        meta_X = self._make_meta_features_from_full_base_models(df)
        pred_enc = self.artifacts_.meta_model.predict(meta_X)
        return self.artifacts_.label_encoder.inverse_transform(pred_enc.astype(int))

    def predict_proba(self, df: pd.DataFrame):
        meta_X = self._make_meta_features_from_full_base_models(df)
        return self.artifacts_.meta_model.predict_proba(meta_X)

    def evaluate(self, test_df: pd.DataFrame):
        y_true = test_df["tb_label"]
        y_pred = self.predict(test_df)
        metrics = compute_metrics(y_true, y_pred)

        print("\nStacking test metrics:")
        print(metrics)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        labels_sorted = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        print("\nConfusion matrix (labels order =", labels_sorted, "):")
        print(cm)

        return {
            "metrics": metrics,
            "y_test": y_true,
            "y_pred": y_pred,
            "confusion_matrix": cm,
        }

    def save(self, output_dir: str):
        if self.artifacts_ is None:
            raise RuntimeError("Nothing to save. Fit the model first.")
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self, os.path.join(output_dir, "stacking_model.joblib"))
        return os.path.join(output_dir, "stacking_model.joblib")


# ---------------------------------------------------------------------
# Example usage inside your notebook:
#
# from stacking_model import TimeSeriesStackingClassifier
#
# stacker = TimeSeriesStackingClassifier(
#     feature_cols=feature_cols,
#     cv_splits=5,
#     embargo_days=10,
#     min_train_size=252 * 2,
#     meta_C=1.0,
# )
# stacker.fit(trainval_df, cv_results_path="results/cross_val.csv")
# stack_res = stacker.evaluate(test_df)
# stacker.save("./models")
# ---------------------------------------------------------------------
