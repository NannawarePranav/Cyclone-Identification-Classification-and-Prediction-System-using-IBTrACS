"""
Cyclone ML Custom Estimators
============================
Portable Scikit-learn compatible custom estimators for tropical cyclone modeling.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


class CycloneClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier wrapping XGBClassifier with integer target
    mapping, permitting string class labels inside Pipeline and permutation importance.
    """
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_lambda: float = 5.0,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            tree_method='hist',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("CycloneClassifier is not fitted.")
        preds_enc = self.model_.predict(X)
        return self.classes_[preds_enc]

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("CycloneClassifier is not fitted.")
        return self.model_.predict_proba(X)
