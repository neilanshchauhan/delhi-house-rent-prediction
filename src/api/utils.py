# src/api/utils.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.default_class_for_col = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X_df.columns:
            ser = X_df[col].astype(str).fillna('nan')
            le = LabelEncoder()
            le.fit(ser)
            self.encoders[col] = le
            most_freq = ser.mode().iloc[0] if not ser.mode().empty else le.classes_[0]
            self.default_class_for_col[col] = most_freq
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        out_cols = []
        for col in X_df.columns:
            if col not in self.encoders:
                raise ValueError(f"Column '{col}' was not seen during fit.")
            
            le, default = self.encoders[col], self.default_class_for_col[col]
            known_classes = set(le.classes_)
            
            ser = X_df[col].astype(str).fillna('nan').apply(lambda x: x if x in known_classes else default)
            out_cols.append(le.transform(ser).reshape(-1, 1))

        return np.hstack(out_cols)