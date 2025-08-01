
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, scale_method: Optional[str] = 'standard', n_components: Optional[int] = None):
        self.scale_method = scale_method
        self.n_components = n_components
        
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        if self.scale_method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scale_method == 'robust':
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = None
        X_scaled = self.scaler_.fit_transform(X) if self.scaler_ is not None else X.values
        self.pca_ = PCA(n_components=self.n_components).fit(X_scaled)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler_.transform(X) if self.scaler_ is not None else X.values
        X_pca = self.pca_.transform(X_scaled)
        columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=columns, index=X.index)
        
        