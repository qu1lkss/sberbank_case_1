import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._refresh()

    def _refresh(self):
        self.num_features = [c for c in self.df.columns if self.df[c].dtype != 'object' and c != 'price_doc']
        self.cat_features = [c for c in self.df.columns if self.df[c].dtype == 'object']

    def basic_clean(self):
        df = self.df
        df.loc[(df['full_sq'] < 10) | (df['full_sq'] > 500), 'full_sq'] = np.nan
        df.loc[(df['life_sq'] < 10) | (df['life_sq'] > 500), 'life_sq'] = np.nan
        df.loc[(df['kitch_sq'] < 1) | (df['kitch_sq'] > 100), 'kitch_sq'] = np.nan
        df.loc[(df['floor'] < 1) | (df['floor'] > 60), 'floor'] = np.nan
        df.loc[(df['max_floor'] < 1) | (df['max_floor'] > 60), 'max_floor'] = np.nan
        df.loc[(df['build_year'] < 1800) | (df['build_year'] > 2016), 'build_year'] = np.nan
        df.loc[(df['num_room'] <= 0) | (df['num_room'] > 10), 'num_room'] = np.nan
        if 'area_m' in df.columns:
            df.loc[df['area_m'] > 100000, 'area_m'] = np.nan
        self._refresh()
        return self

    def fill_num_median(self):
        drop = []
        for col in self.num_features:
            if self.df[col].notna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                drop.append(col)
        if drop:
            self.df = self.df.drop(columns=drop)
            self._refresh()
        return self

    def drop_high_corr(self, threshold=0.75):
        corr = self.df[self.num_features].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        if to_drop:
            self.df = self.df.drop(columns=to_drop)
            self._refresh()
        return self

    def drop_low_variance(self, threshold=0.05):
        self._refresh()
        X = self.df[self.num_features]
        if X.shape[1] == 0:
            return self
        cutter = VarianceThreshold(threshold=threshold)
        cutter.fit(X)
        keep = list(cutter.get_feature_names_out(self.num_features))
        drop = [c for c in self.num_features if c not in keep]
        if drop:
            self.df = self.df.drop(columns=drop)
            self._refresh()
        return self

    def add_time_parts(self):
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['year'] = self.df['timestamp'].dt.year
            self.df['month'] = self.df['timestamp'].dt.month.astype('object')
            self.df['quarter'] = self.df['timestamp'].dt.quarter.astype('object')
            self._refresh()
        return self

    def feature_engineering(self):
        df = self.df
        if {'floor','max_floor'}.issubset(df.columns):
            df['floor_ratio'] = df['floor'] / df['max_floor']
        if 'material' in df.columns:
            df['material'] = df['material'].astype('object')
        if 'state' in df.columns:
            df['state'] = df['state'].astype('object')
        if 'build_year' in df.columns:
            df['house_age'] = 2016 - df['build_year']
        if 'raion_popul' in df.columns:
            df['raion_popul_log'] = np.log1p(df['raion_popul'])

        def to_bin(col, cond):
            if col in df.columns:
                df[col] = cond(df).astype(int)

        to_bin('has_top_school', lambda _df: (_df['school_education_centers_top_20_raion'] > 0))
        to_bin('has_top_university', lambda _df: (_df['university_top_20_raion'] > 0))
        to_bin('has_top25_culture_object', lambda _df: (_df['culture_objects_top_25_raion'] > 0))
        to_bin('close_to_water', lambda _df: (_df['water_km'] < 0.5))

        for c in ['incineration_raion','oil_chemistry_raion','radiation_raion','railroad_terminal_raion',
                  'big_market_raion','nuclear_reactor_raion','detention_facility_raion',
                  'big_road1_1line','railroad_1line']:
            if c in df.columns:
                df[c] = (df[c] == 'yes').astype(int)

        if 'ecology' in df.columns:
            mapping = {'no data': 0, 'poor': 1, 'satisfactory': 2, 'good': 3, 'excellent': 4}
            df['ecology'] = df['ecology'].map(mapping)
            mode_val = df['ecology'].mode().iloc[0]
            df['ecology'] = df['ecology'].fillna(mode_val).astype(int)

        for c in ['child_on_acc_pre_school','modern_education_share','old_education_build_share']:
            if c in df.columns:
                s = df[c].astype(str).replace({'#!': np.nan}).str.replace(',', '.', regex=False)
                df[c] = pd.to_numeric(s, errors='coerce')
        self._refresh()
        return self

    def drop_columns(self, cols):
        exist = [c for c in cols if c in self.df.columns]
        if exist:
            self.df = self.df.drop(columns=exist)
            self._refresh()
        return self

    def encode_cats(self, target_col='price_doc', small_cardinality=6):
        self._refresh()
        for col in list(self.cat_features):
            if self.df[col].nunique() <= small_cardinality:
                ohe = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(columns=[col]), ohe], axis=1)
            else:
                mte = self.df.groupby(col)[target_col].mean()
                self.df[col] = self.df[col].map(mte)
        self._refresh()
        return self

    def transform(self):
        return self.df