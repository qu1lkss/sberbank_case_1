import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from preprocessor import Preprocessor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
pd.options.display.float_format = '{:,.0f}'.format
pd.set_option('display.max_columns', None)
df = pd.read_csv('train.csv')
macro = pd.read_csv('macro.csv')
df = df.merge(macro, on='timestamp', how='left')

drop_cols = ['floor', 'build_year', 'max_floor', 'raion_popul', 'water_km', 'school_education_centers_top_20_raion', 'university_top_20_raion',
                      'culture_objects_top_25_raion', 'full_all',
                      'raion_build_count_with_material_info', 'build_count_block', 'build_count_frame', 'build_count_monolith', 'build_count_foam',
                      'build_count_1946-1970', 'build_count_1971-1995', 'ID_metro', 'water_treatment_km', 'cemetery_km', 'incineration_km',
                      'ID_railroad_station_walk', 'ID_railroad_station_avto', 'ID_big_road1', 'big_road2_km', 'ID_big_road2','ID_railroad_terminal',
                      'ID_bus_terminal', 'big_market_km', 'church_synagogue_km', 'cafe_sum_1000_min_price_avg', 'cafe_count_1000_price_high',
                      'market_count_1000', 'cafe_sum_1500_min_price_avg', 'cafe_sum_2000_min_price_avg', 'mosque_count_2000',
                      'cafe_sum_3000_min_price_avg', 'mosque_count_3000', 'cafe_sum_5000_min_price_avg', 'mosque_count_5000', 'timestamp',
                      'thermal_power_plant_raion', 'water_1line', 'id']

pp = (Preprocessor(df)
          .basic_clean()
          .fill_num_median()
          .drop_high_corr(threshold=0.75)
          .drop_low_variance(threshold=0.05)
          .add_time_parts()
          .feature_engineering()
          .drop_columns(drop_cols)
          .encode_cats(target_col='price_doc', small_cardinality=6)
          .fill_num_median())

df = pp.transform()

top_q = df['price_doc'].quantile(0.95)
low_q = df['price_doc'].quantile(0.05)
df = df[(df['price_doc'] > low_q) & (df['price_doc'] < top_q)]

X = df.drop('price_doc', axis=1)
y = df['price_doc']

tscv = TimeSeriesSplit(n_splits=5)

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=3000, tol=1e-3, random_state=42)),
    ])

params = {'lasso__alpha': [0.02, 0.005, 0.1, 0.2, 0.5]}

rmse_tr_all, rmse_te_all, best_alphas = [], [], []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    gs = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)

    best_alpha = gs.best_params_['lasso__alpha']
    best_alphas.append(best_alpha)

    preds_tr = gs.predict(X_train)
    preds_te = gs.predict(X_test)

    rmse_tr = np.sqrt(((preds_tr - y_train) ** 2).mean())
    rmse_te = np.sqrt(((preds_te - y_test) ** 2).mean())

    rmse_tr_all.append(rmse_tr)
    rmse_te_all.append(rmse_te)

    print(f'Fold {fold+1}: best alpha={best_alpha}, RMSE train={rmse_tr:.4f}, RMSE test={rmse_te:.4f}')

print(f'\nMean RMSE train: {sum(rmse_tr_all)/len(rmse_tr_all):.4f}')
print(f'Mean RMSE test : {sum(rmse_te_all)/len(rmse_te_all):.4f}')
print(f'Best alphas by fold: {best_alphas}')