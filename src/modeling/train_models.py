import os
from joblib import dump
from utils import load_csv, preprocess
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from ISLP.bart import BART
from tabpfn import TabPFNRegressor
from autogluon.tabular import TabularPredictor

os.makedirs("models", exist_ok=True)

train_path = "data/salary_data_train.csv"
train_df = preprocess(load_csv(train_path))
features = ["age", "gender", "education_level", "job_level", "experience_years", "distance_from_home", "absence", "child"]
X_train = train_df[features].astype(float)
y_train = train_df["income"].astype(float)

# --- Linear Regression ---
lr_model = LinearRegression().fit(X_train, y_train)
dump(lr_model, "models/lr_model.joblib")

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_train)
dump(rf_model, "models/rf_model.joblib")

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, objective="reg:squarederror")
xgb_model.fit(X_train, y_train)
xgb_model.save_model("models/xgb_model.json")

# --- Bayesian Ridge ---
bayes_model = BayesianRidge().fit(X_train, y_train)
dump(bayes_model, "models/bayes_model.joblib")

# --- BART ---
bart_model = BART(num_trees=150, num_particles=8, max_stages=4000, burnin=300, random_state=42, n_jobs=-1)
bart_model.fit(X_train, y_train)
dump(bart_model, "models/bart_model.joblib")

# --- TabPFN ---
tabpfn_model = TabPFNRegressor(random_state=42)
tabpfn_model.fit(X_train, y_train)
dump(tabpfn_model, "models/tabpfn_model.joblib")

# --- AutoGluon ---
predictor = TabularPredictor(label='income', problem_type='regression').fit(train_df)
predictor.save("models/autogluon_model")
