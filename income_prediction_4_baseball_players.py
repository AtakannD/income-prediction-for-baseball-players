import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r"C:\Users\atakan.dalkiran\PycharmProjects\Income Predict for Baseball Players in '86\hitters.csv")

# Task 1


def check_df(dataframe, head=10):
    """
    This function gives us a first look when we import our dataset.

    Parameters
    ----------
    dataframe: pandas.dataframe
    It is the dataframe from which variable names are wanted to be taken.
    head: int
    The variable that determines how many values we want to look at beginning

    Returns
    -------
    shape: tuple
    that variable gives us to dataset's information which how many columns and rows have
    type: pandas.series
    that variable gives us to our variables' types.
    columns: pandas.Index
    gives us the names of the columns in the dataset.
    head: pandas.Dataframe
    It gives us the variables and values of our dataset, starting from the zero index to the number we entered, as a dataframe.
    tail: pandas.Dataframe
    Contrary to head, this method counts us down starting from the index at the end.
    isnull().sum(): pandas.series
    It visits the variables in the data set and checks if there are any null values and gives us the statistics of how
    many of them are in each variable.
    quantile: pandas.dataframe
    It gives the range values of the variables in our data set as a percentage according to the values we entered.

    Examples
    --------
    The shape return output is given to us as a tuple (5000, 5).
    """
    print("######################### Shape #########################")
    print(dataframe.shape)
    print("\n######################### Type #########################")
    print(dataframe.dtypes)
    print("\n######################## Columns ########################")
    print(dataframe.columns)
    print("\n######################### Head #########################")
    print(dataframe.head(head))
    print("\n######################### Tail #########################")
    print(dataframe.tail(head))
    print("\n######################### NA #########################")
    print(dataframe.isnull().sum())
    print("\n######################### Quantiles #########################")
    print(dataframe.quantile([0, 0.25, 0.5, 0.75, 0.95, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of the categorical, numerical and cardinal variables in the dataset.

    Parameters
    ----------
    dataframe: dataframe
        It is the dataframe with variable names.
    cat_th: int, float
        Returns the names of variables that are numerical as well as categorical.
    car_th: int, float
        Returns the names of categorical but as well as cardinal variables

    Returns
    -------
    cat_cols: list
        list of categorical variables
    num_cols: list
        list of numerical variables
    cat_but_car: list
        list of categorical but as well as cardinal variables

    Notes
    -----
    cat_cols + num_cols + cat_but_car = sum of variables
    num_but_cat in the cat_cols

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Salary")


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col)
df.groupby("Salary").agg({"AtBat": "mean",
                          "Hits": "mean",
                          "HmRun": "mean",
                          "Runs": "mean",
                          "RBI": "mean",
                          "Walks": "mean",
                          "Years": "mean",
                          "CAtBat": "mean",
                          "CHits": "mean",
                          "CHmRun": "mean",
                          "CRuns": "mean",
                          "CRBI": "mean",
                          "CWalks": "mean",
                          "PutOuts": "mean",
                          "Assists": "mean",
                          "Errors": "mean",
                          "Salary": "mean"})
df.isnull().sum()
df.corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Map", fontsize=20)
plt.show()

# Task 2


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True)
df.dropna(inplace=True)

# Task 2

new_num_cols = [col for col in num_cols if col != "Salary"]
df[new_num_cols] = df[new_num_cols] + 0.00000000001

df["NEW_Hits"] = df["Hits"] / df["CHits"]
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_Walks"] = df["Walks"] / df["CWalks"]
df["NEW_Putouts"] = df["PutOuts"] * df["Years"]
df["Hit_Success"] = (df["Hits"] / df["AtBat"]) * 100
df["CHit_Success"] = (df["CHits"] / df["CAtBat"]) * 100
df["Season_Hit_Success_Ratio"] = (df["Hits"] / df["AtBat"]) / (df["CHits"] / df["CAtBat"])
df["NEW_CRBI*CATBAT"] = df["CRBI"] * df["CAtBat"]
df["NEW_Chits"] = df["CHits"] / df["Years"]
df["NEW_CHmRuns"] = df["CHmRun"] * df["Years"]
df["NEW_CRuns"] = df["CRuns"] / df["Years"]
df["NEW_RW"] = df["RBI"] * df["Walks"]
df["NEW_RWALK"] = df["RBI"] / df["Walks"]
df["NEW_CH_CB"] = df["CHits"] * df["CAtBat"]
df["NEW_CHm_CAT"] = df["CHmRun"] / df["CAtBat"]
df["NEW_Diff_AtBat"] = df["AtBat"] - (df["CAtBat"] / df["Years"])
df["NEW_Diff_Hits"] = df["Hits"] - (df["CHits"] / df["Years"])
df["NEW_Diff_HmRun"] = df["HmRun"] - (df["CHmRun"] / df["Years"])
df["NEW_Diff_Runs"] = df["Runs"] - (df["CRuns"] / df["Years"])
df["NEW_Diff_RBI"] = df["RBI"] - (df["CRBI"] / df["Years"])
df["NEW_Diff_Walks"] = df["Walks"] - (df["CWalks"] / df["Years"])
df["Player_Importance"] = df["PutOuts"] + df["Assists"] - df["Errors"]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

# Task 3 - Creating Models

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ("KNN", KNeighborsRegressor()),
          ("CART", DecisionTreeRegressor()),
          ("RF", RandomForestRegressor()),
          ("SVR", SVR()),
          ("GBM", GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")

# GBM: 240
# XGBoost: 244
# LightGBM: 265
# CatBoost: 243

# Go with random forest methods for Hyperparameter Optimization

# 1. Random Forest

rf_model = RandomForestRegressor(random_state=42)

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=42).fit(X, y)
rf_rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse = 238.9

# 2. GBM

gbm_model = GradientBoostingRegressor(random_state=42)
gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.05, 0.5, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [0.5, 0.7, 1]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=42).fit(X, y)
gbm_rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse = 234.9

# 3. XGBoosting

xgb_model = XGBRegressor(random_state=42)
xgb_model.get_params()

xgb_params = {"learning_rate": [0.001, 0.01, 0.1],
              "max_depth": [5, 8, 11],
              "n_estimators": [100, 500, 1000],
              "colsample_bytree": [0.5, 0.7, 1]}

xgb_best_grid = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgb_best_grid.best_params_

xgb_final = xgb_model.set_params(**xgb_best_grid.best_params_, random_state=42).fit(X, y)
xgb_rmse = np.mean(np.sqrt(-cross_val_score(xgb_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse = 239.3

# 4. LightGBM

lgbm_model = LGBMRegressor(random_state=42)
lgbm_model.get_params()


lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
               "max_depth": [5, 8, 11],
               "n_estimators": [100, 300, 500, 1000, 5000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgb_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=42).fit(X, y)
lgbm_rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse = 259.4

# 5. CatBoost

cb_model = CatBoostRegressor()

cb_params = {"iterations": [200, 500, 800],
             "learning_rate": [0.001, 0.01, 0.1],
             "depth": [3, 6]}

cb_best_grid = GridSearchCV(cb_model, cb_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
cb_final = cb_model.set_params(**cb_best_grid.best_params_, random_state=42).fit(X, y)
cb_rmse = np.mean(np.sqrt(-cross_val_score(cb_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# rmse=235.8


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgb_final, X)
plot_importance(lgbm_final, X)
plot_importance(cb_final, X)

# Analyzing Model Complexity


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Test Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_model = RandomForestRegressor(random_state=42)

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1], scoring="neg_mean_absolute_error")
