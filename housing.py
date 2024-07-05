import pandas as pd
import matplotlib
import sklearn
import numpy as np

# Fetching the data sets from .csv file
housing = pd.read_csv("housing_data/data.csv")

# print(housing.head())
# print(housing.info())


# Splitting into train and test data 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# print(strat_test_set.head())
housing = strat_train_set.copy()

housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set["MEDV"].copy()
 
# print(housing_labels.head())

# Making the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),])

housing_num_tr = my_pipeline.fit_transform(housing)
# print(housing_num_tr)


# Model Selection and training
# Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# Prediction using some data

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = my_pipeline.transform(some_data)

# print(f"Predictions: {model.predict(some_data_prepared)}")

# -------------Model Evaluation---------------------

# Using Root mean squared error rmse

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# Using Cross-validation Technique
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print_scores(rmse_scores)

# Saving the Mode-----------------------------------
from joblib import dump, load
dump(model, 'boston_Real_State.joblib') 

# Testing on the Test Set----------------------------
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_rmse)

# Using the Model--------------------------------------
model = load('boston_Real_State.joblib')
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
print("Using sample features to test the model")
print(model.predict(features))







