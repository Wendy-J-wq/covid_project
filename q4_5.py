from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

path_fat_quantity = '/content/Fat_Supply_Quantity_Data.csv'
path_food_quantity = '/content/Food_Supply_Quantity_kg_Data.csv'
path_food_kcal = '/content/Food_Supply_kcal_Data.csv'
path_protein_quantity = '/content/Protein_Supply_Quantity_Data.csv'
path_food_description = '/content/Supply_Food_Data_Descriptions.csv'

"""
### Q4: Does there appear to be a relationship
between the number of deaths from
COVID and the amount of intake from fat?
"""

df_fat_quantity = pd.read_csv(path_fat_quantity)
df_fat_quantity.columns = df_fat_quantity.columns.str.replace(' ', '_')
df_fat_quantity.describe()

df_fat_quantity.columns

data = df_fat_quantity[['Alcoholic_Beverages',
                        'Animal_Products', 'Animal_fats',
                        'Aquatic_Products,_Other', 'Cereals_-_Excluding_Beer',
                        'Eggs', 'Fish,_Seafood',
                        'Fruits_-_Excluding_Wine', 'Meat',
                        'Milk_-_Excluding_Butter', 'Miscellaneous',
                        'Offals', 'Oilcrops',
                        'Pulses', 'Spices', 'Starchy_Roots',
                        'Stimulants', 'Sugar_Crops',
                        'Sugar_&_Sweeteners', 'Treenuts',
                        'Vegetal_Products', 'Vegetable_Oils',
                        'Vegetables', 'Deaths']]
data = data.dropna()
# Use the correlation matrix
corr = data.corr()

corr['Deaths']

y_axis_labels = ['Alcoholic_Beverages', 'Animal_Products', 'Animal_fats',
                 'Aquatic_Products,_Other', 'Cereals_-_Excluding_Beer', 'Eggs',
                 'Fish,_Seafood', 'Fruits_-_Excluding_Wine', 'Meat',
                 'Milk_-_Excluding_Butter', 'Miscellaneous',
                 'Offals', 'Oilcrops',
                 'Pulses', 'Spices', 'Starchy_Roots',
                 'Stimulants', 'Sugar_Crops',
                 'Sugar_&_Sweeteners', 'Treenuts',
                 'Vegetal_Products', 'Vegetable_Oils',
                 'Vegetables', 'Deaths']
sns.heatmap(corr['Deaths'].values.reshape(24, 1), cmap="RdBu_r",
            annot=True, yticklabels=y_axis_labels)
plt.ylabel('Type of fat')

sns.heatmap(corr, cmap="RdBu_r", annot=True)
plt.gcf().set_size_inches(15, 8)

sns.pairplot(data, height=1.5, corner=True)

animal_fat = data['Animal_fats'].T
animal_product = data['Animal_Products'].T
cereal = data['Cereals_-_Excluding_Beer'].T
vegetal_products = data['Vegetal_Products'].T
death_r = data['Deaths'].T

new = np.array([animal_fat, animal_product, cereal, vegetal_products, death_r])
new.shape

covMatrix = np.cov(new, bias=True)
print(covMatrix)

y_axis_labels = ['Animal_Products', 'Animal_fats',
                 'Cereals_-_Excluding_Beer', 'Vegetal_Products',
                 'Deaths']
sns.heatmap(covMatrix, annot=True, fmt='g', yticklabels=y_axis_labels)
plt.title('covariance matrix Heapmap')
plt.show()

deaths = data['Deaths']
animal_fat = data['Animal_fats']
df = [deaths, animal_fat]
df = pd.DataFrame(df)
df = df.T

fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=df, x="Deaths", y="Animal_fats")
plt.title('AnimalFat vs Deaths')

country_animal_fat = px.bar(df_fat_quantity, x="Country",
                            y="Animal_fats").update_xaxes()
country_animal_fat.show()

country_animal_fat = px.bar(df_fat_quantity, x="Country",
                            y="Vegetal_Products").update_xaxes()
country_animal_fat.show()

country_death = px.bar(df_fat_quantity, x="Country", y="Deaths").update_xaxes()
country_death.show()

pg.linear_regression(df[["Animal_fats", "Vegetal_Products"]],
                     df["Deaths"])

"""
Q5
"""
# Decision Tree

df_food_quantity = pd.read_csv(path_food_quantity)
df = df_food_quantity
df.columns = df.columns.str.replace(' ', '_')
df.columns

print(df.isnull().sum())

df = df.dropna()

data = df[['Alcoholic_Beverages', 'Animal_fats', 'Animal_Products',
           'Aquatic_Products,_Other', 'Cereals_-_Excluding_Beer', 'Eggs',
           'Fish,_Seafood', 'Fruits_-_Excluding_Wine', 'Meat',
           'Milk_-_Excluding_Butter', 'Miscellaneous', 'Offals', 'Oilcrops',
           'Pulses', 'Spices', 'Starchy_Roots', 'Stimulants',
           'Sugar_&_Sweeteners',
           'Sugar_Crops', 'Treenuts', 'Vegetable_Oils', 'Vegetables',
           'Vegetal_Products', 'Recovered']]

# decision tree regression

X = df[['Alcoholic_Beverages', 'Animal_fats', 'Animal_Products',
        'Aquatic_Products,_Other', 'Cereals_-_Excluding_Beer', 'Eggs',
        'Fish,_Seafood', 'Fruits_-_Excluding_Wine', 'Meat',
        'Milk_-_Excluding_Butter', 'Miscellaneous', 'Offals', 'Oilcrops',
        'Pulses', 'Spices', 'Starchy_Roots', 'Stimulants',
        'Sugar_&_Sweeteners', 'Sugar_Crops', 'Treenuts',
        'Vegetable_Oils', 'Vegetables', 'Vegetal_Products']]
y = df['Recovered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# size of train data set
X_train.shape

# size of test data set
X_test.shape

reg_decision_model = DecisionTreeRegressor()

# fit independent varaibles to the dependent variables
reg_decision_model.fit(X_train, y_train)

reg_decision_model.score(X_train, y_train)

reg_decision_model.score(X_test, y_test)

prediction = reg_decision_model.predict(X_test)

# checking difference between labled y and predicted y
sns.distplot(y_test-prediction)

x = np.linspace(0, 10, 10)
y2 = x
fig, ax = plt.subplots()
ax.scatter(y_test, prediction)
plt.title('prediction vs. ture value')
ax.plot(x, y2, color='red')
ax.legend(['y=x', 'Values'])

print('MSE:', mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))
print('accuracy score:', reg_decision_model.score(X_test, y_test))

"""Hyperparameter tuning"""

# Hyper parameters range intialization for tuning

parameters = {"splitter": ["best", "random"],
              "max_depth": [1, 3, 5, 7, 9, 11, 12],
              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4,
                                           0.5, 0.6, 0.7, 0.8, 0.9],
              "max_features": ["auto", "log2", "sqrt", None],
              "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}

# calculating different regression metrics

tuning_model = GridSearchCV(reg_decision_model,
                            param_grid=parameters,
                            scoring='neg_mean_squared_error',
                            cv=3, verbose=3)

tuning_model.fit(X, y)

# best hyperparameters
tuning_model.best_params_

tuned_hyper_model = DecisionTreeRegressor(max_depth=5, max_features=None,
                                          max_leaf_nodes=40,
                                          min_samples_leaf=5,
                                          min_weight_fraction_leaf=0.1,
                                          splitter='random')

tuned_hyper_model.fit(X_train, y_train)

tuned_pred = tuned_hyper_model.predict(X_test)

x = np.linspace(0, 10, 10)
y2 = x
fig, ax = plt.subplots()
ax.plot(x, y2, color='red')
ax.scatter(y_test, tuned_pred)
ax.legend(['y=x', 'Values'])

# With hyperparameter tuned
print('MSE:', mean_squared_error(y_test, tuned_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, tuned_pred)))
print('accuracy score:', tuned_hyper_model.score(X_test, y_test))

"""Random Forest + hyperparameter tuning"""

n_estimators = [5, 20, 50, 100]  # number of trees in the random forest
max_features = ['auto', 'sqrt']
# number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
# maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10]  # minimum sample number to split a node
min_samples_leaf = [1, 3, 4]
# minimum sample number that can be stored in a leaf node
bootstrap = [True, False]  # method used to sample data points

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=50, cv=3, verbose=2,
                               random_state=0, n_jobs=-1)

rf_random.fit(X_train, y_train)

print('Random grid: ', random_grid, '\n')
# print the best parameters
print('Best Parameters: ', rf_random.best_params_, ' \n')

best_rf = RandomForestRegressor(n_estimators=20, min_samples_split=10,
                                min_samples_leaf=4, max_features='auto',
                                max_depth=40, bootstrap=True)
best_rf.fit(X_train, y_train)

y_pred_rf = best_rf.predict(X_test)
print(y_pred_rf)

x = np.linspace(0, 10, 10)
y2 = x
fig, ax = plt.subplots()
ax.plot(x, y2, color='red')
ax.scatter(y_test, y_pred_rf)
ax.legend(['y=x', 'Values'])

print('MSE:', mean_squared_error(y_test, y_pred_rf))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('accuracy score:', best_rf.score(X_test, y_test))

"""Predict myself"""

myself = [3.5, 0.2212, 12, 0, 8.0666, 0.7792, 3.275, 4.8723, 5.8477, 2.2041,
          4.302, 0.1704, 0.9734, 0.6391, 0.1565, 4.0812, 0.1721, 5.3344,
          0, 0.0852, 0.8677, 5.4725, 37.5167]
myself = pd.DataFrame(myself)

myself.columns = ["Wendy's"]
myself.index = ['Alcoholic_Beverages', 'Animal_fats', 'Animal_Products',
                'Aquatic_Products,_Other', 'Cereals_-_Excluding_Beer', 'Eggs',
                'Fish,_Seafood', 'Fruits_-_Excluding_Wine', 'Meat',
                'Milk_-_Excluding_Butter', 'Miscellaneous',
                'Offals', 'Oilcrops',
                'Pulses', 'Spices', 'Starchy_Roots', 'Stimulants',
                'Sugar_&_Sweeteners', 'Sugar_Crops', 'Treenuts',
                'Vegetable_Oils', 'Vegetables', 'Vegetal_Products']

print(myself)

my_rate = tuned_hyper_model.predict(myself.T)
print("Wendy's Recovery rate: " + str(my_rate) + '%')

# find the median of all countries' recovery rate
medians = y.median()
print("medians of Recovery rate: " + str(medians) + '%')
