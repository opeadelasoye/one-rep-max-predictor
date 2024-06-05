import pandas as pd
import pickle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


predict_variable = 'BestSquatKg'  # BestSquatKg or BestDeadliftKg
participant_data = 'data/participants-data.csv'


def preprocess_data(data):
    data['Sex'] = data['Sex'].replace(['M', 'F'], [0, 1])
    data['Equipment'] = data['Equipment'].replace(['Raw', 'Wraps', 'Single-ply', 'Multi-ply'], [0, 1, 2, 3])

    data[predict_variable] = pd.to_numeric(data[predict_variable], errors='coerce')
    data = data[data[predict_variable] > 0]

    data.reset_index(drop=True, inplace=True)

    return data


test_data = pd.read_csv('data/X_test.csv')
train_data = pd.read_csv('data/X_train.csv')

test_data = test_data.drop_duplicates(subset=['Name', 'Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg', 'BestDeadliftKg'], keep='first')
train_data = train_data.drop_duplicates(subset=['Name', 'Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg', 'BestDeadliftKg'], keep='first')

test_data = test_data[['Sex', 'Equipment', 'Age', 'BodyweightKg', predict_variable]]
train_data = train_data[['Sex', 'Equipment', 'Age', 'BodyweightKg', predict_variable]]

test_data = test_data.dropna()
train_data = train_data.dropna()

test_data = preprocess_data(test_data)
train_data = preprocess_data(train_data)

X_test = test_data[['Sex', 'Equipment', 'Age', 'BodyweightKg']]
Y_test = test_data[predict_variable]
X_train = train_data[['Sex', 'Equipment', 'Age', 'BodyweightKg']]
Y_train = train_data[predict_variable]

best_features = SelectKBest(score_func=f_regression, k=3)
fit = best_features.fit(X_train, Y_train)

selected_features_indices = best_features.get_support(indices=True)
selected_features = X_train.columns[selected_features_indices]

data_scores = pd.DataFrame(fit.scores_)
data_columns = pd.DataFrame(X_train.columns)

features_scores = pd.concat([data_columns, data_scores], axis=1)
features_scores.columns = ['Features', 'Score']
features_scores.sort_values(by='Score', ascending=False, inplace=True)

# print(features_scores)

dt_reg = DecisionTreeRegressor(random_state=100)
dt_reg.fit(X_train, Y_train)

Y_pred = dt_reg.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = root_mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
"""
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)
"""
with open('decision_tree_regression_model.pkl', 'wb') as file:
    pickle.dump(dt_reg, file)

with open('decision_tree_regression_model.pkl', 'rb') as file:
    dt_reg_loaded = pickle.load(file)

new_data = pd.read_csv(participant_data)

X_new = new_data[['Sex', 'Equipment', 'Age', 'BodyweightKg']]
Y_pred_new = dt_reg_loaded.predict(X_new)

print(X_new)
print(Y_pred_new)
