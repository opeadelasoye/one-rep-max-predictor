import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as pyplot
import pandas as pd
import seaborn as seaborn
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

data = pandas.read_csv('data/X_test.csv')
data = data[['Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg']]
data = data.dropna()

data['Sex'] = data['Sex'].replace(['M', 'F'], [0, 1])
data['Equipment'] = data['Equipment'].fillna('')
data['Equipment'] = data['Equipment'].str.contains('Raw', case=False, regex=True).astype(int)

X = data[['Sex', 'Equipment', 'Age', 'BodyweightKg']]  # sex, equipment, age, weight
Y = data['BestSquatKg']  # squat

best_features = SelectKBest(score_func=f_regression, k=3)
fit = best_features.fit(X, Y)

selected_features_indices = best_features.get_support(indices=True)
selected_features = X.columns[selected_features_indices]

data_scores = pd.DataFrame(fit.scores_)
data_columns = pd.DataFrame(X.columns)

features_scores = pd.concat([data_columns, data_scores], axis=1)
features_scores.columns = ['Features', 'Score']
features_scores.sort_values(by='Score', ascending=False, inplace=True)

X = data[['Equipment', 'Sex', 'BodyweightKg']]
Y = data['BestSquatKg']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)

Y_pred = linreg.predict(X_test)

with open('output.txt', 'w') as file:
    file.write("Selected Features:\n")
    file.write(str(selected_features) + '\n\n')

    file.write("Feature Scores:\n")
    file.write(str(features_scores) + '\n\n')

    file.write("Test Data and Predicted Squat:\n")
    for index, (test_row, pred_value) in enumerate(zip(X_test.iterrows(), Y_pred)):
        row_index, row_data = test_row
        file.write(
            f"Test Row {row_index} - Equipment: {row_data['Equipment']}, Sex: {row_data['Sex']}, BodyweightKg: {row_data['BodyweightKg']} - Predicted Squat: {pred_value}\n")
    file.write('\n')
