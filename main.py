import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as pyplot
import pandas as pd
import seaborn as seaborn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2

data = pandas.read_csv('data/X_test.csv')
data = data[['Sex', 'Equipment', 'Age', 'BodyweightKg', 'BestSquatKg']]
data = data.dropna()

data['Sex'] = data['Sex'].replace(['M', 'F'], [0, 1])
data['Equipment'] = data['Equipment'].fillna('')
data['Equipment'] = data['Equipment'].str.contains('Raw', case=False, regex=True).astype(int)

X = data[['Sex', 'Equipment', 'Age', 'BodyweightKg']]  # sex, equipment, age, weight
Y = data['BestSquatKg']  # squat

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

best_features = SelectKBest(score_func=f_regression, k=3)
fit = best_features.fit(X_train, Y_train)

selected_features_indices = best_features.get_support(indices=True)
selected_features = X_train.columns[selected_features_indices]
print("Selected Features:", selected_features)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
features_scores.sort_values(by = 'Score')

print(features_scores)
