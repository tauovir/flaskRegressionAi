import pandas as pd
import numpy as np

import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#X = X[:, 1:] No need here sklearn take care

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Linear regression accuracy
r2_score(y_test,y_pred)


#===========Test Model with single actual value=======
#we need to transforn categorical data with same object so that pattern will be same
#onehot encoder take care pattern so we just need to provide single pridecting vlaue.
#cateData = np.array([1315.46,115816,297114,'Bangalore']).reshape(1,-1)
#cateData[:,3] = labelencoder.transform(cateData[:,3])
#cateData = onehotencoder.transform(cateData).toarray()
#y_pred1 = regressor.predict(cateData)
#=================End==============
#===================Save Machine Learning Trained Model============
# this is basically serialize the model,we also need encoder object, so that we can use in web
from sklearn.externals import joblib
joblib.dump(regressor,'linearTrainedModel.pkl')
joblib.dump(labelencoder,'labelEncoderObject.pkl')
joblib.dump(onehotencoder,'oneHoteEncoderObject.pkl')

#Now Visualize relationship between features and response using scatter plot
"""
sns.pairplot(dataset,
             x_vars = ['R&D Spend','Administration','Marketing Spend'],
             y_vars = 'Profit',size=7, aspect=0.7, kind = 'reg')
"""
