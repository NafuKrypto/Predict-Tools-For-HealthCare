import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import pickle

'exec(%matplotlib inline)'
from sklearn.externals import joblib

data = pd.read_csv("H:\STUDY MATERIALS\STUDYFILE\cse299\data\diabetes.csv")
 

data.shape
data.head(5)
data.isnull().values.any()
## Correlation
import seaborn as sns
#import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

data.corr()

diabetes_map = {True: 1, False: 0}
data['Outcome'] = data['Outcome'].map(diabetes_map)
data.head(5)

diabetes_true_count = len(data.loc[data['Outcome'] == True])
diabetes_false_count = len(data.loc[data['Outcome'] == False])

(diabetes_true_count,diabetes_false_count)
 
from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness' ,	'Insulin' ,	'BMI' ,	'DiabetesPedigreeFunction' ,	'Age']
predicted_class = ['Outcome']


X = data[feature_columns].values
y = data[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)


print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['Glucose'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['BloodPressure'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['Insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['BMI'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['DiabetesPedigreeFunction'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['Age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['SkinThickness'] == 0])))

from sklearn.preprocessing import Imputer

fill_values = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

## Hyperparameter optimization using RandomizedSearchCV
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

 
coeff_df = pd.DataFrame(regressor.coef_)  
print(regressor.intercept_)
coeff_df
#Check the difference between the actual value and predicted value.
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test[3] , 'Predicted': y_pred[3]})
df
f1 = df.head(25)
f1
 
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y.ravel(),cv=10)

score

score.mean()

# Saving model to disk
pickle.dump(random_forest_model, open('modelrandomforest.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modelrandomforest.pkl','rb'))

 

joblib.dump( y_pred, 'diabeteseModel1.pkl')
diabetesLoadedModel=joblib.load('diabeteseModel1.pkl')
##accuracyModel = diabetesLoadedModel.score(X_test,y_test)

print( model.predict(X_test))
 