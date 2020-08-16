# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:18:47 2020

@author: hp
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
diabetesDF = pd.read_csv("H:\STUDY MATERIALS\STUDYFILE\cse299\data\diabetes.csv")
print(diabetesDF.head())

diabetesDF.info()

corr = diabetesDF.corr()
print(corr)
sns.heatmap(corr, 
         xticklabels=corr.columns, 
         yticklabels=corr.columns)
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1
diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)
accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

from sklearn.model_selection import train_test_split
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness' ,	'Insulin' ,	'BMI' ,	'DiabetesPedigreeFunction' ,	'Age']
predicted_class = ['Outcome']


X = data[feature_columns].values
y = data[predicted_class].values

#coeff = list(diabetesCheck.coef_[0])
#labels = list(X)
#features = pd.DataFrame()
#features['Features'] = labels
#features['importance'] = coeff
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features['positive'] = features['importance'] > 0
##features.set_index('Features', inplace=True)
##features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
##plt.xlabel('Importance')

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")
print(dfCheck.head())
sampleData = dfCheck[:1]
# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)
 