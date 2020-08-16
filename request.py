# -*- coding: utf-8 -*-
 
import requests

url = 'http://localhost:5000/results'
url1 = 'http://localhost:5000/results_bc'
r = requests.post(url,json={['Pregnancies':1, 'Glucose':148, 'BloodPressure':72, 'SkinThickness':31 ,	'Insulin': 88,	'BMI':25.6 ,	'DiabetesPedigreeFunction':1.567 ,	'Age':60]})

print(r.json())


r1 = requests_bc.post(url1,json={[ 'radius_mean':12.46 ,	'texture_mean':	24.04,'perimeter_mean':83.97,	'area_mean':475.9,	
                                'smoothness_mean':0.1186,	'compactness_mean':0.2396,	'concavity_mean':0.2273,
                                'concave points_mean':0.08543,'symmetry_mean':0.203,	'fractal_dimension_mean':0.08243,	
                                'radius_se':0.2976,	'texture_se':1.599	,	
                                'perimeter_se':2.039	,	'area_se':23.94	,	'smoothness_se':0.007149,	
                                'compactness_se':0.07217,	'concavity_se':0.07743,	
                                'concave points_se':0.01432	,	'symmetry_se':0.01789,	'fractal_dimension_se':0.01008,	
                                'radius_worst':15.09,	
                                'texture_worst':40.68,	'perimeter_worst':97.65,	'area_worst':711.4,	
                                'smoothness_worst':0.1853,	
                                'compactness_worst':1.058,	'concavity_worst':1.105,	'concave points_worst':0.221,	
                                'symmetry_worst':0.4366,	
                                'fractal_dimension_worst':0.2075]})

print(r1.json())