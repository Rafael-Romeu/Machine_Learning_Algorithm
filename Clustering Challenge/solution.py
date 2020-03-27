def changeToInt(data):
    i = 0
    while(i < len(data)):
        j = 0
        while(j < len(data[0])):
            data[i][j] = hash(data[i][j])
            j += 1
        i += 1
    return data

import pandas as pd
import numpy  as np

trainData = pd.read_csv('./TRAIN.csv')
testData  = pd.read_csv('./TEST.csv')

#allColumns = ['race',	'gender',	'age',	'weight',	'admission_type_id',	'discharge_disposition_id',	'admission_source_id',	'time_in_hospital',	'payer_code',	'medical_specialty',	'num_lab_procedures',	'num_procedures',	'num_medications',	'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1',	'diag_2',	'diag_3',	'number_diagnoses',	'max_glu_serum',	'A1Cresult',	'metformin',	'repaglinide',	'nateglinide',	'chlorpropamide',	'glimepiride',	'acetohexamide',	'glipizide',	'glyburide',	'tolbutamide',	'pioglitazone',	'rosiglitazone',	'acarbose',	'miglitol',	'troglitazone',	'tolazamide',	'examide',	'citoglipton',	'insulin',	'glyburide-metformin',	'glipizide-metformin',	'glimepiride-pioglitazone',	'metformin-rosiglitazone',	'metformin-pioglitazone',	'change',	'diabetesMed',	'readmitted_NO']
columns = ['race',	'gender',	'age',	'admission_type_id',	'discharge_disposition_id',	'admission_source_id',	'time_in_hospital',	'num_lab_procedures',	'num_procedures',	'num_medications',	'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1',	'diag_2',	'diag_3',	'number_diagnoses',	'metformin',	'repaglinide',	'nateglinide',	'chlorpropamide',	'glimepiride',	'acetohexamide',	'glipizide',	'glyburide',	'tolbutamide',	'pioglitazone',	'rosiglitazone',	'acarbose',	'miglitol',	'troglitazone',	'tolazamide',	'examide',	'citoglipton',	'insulin',	'glyburide-metformin',	'glipizide-metformin',	'glimepiride-pioglitazone',	'metformin-rosiglitazone',	'metformin-pioglitazone',	'change',	'diabetesMed',	'readmitted_NO']

    
trainData = pd.DataFrame(trainData, columns = columns)
testData  = pd.DataFrame(testData,  columns  = columns)

trainData = trainData.to_numpy()
trainData = changeToInt(trainData)

testData = testData.to_numpy()
testData = changeToInt(testData)

from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=2, init='random',
    n_init=50, max_iter=5000, 
    tol=1e-04, random_state=42
)
y_km = km.fit(trainData).predict(testData)

# ------ output CSV ------ #
output = pd.DataFrame()
output['target'] = y_km

output.to_csv(r'./prediction3.csv', index = True, index_label = 'index')


