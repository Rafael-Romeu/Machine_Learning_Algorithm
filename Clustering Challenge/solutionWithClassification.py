def changeToInt(data):
    i = 0
    while(i < len(data)):
        j = 0
        while(j < len(data[0])):
            data[i][j] = hash(data[i][j])
            j += 1
        i += 1
    return data

def clustering(trainData, testData):
    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=2, init='random',
        n_init=50, max_iter=5000, 
        tol=1e-04, random_state=42
    )
    return km.fit(trainData).predict(testData)

def splitRowMissingField(data, field):
    predictableData = data[ data[field].eq(hash("?"))]
    data = data[~data[field].eq(hash("?"))]
    return data, predictableData

# ------ output CSV ------ #
def outputFunction(data):
    output = pd.DataFrame()
    output['target'] = data
    output.to_csv(r'./prediction5.csv', index = True, index_label = 'index')
    
def predictValues(data, length):
    
    i = 0
    while( i < length):
        trainTable, testTable = splitRowMissingField(data, i)
        if not testTable.empty:
            labelTable = trainTable[[i]].astype(int)

            featuresTable = trainTable.drop(columns = [i]).astype(int)

            featuresTestTable = testTable.drop(columns = [i]).astype(int)

            from sklearn.tree import DecisionTreeClassifier

            tree = DecisionTreeClassifier()
            tree.fit(featuresTable, labelTable)
            yPred = tree.predict(featuresTestTable)

            featuresTestTable[i] = yPred

            featuresTable[i] = labelTable

            data = pd.concat([featuresTable, featuresTestTable], ignore_index=False)
            data.sort_index(inplace=True)
            data.sort_index(axis=1, inplace=True)
        i += 1
    
    return data
    
import pandas as pd
import numpy  as np

trainData = pd.read_csv('./TRAIN.csv')
testData  = pd.read_csv('./TEST.csv')

columns = [	'gender',	'age',	'weight',	'admission_type_id',	'discharge_disposition_id',	'admission_source_id',	'time_in_hospital',	'payer_code',	'num_lab_procedures',	'num_procedures',	'num_medications',	'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1',	'diag_2',	'diag_3',	'number_diagnoses',	'max_glu_serum',	'A1Cresult',	'metformin',	'repaglinide',	'nateglinide',	'chlorpropamide',	'glimepiride',	'acetohexamide',	'glipizide',	'glyburide',	'tolbutamide',	'pioglitazone',	'rosiglitazone',	'acarbose',	'miglitol',	'troglitazone',	'tolazamide',	'examide',	'citoglipton',	'insulin',	'glyburide-metformin',	'glipizide-metformin',	'glimepiride-pioglitazone',	'metformin-rosiglitazone',	'metformin-pioglitazone',	'change',	'diabetesMed',	'readmitted_NO']
# columns = ['race', 'gender',	'age',	'admission_type_id',	'discharge_disposition_id',	'admission_source_id',	'time_in_hospital',	'num_lab_procedures',	'num_procedures',	'num_medications',	'number_outpatient',	'number_emergency',	'number_inpatient',	'diag_1',	'diag_2',	'diag_3',	'number_diagnoses',	'metformin',	'repaglinide',	'nateglinide',	'chlorpropamide',	'glimepiride',	'acetohexamide',	'glipizide',	'glyburide',	'tolbutamide',	'pioglitazone',	'rosiglitazone',	'acarbose',	'miglitol',	'troglitazone',	'tolazamide',	'examide',	'citoglipton',	'insulin',	'glyburide-metformin',	'glipizide-metformin',	'glimepiride-pioglitazone',	'metformin-rosiglitazone',	'metformin-pioglitazone',	'change',	'diabetesMed',	'readmitted_NO']

trainData = pd.DataFrame(trainData, columns = columns)
testData  = pd.DataFrame(testData,  columns  = columns)

trainData = trainData.to_numpy()
trainData = changeToInt(trainData)

testData = testData.to_numpy()
testData = changeToInt(testData)

trainData = pd.DataFrame(trainData)
testData  = pd.DataFrame(testData)

trainData = predictValues(trainData, len(columns))
testData = predictValues(testData, len(columns))

data = clustering(trainData, testData)
outputFunction(data)