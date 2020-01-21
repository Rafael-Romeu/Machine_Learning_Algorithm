# ------ Functions ------ #
def cleanDataFeatures(features):
    features['Sex'] = [sex.replace('female', '0') for sex in features['Sex']]
    features['Sex'] = [sex.replace('male', '1') for sex in features['Sex']]
    features['Sex'] = features['Sex'].astype(int)

    i = 0
    for emb in features['Embarked']:
        if emb is 'S':
            features['Embarked'][i] = '0'
        elif emb is 'C':
            features['Embarked'][i] = '1'
        elif emb is 'Q':
            features['Embarked'][i] = '2'
        else:
            features['Embarked'][i] = '-1'
        i += 1

    features['Embarked'] = features['Embarked'].astype(int)

    features.fillna(0, inplace = True)
    return features


# ------ Main Code ------ #
import numpy as np
import pandas as pd

trainData= pd.read_csv('./train.csv')
testData = pd.read_csv('./test.csv')

features = pd.DataFrame(trainData, columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'])
target   = pd.DataFrame(trainData, columns = ['Survived'])

testFeatures = pd.DataFrame(testData, columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'])

features = cleanDataFeatures(features)
testFeatures = cleanDataFeatures(testFeatures)

# ------ Split Data ------ #
# from sklearn.model_selection import train_test_split
# xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2)

# ------ K-Nearest Neighbors Classificator ------ #
# from sklearn.neighbors import KNeighborsClassifier 

# knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
# knn.fit(xTrain, yTrain)
# yPred = knn.predict(xTest)

# ------ Decision Tree Classificator ------ #
# from sklearn.tree import DecisionTreeClassifier

# tree = DecisionTreeClassifier()
# tree.fit(xTrain, yTrain)
# yPred = tree.predict(xTest)

# ------ Logistic Regression ------ #
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(features, target)
yPred = lr.predict(testFeatures)

# ------ output CSV ------ #
output = pd.DataFrame(testData, columns = ['PassengerId'])
output['Survived'] = yPred

output.to_csv(r'./prediction.csv', index = False)


# ------ Metrics ------ #
# from sklearn.metrics import confusion_matrix, f1_score

# confusion_matrix = confusion_matrix(yTest, yPred)
# f1_score         = f1_score(yTest, yPred, average='weighted')

# print("Confusion Matrix: \n", confusion_matrix)
# print("F1-Score: ", f1_score)