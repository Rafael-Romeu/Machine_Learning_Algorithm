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
    
    # meanAge = ( features['Age'].mean( skipna = True) + testFeatures['Age'].mean( skipna = True) ) / 2
    # features['Age'] = features['Age'].fillna(meanAge)
    
    return features

def changeToInt(data):
    i = 0
    while(i < len(data)):
        j = 0
        while(j < len(data[0])):
            data[i][j] = hash(data[i][j])
            j += 1
        i += 1
    return data

# ------ Split Data ------ #
def splitData(features, target):
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(features, target, test_size=0.2)
    return xTrain, xTest, yTrain, yTest

# ------ K-Nearest Neighbors Classificator ------ #
def KNN(xTrain, xTest, yTrain):
    from sklearn.neighbors import KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)
    return yPred

# ------ Decision Tree Classificator ------ #
def DTree(xTrain, yTrain, xTest):
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier()
    tree.fit(xTrain, yTrain)
    yPred = tree.predict(xTest)
    return yPred

# ------ Logistic Regression ------ #
def LogRegression(xTrain, yTrain, xTest):
        
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(random_state=0)
    lr.fit(features, target)
    yPred = lr.predict(testFeatures)
    return yPred

# ------ output CSV ------ #
def output(testData, yPred):
    output = pd.DataFrame(testData, columns = ['PassengerId'])
    output['Survived'] = yPred
    output.to_csv(r'./prediction2.csv', index = False)
    
# ------ Metrics ------ #
def metrics(yTest, yPred):
    from sklearn.metrics import confusion_matrix, f1_score
    confusion_matrix = confusion_matrix(yTest, yPred)
    f1_score         = f1_score(yTest, yPred, average='weighted')
    print("Confusion Matrix: \n", confusion_matrix)
    print("F1-Score: ", f1_score)
    
def neuralNet(xTrain, xTest, yTrain, yTest):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(600, activation='sigmoid'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(300, activation='sigmoid'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(1, activation='sigmoid')
        # tf.keras.layers.Dense(126, activation='relu'),
        # tf.keras.layers.Dense(36, activation='relu'),
        # tf.keras.layers.Dropout(0.2),        
        # tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=1000, batch_size=8)
    # model.evaluate(xTest, yTest, verbose=2)
    return model.predict(xTest)
    

# ------ Main Code ------ #
import numpy as np
import pandas as pd

trainData= pd.read_csv('./train.csv')
testData = pd.read_csv('./test.csv')


columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked' ]
# columns = ['Pclass'	,'Sex'	,'Age'	,'SibSp','Parch','Fare','Embarked']


features = pd.DataFrame(trainData, columns = columns)
target   = pd.DataFrame(trainData, columns = ['Survived'])

testFeatures = pd.DataFrame(testData, columns = columns)

print(target)

features = cleanDataFeatures(features)
testFeatures = cleanDataFeatures(testFeatures)

xTrain, xTest, yTrain, yTest = splitData(features, target)

yPred = neuralNet(features.to_numpy(), testFeatures.to_numpy(), target.to_numpy(), yTest.to_numpy())
# neuralNet(features.to_numpy(), features.to_numpy(), target.to_numpy(), target.to_numpy())

predictions = []
i = 0
for x in yPred:
    if(x > 0.5):
        predictions.insert(i, 1)
    else:
        predictions.insert(i, 0)    
    i += 1
    
print(predictions)
output(testData, predictions)
