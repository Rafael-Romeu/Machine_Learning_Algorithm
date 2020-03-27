import pandas as pd
import numpy  as np
import tensorflow as tf

def changeColumn(column, dictionary):
    
    column = column.fillna('-1')
    
    for i, c in enumerate(column):
        for key in dictionary.keys():
            if key in c:
                column.iloc[i] = dictionary[key]
                break
            
    return column

def changeEmbarked(dataset):

    dataset['Embarked'] = dataset['Embarked'].fillna('-1')
    
    embarkedDictionary = {
        'C': 0,
        'S': 1,
        'Q': 2,
        'other': -1,
    }

    for i, emb in enumerate(dataset['Embarked']):
        for key in embarkedDictionary.keys():
            if key in emb:
                dataset.loc[i, 'Embarked'] = embarkedDictionary[key]
                break
            
    return dataset
    
def changeCabin(dataset):
    
    dataset['Cabin'] = dataset['Cabin'].fillna('-1')
    
    cabinDictionary = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'T': 7,
        'other': -1,
    }

    for i, cabins in enumerate(dataset['Cabin']):
        for key in cabinDictionary.keys():
            if key in cabins:
                dataset['Cabin'][i] = cabinDictionary[key]
                break
    dataset['Cabin'] = dataset['Cabin'].astype(int) 

    for i, cabins in enumerate(dataset['Cabin']):
        assert dataset['Cabin'][i] in cabinDictionary.values()          

    return dataset


def changeName(dataset):
    
    nameDictionary = {
        "Capt":       0,
        "Col":        0,
        "Major":      0,
        "Jonkheer":   1,
        "Don":        1,
        "Sir" :       1,
        "Dr":         0,
        "Rev":        0,
        "the Countess":1,
        "Dona":       1,
        "Mme":        2,
        "Mlle":       3,
        "Ms":         2,
        "Mr" :        4,
        "Mrs" :       2,
        "Miss" :      3,
        "Master" :    5,
        "Lady" :      1
    }
    
    for i, name in enumerate(dataset["Name"]):
        for title in nameDictionary.keys():
            if title in name:
                dataset["Name"][i] = nameDictionary[title]
                break
    
        assert dataset["Name"][i] in nameDictionary.values()
    
    return dataset
    
def changeSex(sexColumn):
    newColumn = []
    for i, sex in enumerate(sexColumn):
        if 'male' == sex:
            newColumn.insert(i, 0)
        else:
            newColumn.insert(i, 1)
    return newColumn


def changeTicket(ticketColumn):
    
    newColumn = []
    for i, ticket in enumerate(ticketColumn):
        split = ticket.split(' ')
        if len(split) > 2:
            newColumn.insert(i, hash(split[0]))
        else:
            if len(split) > 1:
                newColumn.insert(i, hash(split[0]))
            else:
                if split[0].isdigit():
                    newColumn.insert(i, hash(split[0]))                                                
                else:
                    newColumn.insert(i, hash(' '))                
    
    return newColumn

def changeAge(ageColumn):
    
    mean = ageColumn.dropna().sum() / len(ageColumn.dropna())
    newColumn = ageColumn.fillna(mean.round(1))
    return newColumn

def changeFare(fareColumn):
    
    mean = fareColumn.dropna().sum() / len(fareColumn.dropna())
    newColumn = fareColumn.fillna(mean.round(1))
    return newColumn

def cleanDataset(dataset):
    
    embarkedDictionary = {
        'C': 0,
        'S': 1,
        'Q': 2,
        'other': -1,
    }
    cabinDictionary = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'T': 7,
        'other': -1,
    }
    nameDictionary = {
        "Capt":       0,
        "Col":        0,
        "Major":      0,
        "Jonkheer":   1,
        "Don":        1,
        "Sir" :       1,
        "Dr":         0,
        "Rev":        0,
        "the Countess":1,
        "Dona":       1,
        "Mme":        2,
        "Mlle":       3,
        "Ms":         2,
        "Mr" :        4,
        "Mrs" :       2,
        "Miss" :      3,
        "Master" :    5,
        "Lady" :      1
    }
    dataset['Embarked'] = changeColumn(dataset['Embarked'], embarkedDictionary)
    dataset['Cabin'] = changeColumn(dataset['Cabin'], cabinDictionary)
    dataset['Name'] = changeColumn(dataset['Name'], nameDictionary)
    
    dataset['Sex'] = changeSex(dataset['Sex'])
    dataset['Ticket'] = changeTicket(dataset['Ticket'])
    dataset['Age'] = changeAge(dataset['Age'])
    dataset['Fare'] = changeFare(dataset['Fare'])
    
    return dataset.astype(float)
    

# ------ K-Nearest Neighbors Classificator ------ #
def KNN(xTrain, yTrain, xTest):
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
    lr.fit(xTrain, target)
    yPred = lr.predict(xTest)
    return yPred

# ------ output CSV ------ #
def output(testData, yPred):
    output = pd.DataFrame(testData, columns = ['PassengerId'])
    output['Survived'] = yPred
    output.to_csv(r'./predictionNN.csv', index = False)
    
# ------ Metrics ------ #
def metrics(yTest, yPred):
    from sklearn.metrics import confusion_matrix, f1_score
    confusion_matrix = confusion_matrix(yTest, yPred)
    f1_score         = f1_score(yTest, yPred, average='weighted')
    print("Confusion Matrix: \n", confusion_matrix)
    print("F1-Score: ", f1_score)
    return f1_score
    
def NN(xTrain, yTrain, xTest):
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dropout(0.1),
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
    return model
    # model.evaluate(xTest, yTest, verbose=2)
    # return model.predict(xTest)

trainData = pd.read_csv('./train.csv')
testData  = pd.read_csv('./test.csv')

target = trainData.pop('Survived')

dataset = pd.concat([trainData, testData], keys=['train', 'test'])
dataset = cleanDataset(dataset)
dataset.pop('Ticket')
# dataset.pop('Cabin')
# dataset.pop('Parch')
# dataset.pop('Fare')
# dataset.pop('Name')
# dataset.pop('PassengerId')
# dataset.pop('SibSp')
# dataset.pop('Embarked')



train = dataset.loc['train']
test  = dataset.loc['test']

# print(np.where(pd.isnull(target)))

# training_dataset = tf.data.Dataset.from_tensor_slices((trainData.values, target.values))
# training_dataset = training_dataset.shuffle(len(trainData)).batch(1)

# testing_dataset  = tf.data.Dataset.from_tensor_slices((testData.values))

# prediction = neuralNetwork(training_dataset, testing_dataset)
test   = np.asarray(test)
train  = np.asarray(train)
target = np.asarray(target)

testeLocal = train[800:891,:]
targetTesteLocal = target[800:891]
trainLocal = train[0:800,:]
targetLocal = target[0:800]

# test   = np.expand_dims(test, -1)
# train  = np.expand_dims(train, -1)
# target = np.expand_dims(target, -1)

# prediction = NN(trainLocal, targetLocal, testeLocal)

model = NN(train, target, testeLocal)

prediction = model.predict(train)
predictions = []
i = 0
for x in prediction:
    if(x > 0.3):
        predictions.insert(i, 1)
    else:
        predictions.insert(i, 0)
    i += 1
    
score = metrics(target, predictions)

prediction = model.predict(test)
# predictions = []
# i = 0
# for x in prediction:
#     if(x > 0.5):
#         predictions.insert(i, 1)
#     else:
#         predictions.insert(i, 0)
#     i += 1
output(testData, prediction)

# print(prediction)