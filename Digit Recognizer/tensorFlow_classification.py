import tensorflow as tf
import pandas as pd
import numpy as np

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
target = train.pop('label')

matrix_len = 28
matrix = np.array([[0.0 for x in range(matrix_len)] for y in range(matrix_len)] )

matrix_list_train = []
for l, row in train.iterrows():
    for i in range(matrix_len):
        for j in range(matrix_len):         
            matrix[i][j] = row[j * matrix_len + i] / 250.0
    matrix_list_train.insert(l, matrix)
matrix_list_train = np.array(matrix_list_train)

matrix_list_test  = []
for l, row in train.iterrows():
    for i in range(matrix_len):
        for j in range(matrix_len):         
            matrix[i][j] = row[j * matrix_len + i] / 250.0
    matrix_list_test.insert(l, matrix)
matrix_list_test = np.array(matrix_list_test)

print(type(matrix_list_train[0]))
print(type(matrix_list_train[0][0]))
print(type(matrix_list_train[0][0][0]))

            
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(matrix_list_train, target.to_numpy(), epochs=5)
# model.evaluate(xTest, yTest, verbose=2)
predictions = model.predict(matrix_list_test)

# from sklearn.metrics import confusion_matrix, f1_score

# confusion_matrix = confusion_matrix(prediction,  )
# f1_score         = f1_score(yTest, yPred, average='weighted')
# print("Confusion Matrix: \n", confusion_matrix)
# print("F1-Score: ", f1_score)

predictions.to_csv(r'./prediction.csv', index = True, index_label = 'ImageId')