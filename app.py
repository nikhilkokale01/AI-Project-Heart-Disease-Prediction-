from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)


heart_data = pd.read_csv('heart_data_updated.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# print(heart_data)

heart_data.isnull().sum()

heart_data.describe()

# heart_data.hist(figsize = (15,15))
# plt.show()

heart_data['target'].value_counts()

# 1-->Defective Heart
# 0-->Healthy Heart

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# print(X)
# print(Y)

def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LogisticRegression():

    def __init__(self, lr=0.1, n_iters=30000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def predict(self, X):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_pred)
            class_pred = [0 if y<=0.5 else 1 for y in y_pred]
            return class_pred
    
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

            

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.45)

model = LogisticRegression()

model.fit(X_train,Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
# X_train_prediction = model.predict(X_train)
training_data_accuracy = np.mean(X_train_prediction==Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = np.mean(X_test_prediction==Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

@app.route('/')
def home():
    return render_template('index.html', columns=X.columns)

@app.route('/predict', methods=['POST'])

def predict():
    try:
        #User input
        input_data = [float(request.form[column]) for column in X.columns]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        
        prediction = model.predict(input_data_reshaped)

        
        result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

        
        return redirect(url_for('result', prediction_result=result))

    except Exception as e:
        return render_template('index.html', columns=X.columns, prediction_result='Error: {}'.format(str(e)))

@app.route('/result/<prediction_result>')
def result(prediction_result):
    return render_template('result.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

