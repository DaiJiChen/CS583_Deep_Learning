import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# load dataset and create tarining/test data.
dataset = load_wine()
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df['target'] = dataset.target

x = df.drop('target', axis = 1)
y = df['target']

# 60% training, 20% validation, 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

# scale feature value between (0, 1)
scale = MinMaxScaler(feature_range=(0,1))
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)


def construct():
    model = Sequential()
    model.add(Dense(13, input_dim = 13, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    return model

def training_and_test(model, learning_rate, batchsize):
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizers.adam(lr = learning_rate), metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size = batchsize, epochs = 200, verbose = 0, validation_data = (x_val, y_val))
    loss_and_acc = model.evaluate(x_test, y_test)
    return loss_and_acc[1]



################# step 3: build model and output accuracy #################
model = construct()
print("step 3 Accuracy: ", training_and_test(model, 1e-3, 10))


################# Question 1: tune the learning rate and batch size #################

# 1.1 choose optimal learning rate
learning_rate = [1e-5, 1e-4,1e-3,1e-2, 1e-1]
optimal_rate = 0.0
max_acc = 0.0
accuracy = []

for rate in learning_rate:
    model = construct()
    acc = training_and_test(model, rate, 10)
    print(acc)
    accuracy.append(acc)
    if(acc > max_acc):
        max_acc = acc
        optimal_rate = rate
print("optimal learning rate: ", optimal_rate)

plt.plot(['1e-5','1e-4','1e-3','1e-2','1e-1'], accuracy)
plt.xlabel("learning_rate")
plt.ylabel("accuracy")
plt.show()

# 1.2 choose optimal batch size
batch_size = [1, 5, 10, 30, 60, 100]
optimal_batch_size = 0
max_acc = 0.0
accuracy = []

for size in batch_size:
    model = construct()
    acc = training_and_test(model, 1e-1, size)
    print(acc)
    accuracy.append(acc)
    if(acc > max_acc):
        max_acc = acc
        optimal_batch_size = size
print("optimal batch size: ", optimal_batch_size)

plt.plot(['1', '5', '10', '30', '60', '100'], accuracy)
plt.xlabel("batch size")
plt.ylabel("accuracy")
plt.show()






