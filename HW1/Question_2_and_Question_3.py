import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def get_data(feature_size):
    # load dataset and create tarining/test data.
    dataset = load_wine()
    df = pd.DataFrame(dataset.data)
    df.columns = dataset.feature_names
    df['target'] = dataset.target


    x = df.drop('target', axis = 1)
    y = df['target']

    # choose certen percentage of training data
    if(feature_size != 13):
        x = df.sample(feature_size, axis = 'columns')

    # 60% training, 20% validation, 20% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    # scale feature value between (0, 1)
    scale = MinMaxScaler(feature_range=(0,1))
    x_train = scale.fit_transform(x_train)
    x_test = scale.fit_transform(x_test)

    return x_train, x_test, y_train, y_test


def construct(feature_size):
    model = Sequential()
    model.add(Dense(13, input_dim = feature_size, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    return model

def training_and_test(model, learning_rate, batchsize, data_percentage, feature_size):
    x_train, x_test, y_train, y_test = get_data(feature_size)

    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = optimizers.adam(lr = learning_rate), metrics = ['accuracy'])

    val_percentage = round(1-data_percentage, 2)
    model.fit(x_train, y_train, batch_size = batchsize, epochs = 200, verbose = 0, validation_split = val_percentage)

    loss_and_acc = model.evaluate(x_test, y_test)
    return loss_and_acc[1]



################# Question 2: different percentage of training data #################
data_per = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracy = []

for per in data_per:
    model = construct(13)
    acc = training_and_test(model, 1e-3, 30, per, 13)
    accuracy.append(acc)

plt.plot(data_per, accuracy)
plt.xlabel("training data percentage")
plt.ylabel("accuracy")
plt.show()

################# Question 3: different percentage of features #################
feature_per = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracy = []

for per in feature_per:
    feature_size = (int)(13 * per)
    model = construct(feature_size)
    acc = training_and_test(model, 1e-3, 30, 1.0, feature_size)
    accuracy.append(acc)

plt.plot(feature_per, accuracy)
plt.xlabel("feature percentage")
plt.ylabel("accuracy")
plt.show()




