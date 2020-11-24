import numpy as np
import matplotlib.pyplot as plt



# define hyperparameter
Alpha = 0.1
Input_length = 8
Input_size = 2
Hidden_size = 8
Output_size = 1
Iteration = 10000

# define util functions
def sigmoid(x):
    return 1/(1+np.exp(-x))



# record accuracy in every 100 iteration
accuracy_records = list()
num_right = 0

# initialize weight
w0 = 2 * np.random.random((Input_size, Hidden_size)) - 1
w1 = 2 * np.random.random((Hidden_size, Output_size)) - 1
w_h = 2 * np.random.random((Hidden_size, Hidden_size)) - 1
w0_increment = np.zeros((Input_size, Hidden_size))
w1_increment = np.zeros((Hidden_size, Output_size))
w_h_increment = np.zeros((Hidden_size, Hidden_size))

# a dictionary that map int to its banary form.
mapIntToBinary = {}
integers = np.array([range(pow(2, Input_length))], dtype=np.uint8).T
binaries = np.unpackbits(integers, axis = 1)
for integer in range(pow(2, Input_length)):
    mapIntToBinary[integer] = binaries[integer]


########################### Training ########################
for itera in range(Iteration // 100):
    for ite in range(100): # calculate accuracy after every 100 iteration
        prediction = np.zeros(Input_length)

        output_deltas = list()
        output_history = list()
        output_history.append(np.zeros(Hidden_size))
        post_h_delta = np.zeros(Hidden_size)

        # create two random inputs in range [0,127]
        input1_int = np.random.randint(1, pow(2, Input_length - 1))
        input2_int = np.random.randint(1, pow(2, Input_length - 1))
        result_int = input1_int + input2_int

        input1 = mapIntToBinary[input1_int]
        input2 = mapIntToBinary[input2_int]
        result = mapIntToBinary[result_int]

        # feed forward
        for i in np.flip(range(Input_length)):
            x = np.array([input1[i], input2[i]]).reshape(1, 2)
            y = np.array([result[i]]).reshape(1,1)

            output_h = sigmoid(np.dot(x, w0) + np.dot(output_history[-1], w_h))
            output = sigmoid(np.dot(output_h, w1))
            prediction[i] = np.round(output[0][0])

            output_history.append(np.copy(output_h))
            output_deltas.append((y - output) * (output * (1 - output)))

        # backprop
        for i in range(Input_length):
            x = np.array([input1[i], input2[i]]).reshape(1,2)
            output_h = output_history[-i - 1]
            prev_h = output_history[-i - 2]
            output_delta = output_deltas[-i - 1]

            h_delta = (np.dot(post_h_delta, w_h.T) + np.dot(output_delta, w1.T)) * (output_h * (1 - output_h))
            post_h_delta = h_delta

            w0_increment += x.T.dot(h_delta)
            w1_increment += np.atleast_2d(output_h).T.dot(output_delta)
            w_h_increment += np.atleast_2d(prev_h).T.dot(h_delta)

        # update number of right prefictions
        if np.sum(np.equal(prediction, result)) == Input_length:
            num_right += 1

        # weight update (learning rate)
        w0 += Alpha * w0_increment
        w1 += Alpha * w1_increment
        w_h += Alpha * w_h_increment
        w0_increment = np.zeros((Input_size, Hidden_size))
        w1_increment = np.zeros((Hidden_size, Output_size))
        w_h_increment = np.zeros((Hidden_size, Hidden_size))

    accuracy_records.append(num_right)
    num_right = 0



###################### plot records ########################
x_axis = range(Iteration // 100)
y_axis = accuracy_records

plt.plot(x_axis, y_axis)
plt.ylabel('Accuracy')
plt.show()