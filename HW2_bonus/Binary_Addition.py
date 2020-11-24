import copy
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

# weight initialization
synapse_0 = 2 * np.random.random((Input_size, Hidden_size)) - 1
synapse_1 = 2 * np.random.random((Hidden_size, Output_size)) - 1
synapse_h = 2 * np.random.random((Hidden_size, Hidden_size)) - 1

print(synapse_0.shape, synapse_1.shape, synapse_h.shape)

s0_update = np.zeros(synapse_0.shape) # s0_update = np.zeros_like(synapse_0)
s1_update = np.zeros(synapse_1.shape)
sh_update = np.zeros(synapse_h.shape)

accuracy = list()
accuracy_history = list()
accuracy_count = 0

mapIntToBinary = {}
integers = np.array([range(pow(2, Input_length))], dtype=np.uint8).T
binariess = np.unpackbits(integers, axis = 1)

for integer in range(pow(2, Input_length)):
    mapIntToBinary[integer] = binaries[integer]


# Training
for j in range(Iteration):
    prediction = np.zeros(Input_length)
    error = 0

    output_layer_deltas = list()
    hidden_layer_values = list()
    hidden_layer_values.append(np.zeros(Hidden_size))

    # create two random inputs in range [0,127]
    input1_int = np.random.randint(1, pow(2, Input_length - 1))
    input2_int = np.random.randint(1, pow(2, Input_length - 1))
    result_int = input1_int + input2_int

    input1 = mapIntToBinary[input1_int]
    input2 = mapIntToBinary[input2_int]
    result = mapIntToBinary[result_int]


    # feed forward !
    # As you have to calculate from the "first" position of the binary number, which stands for the lowest value, loop backward.
    # e.g. 10(2) + 11(2), for the first iteration: X = [[0,1]] y = [[1]]
    for position in reversed(range(Input_length)):
        # Take the input and output label binary values
        X = np.array([[input1[position], input2[position]]])  # dim: (1, 2), e.g. [[1,0]]
        y = np.array([[result[position]]])  # dim: (1, 1), e.g. [[1]]

        # hidden layer h_t = sigmoid(X*W_{hx} + h_{t-1}*W_{hh})
        hidden_layer = sigmoid(np.dot(X, synapse_0) + np.dot(hidden_layer_values[-1], synapse_h))  # dim: (1, 16)

        # output_layer
        output_layer = sigmoid(np.dot(hidden_layer, synapse_1))  # dim: (1, 1), e.g. [[0.47174173]]

        output_layer_error = y - output_layer  # dim: (1, 1)

        # display (just for displying error curve)
        error += np.abs(output_layer_error[0])  # dim: (1, )

        # Save it for the later use in backpropagation step
        output_layer_deltas.append((output_layer_error) * (output_layer*(1 - output_layer)) )

        # save the prediction by my model on this position
        prediction[position] = np.round(output_layer[0][0])

        # save the hidden layer by appending the values to the list
        hidden_layer_values.append(copy.deepcopy(hidden_layer))

    future_hidden_layer_delta = np.zeros(Hidden_size)

    # backprop
    for position in range(Input_length):
        X = np.array([[input1[position], input2[position]]])
        hidden_layer = hidden_layer_values[-position - 1]
        prev_hidden_layer = hidden_layer_values[-position - 2]

        # Get the gradients flowing back from the error of my output at this position, or time step
        output_layer_delta = output_layer_deltas[-position - 1]

        # (Backpropagation)
        # Think about the feed forward step you have done before: h_t = sigmoid(X*W_{hx} + h_{t-1}*W_{hh})
        hidden_layer_delta = (np.dot(future_hidden_layer_delta, synapse_h.T) + np.dot(output_layer_delta, synapse_1.T)) * (hidden_layer*(1 - hidden_layer))

        # Save the updates until the for loop finishes calculation for every position
        # Hidden layer values must be changed ONLY AFTER backpropagation is fully done at every position.
        s1_update += np.atleast_2d(hidden_layer).T.dot(output_layer_delta)
        sh_update += np.atleast_2d(prev_hidden_layer).T.dot(hidden_layer_delta)
        s0_update += X.T.dot(hidden_layer_delta)

        # Preparation for the next step. Now the current hidden_layer_delta becomes the future hidden_layer_delta.
        future_hidden_layer_delta = hidden_layer_delta

    # weight update (learning rate)
    synapse_1 += s1_update * Alpha
    synapse_0 += s0_update * Alpha
    synapse_h += sh_update * Alpha

    # update value initialization for the new training data
    s1_update *= 0
    s0_update *= 0
    sh_update *= 0

    # accuracy
    check = np.equal(prediction, result)
    if np.sum(check) == Input_length:
        accuracy_count += 1
    if (j % 100 == 0):
        accuracy_history.append(accuracy_count)
        accuracy_count = 0

    if (j % 100 == 0):
        # print("Error:" + str(overallError))
        # print("Pred:" + str(pred))
        # print("True:" + str(c))

        final_check = np.equal(prediction, result)
        # print(np.sum(final_check) == INPUT_LENGTH)

        out = 0

        for index, x in enumerate(reversed(prediction)):
            out += x * pow(2, index)
        # print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        # print("------------")

# plot records
x_range = range(Iteration // 100)
plt.plot(x_range,accuracy_history,'b-')
plt.ylabel('accuracy')
plt.show()