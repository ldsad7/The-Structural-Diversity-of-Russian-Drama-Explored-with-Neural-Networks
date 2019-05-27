import argparse
import re
import numpy as np
# import gensim
# from gensim.models import Word2Vec
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def relu(arr):
    return arr * (arr > 0)

def elu(arr):
    # alpha = 1
    for i, elem in enumerate(arr):
        if elem <= 0:
            arr[i] = np.exp(elem) - 1
    return arr

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

def tanh_derivative(arr):
    return 1 - arr * arr

def relu_derivative(arr):
    arr[arr <= 0] = 0
    arr[arr > 0] = 1
    return arr

def elu_derivative(arr):
    for i, elem in enumerate(arr):
        if elem <= 0:
            arr[i] += 1 # alpha == 1
        else:
            arr[i] = 1
    return arr

def sigmoid_derivative(arr):
    return arr * (1 - arr)

def softmax(arr):
    f = np.exp(arr - np.max(arr))
    return f / f.sum(axis=0)

def forward_prop(input_indices, correct_output_indices, prev_hidden_state, index_to_symbol, function_of_activation, \
                 W, bias_for_hidden_state, V, V_bias, U, symbols):
    loss = 0
    input_vectors, hidden_states, outputs, probabilities = {}, {}, {}, {}

    # initializing first previous hidden state
    hidden_states[-1] = prev_hidden_state
    for index in range(len(input_indices)):

        # initializing input vectors
        # initializing with one-hot encoding
        input_vectors[index] = np.zeros((len(symbols), 1))
        input_vectors[index][input_indices[index]] = 1

        # if necessary
        input_vectors[index] = input_vectors[index].reshape((input_vectors[index].shape[0], 1))
        # initializing activation function
        if function_of_activation == 0:
            func = np.tanh
        elif function_of_activation == 1:
            func = relu
        elif function_of_activation == 2:
            func = elu
        elif function_of_activation == 3:
            func = sigmoid
        else:
            raise ValueError('Unknown activation function\'s number')

        # get the next hidden state
        hidden_states[index] = func(np.dot(U, input_vectors[index]) + \
            np.dot(W, hidden_states[index - 1]) + bias_for_hidden_state)

        # vector for the next output symbol
        outputs[index] = np.dot(V, hidden_states[index]) + V_bias
        # vector of probabilities for the next output symbol
        probabilities[index] = softmax(outputs[index]) # np.exp(outputs[index]) / np.sum(np.exp(outputs[index]))
        # counting cross-entropy loss via softmax
        # in order not to get inf on relu activation
        loss += -np.log(probabilities[index][correct_output_indices[index]][0] + 0.0001)
    return loss, input_vectors, hidden_states, outputs, probabilities

def backward_prop(input_indices, correct_output_indices, probabilities, \
                  hidden_states, size_of_hidden_layer, symbols, function_of_activation, \
                  W, bias_for_hidden_state, V, V_bias, U, input_vectors):
    # initializing gradients for the parameters
    W_grad = np.zeros((size_of_hidden_layer, size_of_hidden_layer))
    bias_for_hidden_state_grad = np.zeros((size_of_hidden_layer, 1))
    V_grad = np.zeros((len(symbols), size_of_hidden_layer))
    V_bias_grad = np.zeros((len(symbols), 1))
    U_grad = np.zeros((size_of_hidden_layer, len(symbols)))
    next_hidden_state_grad = np.zeros((size_of_hidden_layer, 1))
    
    for index in range(len(input_indices))[::-1]:
        # reducing only correct_output_indices[index]' component
        dy = probabilities[index]
        # probabilities[index][correct_output_indices[index]] -= 1
        dy[correct_output_indices[index]] -= 1
        # V_grad += np.dot(probabilities[index], hidden_states[index].T)
        V_grad += np.dot(dy, hidden_states[index].T)
        # V_bias_grad += probabilities[index]
        V_bias_grad += dy

        dh = np.dot(V.T, dy) + next_hidden_state_grad # backprop into h
        
        # hidden_state_grad = np.dot(V.T, probabilities[index]) + next_hidden_state_grad
        hidden_state_grad = np.dot(V.T, dy) + next_hidden_state_grad

        if function_of_activation == 0:
            derivative = tanh_derivative(hidden_states[index])
        elif function_of_activation == 1:
            derivative = relu_derivative(hidden_states[index])
        elif function_of_activation == 2:
            derivative = elu_derivative(hidden_states[index])
        elif function_of_activation == 3:
            derivative = sigmoid_derivative(hidden_states[index])
        else:
            raise ValueError('Unknown activation function\'s number')
        hidden_state_raw_grad = derivative * hidden_state_grad
        bias_for_hidden_state_grad += hidden_state_raw_grad
        W_grad += np.dot(hidden_state_raw_grad, hidden_states[index - 1].T)
        U_grad += np.dot(hidden_state_raw_grad, input_vectors[index].T)
        next_hidden_state_grad = np.dot(W.T, hidden_state_raw_grad)
    return W_grad, bias_for_hidden_state_grad, V_grad, V_bias_grad, U_grad

def cross_entropy_loss(input_indices, correct_output_indices, prev_hidden_state, \
                       index_to_symbol, function_of_activation, W, bias_for_hidden_state, \
                       V, V_bias, U, symbols, size_of_hidden_layer):

    loss, input_vectors, hidden_states, outputs, probabilities = forward_prop(input_indices, correct_output_indices, \
    prev_hidden_state, index_to_symbol, function_of_activation, W, bias_for_hidden_state, V, V_bias, U, symbols)

    W_grad, bias_for_hidden_state_grad, V_grad, V_bias_grad, U_grad = backward_prop(input_indices, correct_output_indices, probabilities, \
    hidden_states, size_of_hidden_layer, symbols, function_of_activation, W, bias_for_hidden_state, V, V_bias, U, input_vectors)

    # clip to mitigate exploding gradients
    for grad_param in [U_grad, W_grad, V_grad, bias_for_hidden_state_grad, V_bias_grad]:
        np.clip(grad_param, -5, 5, out=grad_param)
    return loss, U_grad, W_grad, V_grad, bias_for_hidden_state_grad, V_bias_grad, hidden_states[len(input_indices)-1]

def sampling(index_of_first_letter, hidden_state, len_of_output, \
    function_of_activation, way_of_choosing_next_char, index_to_symbol, \
    W, bias_for_hidden_state, V, V_bias, U, symbols):

    # initializing input vector
    # initializing with one-hot encoding
    input_vector = np.zeros((len(symbols), 1))
    input_vector[index_of_first_letter] = 1

    seq = []
    for _ in range(len_of_output):
        # initializing activation function
        if function_of_activation == 0:
            func = np.tanh
        elif function_of_activation == 1:
            func = relu
        elif function_of_activation == 2:
            func = elu
        elif function_of_activation == 3:
            func = sigmoid
        else:
            raise ValueError('Unknown activation function\'s number')
        input_vector = input_vector.reshape((input_vector.shape[0], 1))
        hidden_state = func(np.dot(U, input_vector) + np.dot(W, hidden_state) + bias_for_hidden_state)
        output = np.dot(V, hidden_state) + V_bias
        probability = softmax(output)

        if way_of_choosing_next_char == 0:
            index = np.random.choice(range(len(symbols)), p=probability.ravel())
        elif way_of_choosing_next_char == 1:
            index = np.argmax(probability)
        else:
            raise ValueError('Unknown way_of_choosing_next_char\'s number')

        # initializing input vector
        # initializing with one-hot encoding
        input_vector = np.zeros((len(symbols), 1))
        input_vector[index] = 1

        seq.append(index)
    return seq

def matrices_init(size_of_hidden_layer, symbols):
    # initializing W with an orthogonal matrix from svd
    W = np.random.randn(size_of_hidden_layer, size_of_hidden_layer)
    W, _, _ = np.linalg.svd(W)
    bias_for_hidden_state = np.random.randn(size_of_hidden_layer, 1) * 0.001
    # initializing V and U with random matrices
    V = np.random.randn(len(symbols), size_of_hidden_layer) * 0.001
    V_bias = np.random.randn(len(symbols), 1) * 0.001
    U = np.random.randn(size_of_hidden_layer, len(symbols)) * 0.001
    return W, bias_for_hidden_state, V, V_bias, U

def train_and_generate(file, size_of_hidden_layer=125, \
          size_of_learned_sequence=25, rate_of_learning=0.1, \
          type_of_model=0, function_of_activation=0, \
          way_of_choosing_next_char=0, number_of_iterations=1000, \
          size_of_generated_sequence=250, draw_graph=0):

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        if type_of_model == 1:
            text = re.sub('<', ' < ', text)
            text = re.sub('>', ' > ', text)
            text = re.sub('=', ' = ', text)
            # save all separators
            text = re.split('(\s)', text)

    symbols = tuple(set(text))

    index_to_symbol = dict(enumerate(symbols))
    symbol_to_index = {symbol:index for index, symbol in enumerate(symbols)}

    W, bias_for_hidden_state, V, V_bias, U = matrices_init(size_of_hidden_layer, symbols)

    start_position = 0
    prev_hidden_state = np.random.randn(size_of_hidden_layer, 1)

    mV = np.zeros((len(symbols), size_of_hidden_layer))
    mU = np.zeros((size_of_hidden_layer, len(symbols)))
    mW = np.zeros((size_of_hidden_layer, size_of_hidden_layer))
    mbias_for_hidden_state = np.zeros((size_of_hidden_layer, 1))
    mV_bias = np.zeros((len(symbols), 1))

    smooth_loss = -np.log(1.0 / len(symbols)) * size_of_learned_sequence # smoothed loss
    iterations = []
    losses = []
    smoothed_losses = []
    for iteration in range(number_of_iterations):
        iterations.append(iteration)
        if start_position > len(text) - size_of_learned_sequence:
            start_position = 0
            prev_hidden_state = np.zeros((size_of_hidden_layer,1))
        input_indices = []
        for symbol in text[start_position:start_position + size_of_learned_sequence]:
            input_indices.append(symbol_to_index[symbol])
        correct_output_indices = []
        for symbol in text[start_position + 1:start_position + size_of_learned_sequence + 1]:
            correct_output_indices.append(symbol_to_index[symbol])

        # sampling
        if iteration % 100 == 0:
            indices = sampling(input_indices[0], prev_hidden_state, size_of_generated_sequence, \
                      function_of_activation, way_of_choosing_next_char, index_to_symbol, \
                      W, bias_for_hidden_state, V, V_bias, U, symbols)
            if type_of_model == 1:
                output_text = ' '.join([index_to_symbol[index] for index in indices])
                output_text = re.sub(' <\s+', '<', output_text)
                output_text = re.sub('\s+> ', '>', output_text)
                output_text = re.sub('\s+=\s+', '=', output_text)
                output_text = re.sub(' (\s) ', r'\1', output_text)
            else:
                output_text = ''.join([index_to_symbol[index] for index in indices])
            print(str(iteration) + ':\n', output_text, '\n')

        # forward size_of_learned_sequence characters through the net and fetch gradient
        loss, U_grad, W_grad, V_grad, bias_for_hidden_state_grad, V_bias_grad, prev_hidden_state \
        = cross_entropy_loss(input_indices, correct_output_indices, prev_hidden_state, 
                             index_to_symbol, function_of_activation, W, \
                             bias_for_hidden_state, V, V_bias, U, symbols, size_of_hidden_layer)

        # print progress
        # losses.append(loss)
        smoothed_losses.append(loss)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(smooth_loss)
        if iteration % 100 == 0:
            print('iter %d, loss: %f' % (iteration, smooth_loss))
        # Adagrad parameters update
        for param, param_grad, mparam in zip([U, W, V, bias_for_hidden_state, V_bias],
                                [U_grad, W_grad, V_grad, bias_for_hidden_state_grad, V_bias_grad],
                                [mU, mW, mV, mbias_for_hidden_state, mV_bias]):
            mparam += param_grad * param_grad
            param += -rate_of_learning * param_grad / np.sqrt(mparam + 1e-7)

        start_position += size_of_learned_sequence
    return iterations, losses, smoothed_losses

def draw_graph(arguments):
    labels = ['Tanh', 'ReLU', 'ELU', 'Sigmoid']
    colors = ['black', 'r', 'g', 'b']
    fig = plt.figure(figsize=(10, 6))
    axes = fig.add_axes([0.1, 0.1, 1.8, 1.8])
    for i in range(4):
        iterations, losses, smoothed_losses = train_and_generate(arguments.file,
                  size_of_hidden_layer=arguments.size_of_hidden_layer, \
                  size_of_learned_sequence=arguments.size_of_learned_sequence, \
                  rate_of_learning=arguments.rate_of_learning, \
                  type_of_model=arguments.type_of_model, \
                  function_of_activation=i, \
                  way_of_choosing_next_char=arguments.way_of_choosing_next_char, \
                  number_of_iterations=arguments.number_of_iterations, \
                  size_of_generated_sequence=arguments.size_of_generated_sequence, \
                  draw_graph=arguments.draw_graph)
        # replace here losses by smoothed_losses if necessary
        axes.plot(iterations, losses, colors[i], label=labels[i], linewidth=3)
    axes.legend(fontsize=30)
    axes.set_title('Cross Entropy Loss Change Over time (char-level)', fontsize=30)
    axes.set_xlabel('Iteration', fontsize=30)
    axes.set_ylabel('Cross Entropy Loss', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.savefig('loss_function.png', bbox_inches = 'tight', fontsize=30)
    # plt.show()

if __name__ == '__main__':
    # parsing parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('-hsize', '--size_of_hidden_layer', type=int, default=125)
    parser.add_argument('-lseql', '--size_of_learned_sequence', type=int, default=25)
    parser.add_argument('-lrate', '--rate_of_learning', type=float, default=0.1)

    # 0 is char-based model
    # 1 is word-based model
    parser.add_argument('-model', '--type_of_model', type=int, default=0)

    # 0 is tanh
    # 1 is relu
    # 2 is elu
    # 3 is sigmoid
    parser.add_argument('-actfunc', '--function_of_activation', type=int, default=0)

    # 0 is random choice
    # 1 is argmax
    parser.add_argument('-choose', '--way_of_choosing_next_char', type=int, default=0)

    parser.add_argument('-iternum', '--number_of_iterations', type=int, default=10000)
    parser.add_argument('-gseql', '--size_of_generated_sequence', type=int, default=250)
    parser.add_argument('-dgraph', '--draw_graph', type=int, default=0)
    arguments = parser.parse_args()
    if arguments.draw_graph == 0:
        train_and_generate(**vars(arguments))
    else:
        draw_graph(arguments)
