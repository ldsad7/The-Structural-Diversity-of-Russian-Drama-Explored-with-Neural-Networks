import argparse
import torch
from torch.autograd import Variable
import re
import numpy as np
import string
from model import CharModel
from sampling import sampling, string_to_tensor, transliterate, detransliterate
import matplotlib.pyplot as plt
import random
from itertools import cycle
import json

# print(torch.__version__) # 1.0.1

def count_loss(char_model, input_symbols, correct_output_symbols):
    hidden_state = char_model.hidden_state_initialization()

    if arguments.cuda: #
        if arguments.type_of_model == 0:
            hidden_state = (hidden_state[0].cuda(), hidden_state[1].cuda()) #
        else:
            hidden_state = hidden_state.cuda() #

    # We need to zero out the gradients because \
    # we don't want to accumulate them on subsequent backward passes
    char_model.zero_grad()

    loss = 0
    for index in range(arguments.size_of_learned_sequence):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols[:, index], hidden_state)
        loss += char_model.metrics(output_vector.view(arguments.batch_size, -1), \
                                   correct_output_symbols[:, index])
    # backward pass
    loss.backward()

    # optimize parameters
    char_model.optimizer.step()

    # return mean loss
    return loss.item() / arguments.size_of_learned_sequence

def data_for_training(text, symbols):
    # batch_size X size_of_learned_sequence
    input_symbols = torch.LongTensor(arguments.batch_size, arguments.size_of_learned_sequence)
    correct_output_symbols = torch.LongTensor(arguments.batch_size, arguments.size_of_learned_sequence)

    for batch_num in range(arguments.batch_size):
        start_position = np.random.randint(0, len(text) - arguments.size_of_learned_sequence)
        input_symbols[batch_num] = string_to_tensor(text[start_position:start_position + \
                                                    arguments.size_of_learned_sequence], symbols)
        correct_output_symbols[batch_num] = string_to_tensor(text[start_position + 1:start_position + \
                                                             arguments.size_of_learned_sequence + 1], symbols)
    if arguments.cuda: #
        input_symbols = input_symbols.cuda() #
        correct_output_symbols = correct_output_symbols.cuda() #
    
    # return tensors wrapped by Variable
    return Variable(input_symbols), Variable(correct_output_symbols)

def train_model(arguments):
    with open(arguments.file, 'r', encoding='utf-8') as f:
        text_without_translit = f.read()
        text = transliterate(text_without_translit) # , 'ru', reversed=True

    symbols = string.printable
    len_symbols = len(string.printable)

    # char_model
    char_model = CharModel(len_symbols, arguments.size_of_hidden_layer, len_symbols, batch_size=arguments.batch_size, \
                           type_of_model=arguments.type_of_model, num_layers=arguments.number_of_layers, \
                           bias=arguments.bias, dropout=arguments.dropout, bidirectional=arguments.bidirectional, \
                           output_bias=arguments.output_bias)

    #char_model = torch.load("/content/GRU_5000_correct_to_Russian.model")

    if arguments.cuda: #
        char_model.cuda() #

    # define optimizer
    if arguments.optimizer == 0:
        func = torch.optim.Adam
    elif arguments.optimizer == 1:
        func = torch.optim.Adagrad
    elif arguments.optimizer == 2:
        func = torch.optim.Adamax
    elif arguments.optimizer == 3:
        func = torch.optim.Adadelta
    elif arguments.optimizer == 4:
        func =  torch.optim.SGD
    elif arguments.optimizer == 5:
        func = torch.optim.RMSprop
    else:
        raise ValueError("Unknown optimizer number")
    char_model.optimizer = func(char_model.parameters(), lr=arguments.rate_of_learning)

    losses = []
    iterations = []
    try:
        for iteration in range(arguments.number_of_iterations):
            iterations.append(iteration)
            print("Now iteration", iteration)
            input_symbols, correct_output_symbols = data_for_training(text, symbols)
            loss = count_loss(char_model, input_symbols, correct_output_symbols)
            losses.append(loss)
            if iteration % 100 == 1: # here you can change number of iteration when you print loss
                print('loss:', loss)
                output_text = detransliterate(sampling(char_model, symbols, arguments)) # , 'ru'

                # transliterate names in tags back to English
                tags = re.findall('<.*?>', output_text)
                for tag in tags:
                    try:
                        output_text = re.sub(re.escape(tag), transliterate(tag), output_text)
                    except:
                        print(tag, transliterate(tag))
                print('output:\n', output_text, '\n')

        # NB: конкретно для этого случая
    except:
        pass
    print("!!!")
    torch.save(char_model, "LSTM_500.model")

    # save only the model parameters
    # torch.save(char_model.state_dict(), arguments.file[:-4] + '.model')
    return iterations, losses

def draw_graph(arguments):
    cycol = cycle('bgrcmk')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 1.8, 1.8])
    arguments.model_name = 'GRU_' + arguments.model_name
    arguments.type_of_model = 0
    arguments.size_of_hidden_layer = 550
    arguments.number_of_layers = 2
    arguments.rate_of_learning = 0.005
    arguments.batch_size = 400
    arguments.size_of_learned_sequence = 250
    arguments.bias = True
    arguments.dropout = 0
    arguments.output_bias = True
    arguments.bidirectional = False
    labels = ['Adam', 'SGD', 'RMSProp']
    for i, size in enumerate([0, 4, 5]):
        arguments.optimizer = size
        iterations, losses = train_model(arguments)
        ax.plot(iterations, losses, color=next(cycol), label=labels[i], linewidth=3)
    ax.legend(fontsize=30)
    ax.set_title('Cross Entropy Loss Change Over time (LSTM)', fontsize=30)
    ax.set_xlabel('Iteration', fontsize=30)
    ax.set_ylabel('Cross Entropy Loss', fontsize=30)
    plt.tick_params(labelsize=20)
    plt.savefig('loss_function.png', bbox_inches = 'tight', fontsize=30)
    # plt.show()

if __name__ == '__main__':
    # parsing parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)

    # 0 = LSTM
    # 1 = LSTMCell
    # 2 = GRU
    parser.add_argument('-model', '--type_of_model', type=int, default=0) #
    parser.add_argument('-hsize', '--size_of_hidden_layer', type=int, default=550)
    parser.add_argument('-lnum', '--number_of_layers', type=int, default=2) #
    parser.add_argument('-iternum', '--number_of_iterations', type=int, default=5000)
    parser.add_argument('-lrate', '--rate_of_learning', type=float, default=0.005) #
    parser.add_argument('-bsize', '--batch_size', type=int, default=500)
    parser.add_argument('-b', '--bias', type=bool, default=True) #
    parser.add_argument('-d', '--dropout', type=float, default=0)
    parser.add_argument('-bidir', '--bidirectional', type=bool, default=False)
    parser.add_argument('-ob', '--output_bias', type=bool, default=True)
    parser.add_argument('-lseql', '--size_of_learned_sequence', type=int, default=400)

    # 0 = Adam
    # 1 = Adagrad
    # 2 = Adamax
    # 3 = Adadelta
    # 4 = SGD
    # 5 = RMSProp
    parser.add_argument('-o', '--optimizer', type=int, default=0)
    parser.add_argument('-gseql', '--size_of_generated_sequence', type=int, default=250) #
    parser.add_argument('-start', '--start_of_sequence', type=str, default="<speaker>")
    parser.add_argument('-t', '--temperature', type=float, default=1)
    parser.add_argument('-dgraph', '--draw_graph', type=int, default=0)
    parser.add_argument('-mname', '--model_name', type=str, default="GRU.model")
    parser.add_argument('-c', '--cuda', type=int, default=0) #
    arguments = parser.parse_args()
    if arguments.draw_graph == 0:
        iterations, losses = train_model(arguments)
        with open("GRU_iters_and_losses.json", 'w', encoding='utf-8') as f:
            json.dump((iterations, losses), f)
    else:
        draw_graph(arguments)
