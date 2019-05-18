import argparse
import torch
from torch.autograd import Variable
from model import CharModel # may be needed when we are saving only the model parameters
import string
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

def detransliterate(string):
    from_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    to_list = 'a,b,v,g,d,e,jo,zh,z,i,y,k,l,m,n,o,p,r,s,t,u,f,kh,c,ch,sh,shh,jhh,ih,jh,eh,ju,ja,A,B,V,G,D,E,JO,ZH,Z,I,Y,K,L,M,N,O,P,R,S,T,U,F,KH,C,CH,SH,SHH,JHH,IH,JH,EH,JU,JA'.split(',')
    detransliterated_string = ''
    index_of_symbol = 0
    while index_of_symbol < len(string):
        was_combination = 0
        for i in range(3, 0, -1):            
            if string[index_of_symbol:index_of_symbol+i] in to_list:
                index = to_list.index(string[index_of_symbol:index_of_symbol+i])
                detransliterated_string += from_list[index]
                index_of_symbol += i
                was_combination = 1
                break
        if not was_combination:
            detransliterated_string += string[index_of_symbol]
            index_of_symbol += 1
    return detransliterated_string

def transliterate(string):
    from_list = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    to_list = 'a,b,v,g,d,e,jo,zh,z,i,y,k,l,m,n,o,p,r,s,t,u,f,kh,c,ch,sh,shh,jhh,ih,jh,eh,ju,ja,A,B,V,G,D,E,JO,ZH,Z,I,Y,K,L,M,N,O,P,R,S,T,U,F,KH,C,CH,SH,SHH,JHH,IH,JH,EH,JU,JA'.split(',')
    transliterated_string = ''
    for symbol in string:
        if symbol in from_list:
            index = from_list.index(symbol)
            transliterated_string += to_list[index]
        else:
            transliterated_string += symbol
    return transliterated_string

# Turning a string into a tensor
def string_to_tensor(string, symbols):
    tensor = torch.zeros(len(string)).to(torch.int64)
    for index in range(len(string)):
        try:
            tensor[index] = symbols.index(string[index])
        except ValueError:
            continue
    return Variable(tensor)

def sampling(char_model, symbols, arguments): # 
    # save previous value
    tmp = char_model.batch_size

    char_model.batch_size = 1
    hidden_state = char_model.hidden_state_initialization()
    char_model.zero_grad()

    # set back previous value
    char_model.batch_size = tmp

    # insert a singleton dimension at the first place and wrap by Variable
    input_symbols = Variable(string_to_tensor(arguments.start_of_sequence, symbols).unsqueeze(0))

    if arguments.cuda: #
        if char_model.type_of_model == 0:
            hidden_state = (hidden_state[0].cuda(), hidden_state[1].cuda()) #
        else:
            hidden_state = hidden_state.cuda() #
        input_symbols = input_symbols.cuda() #

    for index in range(len(arguments.start_of_sequence) - 1):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols[:, index], hidden_state)

    input_symbols = input_symbols[:, -1]
    generated_text = arguments.start_of_sequence

    for _ in range(arguments.size_of_generated_sequence):
        # forward pass
        output_vector, hidden_state = char_model.forward(input_symbols, hidden_state)

        # divide by temperature so that to get more conservative or more arbitrary samples
        # the lower the temperature, the more conservative output
        # the higher is temperature, the more random output
        # scaling logits before applying softmax

        # we add 0.1 in order to avoid INF
        bias = 0.1

        try:
            # output_vector = torch.softmax(output_vector.div(arguments.temperature), 0)
            # output_vector = torch.exp(output_vector / arguments.temperature).view(-1)
            output_vector = output_vector.data.view(-1).div(arguments.temperature).exp()
        except RuntimeError:
            output_vector = torch.exp(output_vector / (arguments.temperature + bias)).view(-1)

        # The rows of input do not need to sum to one (in which case we use the values as weights)
        # and it doesn't in this case
        index_of_symbol = torch.multinomial(input=output_vector, num_samples=1)[0]

        # print(index_of_symbol)
        # also possible via softmax and argmax prediction but terrible quality
        # probs = torch.softmax(output_vector, 0)
        # index_of_symbol = torch.max(probs, 0)[1].item()

        index_of_symbol = index_of_symbol.data
        if index_of_symbol >= 0 and index_of_symbol < len(symbols):
            generated_text += symbols[index_of_symbol]
        else:
            # in case of unprintable symbol
            generated_text += symbols[np.random.randint(0, len(symbols))] # right border not included
        # generated_symbol
        input_symbols = Variable(string_to_tensor(generated_text[-1], symbols).unsqueeze(0))

        if arguments.cuda: #
            input_symbols = input_symbols.cuda() #

    return generated_text

if __name__ == '__main__':

    # parsing parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('-start', '--start_of_sequence', type=str, default="<speaker>")
    parser.add_argument('-gseql', '--size_of_generated_sequence', type=int, default=250)
    parser.add_argument('-t', '--temperature', type=float, default=0.5)
    parser.add_argument('-c', '--cuda', type=int, default=0)
    arguments = parser.parse_args()

    char_model = torch.load(arguments.file)
    # load only the model parameters
    # char_model = CharModel(*args, **kwargs) # but you should know the args parameters of training
    # char_model.load_state_dict(torch.load(arguments.file))
    # char_model.eval()

    output_text = detransliterate(sampling(char_model, string.printable, arguments))

    # transliterate names in tags back to English
    tags = re.findall('<.*?>', output_text)
    for tag in tags:
        try:
            output_text = re.sub(re.escape(tag), transliterate(tag), output_text)
        except:
            print(tag, transliterate(tag))
    # output_text = re.sub('Ь', 'ь', output_text)
    print('output:\n', output_text, '\n')