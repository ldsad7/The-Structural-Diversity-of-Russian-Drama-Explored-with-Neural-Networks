# The Structural Diversity of Russian Drama Explored with Neural Networks
My coursework at the third year of studying

# char_rnn_torch (LSTM and GRU):

char-based (B)LSTM/(B)GRU models (realised via pytorch)

## usage: train_model.py

[-h] -- help

[-model TYPE_OF_MODEL] -- specify type of model: 0 is LSTM, 2 is GRU

[-hsize SIZE_OF_HIDDEN_LAYER] -- size of the hidden layer

[-lnum NUMBER_OF_LAYERS] -- number of layers in a stacked LSTM/GRU

[-iternum NUMBER_OF_ITERATIONS] -- number of epochs

[-lrate RATE_OF_LEARNING] -- rate of learning

[-bsize BATCH_SIZE] -- size of batch

[-b BIAS] -- is there a bias when deriving next hiddet state

[-d DROPOUT] -- percent of dropout

[-bidir BIDIRECTIONAL] -- is LSTM/GRU bidirectional or not?

[-ob OUTPUT_BIAS] -- is there a bias when deriving output vector

[-lseql SIZE_OF_LEARNED_SEQUENCE] -- size of learned sequence

[-o OPTIMIZER] -- 0) Adam; 1) Adagrad; 2) Adamax; 3) Adadelta; 4) SGD; 5) RMSProp

[-gseql SIZE_OF_GENERATED_SEQUENCE] -- size of generated sequence

[-start START_OF_SEQUENCE] -- start of generated sequence

[-t TEMPERATURE] -- temperature coefficient

[-dgraph DRAW_GRAPH] -- draw graph or not?

[-mname MODEL_NAME] -- model_name

[-c CUDA] -- use cuda or not?

file -- which file to take as an input

### Example Usage:

cd char_rnn_torch && python train_model.py name.xml -o 2 -d 0.1 -iternum 100 -hsize 200

On colab: cd char_rnn_torch && !python3 train_model.py name.xml -o 2 -d 0.1 -iternum 100 -hsize 200 -c 1

## usage: sampling.py

[-h] -- help

[-start START_OF_SEQUENCE] -- start of generated sequence

[-gseql SIZE_OF_GENERATED_SEQUENCE] -- size of generated sequence

[-t TEMPERATURE] -- temperature coefficient

[-c CUDA] -- use cuda or not?

file -- model file

### Example Usage:

cd char_rnn_torch && python sampling.py model.model -t 0.4 -start "\<speaker>" -gseql 100

On colab: cd char_rnn_torch && !python3 sampling.py model.model -c 1 -t 0.4 -start "\<speaker>" -gseql 100

# Simple RNN

char-based and word-based RNN models (realised via numpy)

## usage: simple_char_rnn.py

[-h] -- help

[-hsize SIZE_OF_HIDDEN_LAYER] -- size of the hidden layer

[-lseql SIZE_OF_LEARNED_SEQUENCE] -- size of learned sequence

[-lrate RATE_OF_LEARNING] -- rate of learning

[-model TYPE_OF_MODEL] -- specify type of model: 0 is char-based model, 1 is word-based model

[-actfunc FUNCTION_OF_ACTIVATION] -- 0 is tanh, 1 is relu, 2 is elu, 3 is sigmoid

[-choose WAY_OF_CHOOSING_NEXT_CHAR] -- 0 is random choice, 1 is argmax

[-iternum NUMBER_OF_ITERATIONS] -- number of epochs

[-gseql SIZE_OF_GENERATED_SEQUENCE] -- size of generated sequence

[-dgraph DRAW_GRAPH] -- draw graph or not?

file -- which file to take as an input

### Example Usage:

cd simple_char_rnn_numpy && python simple_char_rnn.py -model 0 -actfunc 1 -gseql 50 ../all_correct_xmls_from_rusdracor_in_one_file.xml

# html_to_xml_converter:

converter of html page to xml (rules-based approach)

## usage: html_to_xml_converter.py

[-h] --help

url -- which html page to convert to xml

### Example Usage:

cd html_to_xml_converter && python html_to_xml_converter.py http://az.lib.ru/o/ostrowskij_a_n/text_0066.shtml

cd html_to_xml_converter && python html_to_xml_converter.py http://az.lib.ru/o/ostrowskij_a_n/text_0068.shtml

cd html_to_xml_converter && python html_to_xml_converter.py http://az.lib.ru/o/ostrowskij_a_n/text_0102.shtml
