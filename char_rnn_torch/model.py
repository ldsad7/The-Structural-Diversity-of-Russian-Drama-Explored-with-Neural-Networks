from torch import nn, zeros, cat
from torch.autograd import Variable

class CharModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, type_of_model=0, \
                 num_layers=2, bias=True, dropout=0, bidirectional=False, output_bias=True):
        super().__init__()        
        # initializing attributes
        # the number of expected features in the input
        self.input_size = input_size

        # the number of features in the hidden state
        self.hidden_size = hidden_size

        # the number of features in the output
        self.output_size = output_size

        # set batch_size
        self.batch_size = batch_size

        # type of model (LSTM/LSTMCell/GRU)
        self.type_of_model = type_of_model

        # stacked LSTM/GRU or not? if yes, than how many layers?
        self.num_layers = num_layers
        #if self.type_of_model == 1: # LSTMCell
        #    self.num_layers = 1

        self.bias = bias

        # dropout parameter for LSTM and GRU
        self.dropout = dropout

        # bidirectional LSTM or GRU?
        self.bidirectional = bidirectional

        # bias in Linear layer
        self.output_bias = output_bias

        # define metrics
        self.metrics = nn.CrossEntropyLoss()

        # set optimizer to None
        self.optimizer = None

        # define embedding for input
        self.encoder = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.hidden_size)

        # define model (LSTM/LSTMCell/GRU)
        if self.type_of_model == 0:
            self.model = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                 num_layers=self.num_layers, bias=self.bias, \
                                 dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.type_of_model == 1:
            self.model = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                     bias=self.bias)
        elif self.type_of_model == 2:
            self.model = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, \
                                num_layers=self.num_layers, bias=self.bias, \
                                dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError("Unknown type of model's number")
        if self.bidirectional:
            self.decoder = nn.Linear(2 * self.hidden_size, self.output_size, bias=output_bias)
        else:
            self.decoder = nn.Linear(self.hidden_size, self.output_size, bias=output_bias)

    def forward(self, input_vector, hidden_state):
        batch_size = input_vector.shape[0]
        encoded_vector = self.encoder(input_vector)

        # time dimension X batch dimension X feature dimension
        # timesteps X batch_size X features
        # one batch per epoch => first dimension: 1

        output_vector, hidden_state = self.model(encoded_vector.view(1, batch_size, -1), hidden_state)

        # bidir lstm
        if self.bidirectional:
            forward_output_vector, backward_output_vector = output_vector[:1, :, :self.hidden_size], \
                                                            output_vector[0:, :, self.hidden_size:]
            output_vector = cat((forward_output_vector, backward_output_vector), dim=-1)

        # view is an analogue of numpy reshape
        output_vector = self.decoder(output_vector.view(batch_size, -1))
        return output_vector, hidden_state

    def hidden_state_initialization(self):
        # GPU
        if self.type_of_model == 2:
            # return only hidden state's vector
            if self.bidirectional:
                return cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                            Variable(zeros(self.num_layers, self.batch_size, self.hidden_size))))
            else:
                return Variable(zeros(self.num_layers, self.batch_size, self.hidden_size))
        # LSTM and LSTMCell
        # return hidden state and cell's vectors
        if self.bidirectional:
            return (cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                         Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))),
                    cat((Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                         Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))))
        else:
            return (Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)),
                    Variable(zeros(self.num_layers, self.batch_size, self.hidden_size)))
