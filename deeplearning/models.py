import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os


class generic_model(nn.Module):
    """
    contains basic functions for storing and loading a model
    """

    def __init__(self, config):

        super(generic_model, self).__init__()

        self.config_file = config

    def loss(self, predicted, truth):

        return self.loss_func(predicted, truth)

    # save model, along with loss details and testing accuracy
    # best is the model which has the lowest test loss. This model is used during feature extraction
    def save_model(self, is_best, epoch, train_loss, test_loss, rnn_name, layers, hidden_dim):

        base_path = self.config_file['models']
        if is_best:
            filename = os.path.join(base_path, 'best_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth')
        else:
            filename = os.path.join(base_path, str(epoch) + '_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, filename)

        print("Saved model")

    # Loads saved model for resuming training or inference
    def load_model(self, mode, rnn_name, layers, hidden_dim, epoch=None):

        # if epoch is given, load that particular model, else load the model with name 'best'
        if mode == 'test' or mode == 'test_one':

            try:
                if epoch is None:
                    filename = os.path.join(self.config_file['models'], 'best_' + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth')
                else:
                    filename = os.path.join(self.config_file['models'], str(epoch) + '_' + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth')
                print(filename)

                checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
                # load model parameters
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded pretrained model from:", filename)

            except:
                print("Couldn't find model for testing")
                exit(0)
        # train
        else:
            # if epoch is given, load that particular model else, load the model trained on the most number of epochs
            # e.g. if dir has 400, 500, 600, it will load 600.pth
            if epoch is not None:
                filename = self.config_file['models'] + str(epoch) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth'
            else:
                directory = [x.split('_') for x in os.listdir(self.config_file['models'])]
                to_check = []
                for poss in directory:
                    try:
                        to_check.append(int(poss[0]))
                    except:
                        continue

                if len(to_check) == 0:
                    print("No pretrained model found")
                    return 0, [], []
                # model trained on the most epochs
                filename = os.path.join(self.config_file['models'], str(max(to_check)) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth')

            # load model parameters and return training/testing loss and testing accuracy
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print("Loaded pretrained model from:", filename)

            return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['test_loss']


#generic model contains generic methods for loading and storing a model
class RNN(generic_model):

    def __init__(self, config):

        super(RNN, self).__init__(config)

        # Store important parameters
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim'] 
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']

        #whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        else:
            self.use_embedding = False
            
        #linear layer after RNN output
        if self.rnn_name == 'Transformer':
            in_features = self.embed_dim +config['miss_linear_dim']
        else:
            in_features = config['miss_linear_dim'] + self.hidden_dim*2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

        #linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])        

        #declare RNN
        if self.rnn_name == 'LSTM':
            self.encoder = nn.LSTM(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               dropout=config['dropout'],
                               bidirectional=True, batch_first=True)
        elif self.rnn_name == 'GRU':
            self.encoder = nn.GRU(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              dropout=config['dropout'],
                              bidirectional=True, batch_first=True)
        elif self.rnn_name == 'Transformer':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, dropout=config['dropout'], batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        # set learning rate decay
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay'], gamma=config['lr_decay_rate'])

    def forward(self, x, x_lens, miss_chars):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """        
        if self.use_embedding:
            x = self.embedding(x)
            
        batch_size, seq_len, _ = x.size()
        if self.rnn_name != 'Transformer':
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_name == 'LSTM':
            output, (hidden, _) = self.encoder(x)
        elif self.rnn_name == 'GRU':
            output, hidden = self.encoder(x)
        elif self.rnn_name == 'Transformer':
            x = torch.cat((self.cls_token.repeat(batch_size, 1, 1), x), dim=1)
            hidden = self.encoder(x)
            hidden = hidden[:, 0, :]
        
        if self.rnn_name != 'Transformer':
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = hidden[-1]
            hidden = hidden.permute(1, 0, 2)
            hidden = hidden.contiguous().view(hidden.shape[0], -1)

        #project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)
        #concatenate RNN output and miss chars
        concatenated = torch.cat((hidden, miss_chars), dim=1)
        #predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))

    def calculate_loss(self, model_out, labels, input_lens, miss_chars, use_cuda):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
                            passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs*miss_chars, dim=(0,1))/outputs.shape[0]
        
        input_lens = input_lens.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lens)/torch.sum(1/input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
            weights = weights.cuda()
        
        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels)
        return actual_penalty, miss_penalty
        