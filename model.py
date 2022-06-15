import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from itertools import repeat


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # maybe add a 1D batch norm so the output into the decoder has batch normalization
        
        self.batch_norm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch_norm(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=drop_prob)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # discard the <END> word to avoid strange behavior when <END> is the input of the RNN
        # and this allows us to place the image feaures in the embedding sequence while
        # retaining the desired shape!
        captions = captions[:, :-1]
               
        # TEMP for experimentation 
        # print(features.shape, captions.shape)
        
        embeddings = self.embed(captions)
        # print(embeddings.shape)
        
        # concatenate image features along with word embeddings for input to RNN
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # print(embeddings.shape)
        
        # feed the embedding directly into the LSTM directly
        # since the dataloader provides all inputs of the same length
        lstm_out, _ = self.lstm(embeddings)
        # print(lstm_out.shape)
        
        outputs = self.linear(lstm_out)
        # print(outputs.shape)
        
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # store predicted words in a list
        outputs = []
        
        # iterate and get predictions
        for i in range(max_len):
            # feed predicted tensor back through the LSTM on each pass
            
            lstm_out, states = self.lstm(inputs, states)
            linear_out = self.linear(lstm_out)
            pred = int(torch.argmax(linear_out).cpu().detach().numpy())
            outputs.append(pred)
            
            # break if end token is passed
            if pred == 1:
                break
            
            # embed previous word for input on next iteration
            new_input = torch.tensor(outputs[-1]).type(torch.cuda.LongTensor).unsqueeze(0)
            inputs = self.embed(new_input).unsqueeze(0)
        
        return outputs
    
    
    
    def get_beam_sample(self, inputs, states=None, max_len=20):
        # store predicted words in a list
        outputs = []
        sample = torch.zeros(1, max_len, self.vocab_size)
        
        # iterate and get predictions
        for i in range(max_len):
            # feed predicted tensor back through the LSTM on each pass
            
            lstm_out, states = self.lstm(inputs, states)
            linear_out = self.linear(lstm_out)
            # print(linear_out.shape)
            # print(sample[0, i, :].shape)

            pred = int(torch.argmax(linear_out).cpu().detach().numpy())
            outputs.append(pred)
            
            # break if end token is passed
            # if pred == 1:
            #     break
            
            # update beam sample
            sample[0, i, :] = linear_out.squeeze(0).squeeze(0)
            
            # embed previous word for input on next iteration
            new_input = torch.tensor(outputs[-1]).type(torch.cuda.LongTensor).unsqueeze(0)
            inputs = self.embed(new_input).unsqueeze(0)
        
        return sample
    
    
def beam_search_decoder(post, k):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)
        
    from: https://stackoverflow.com/questions/64356953/batch-wise-beam-search-in-pytorch

    """

    batch_size, seq_length, _ = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


    
    
    