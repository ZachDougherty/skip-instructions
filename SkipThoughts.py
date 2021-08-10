import torch
import torch.nn as nn
import numpy as np

# need to use negative log-likelihood, NLLLoss() bc we use the LogSoftmax() function
# we would need to use CrossEntropyLoss() if we used regular Softmax()
loss_function = nn.NLLLoss(reduction='sum')

class Encoder(nn.Module):
    """
    Encoder class.
    """
    def __init__(self, hidden_size, vocab_size, embeddings=None, embedding_size=50):
        """
        Args:
            - hidden_size: desired hidden size dimension
            - vocab_size: number of unique words in all instructions
            - embeddings: if not None, these are pretrained word embeddings
                          for this dataset
            - embedding_size: if embeddings is None, this is size of embeddings
                              to train
        """
        super().__init__()
        if embeddings is not None:  # if using gLoVe
            self.emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)
        else:
            self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)        
        
    def forward(self, x):
        "Expects a batch where each element is an entire sequence"
        x = self.emb(x)
        output, (hidden, cell) = self.lstm(x)
        return output, hidden
    
class Decoder(nn.Module):
    """
    Decoder class.
    """
    def __init__(self, hidden_size, vocab_size, embeddings=None, embedding_size=50):
        """
        Args:
            - hidden_size: desired hidden size dimension
            - vocab_size: number of unique words in all instructions
            - embeddings: if not None, these are pretrained word embeddings
                          for this dataset
            - embedding_size: if embeddings is None, this is size of embeddings
                              to train
        """
        super().__init__()
        if embeddings is not None:  # if using gLoVe
            self.emb = nn.Embedding.from_pretrained(torch.tensor(embeddings), freeze=True)
        else:
            self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  # the 2nd dimension contains vector of possible word predictions
        
    def forward(self, output, hidden, cell):
        "Expects a batch where each element is a single element. Produces the output and hidden state after a single pass"
        x = self.emb(output)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.linear(output)
        output = self.softmax(output)  # apply softmax so we can calculate loss properly
        return output, (hidden, cell)
    
class SkipThought(nn.Module):
    """
    Skip-Thought model. Uses encoder - decoder architecture.
    
    The encoder is applied to the entire input sequence such that a single
    hidden state is generated for each batch element. These hidden states
    are used as the initial hidden state for the decoder for the batch.
    In order to incorporate teacher forcing, the decoder processes a single
    elememnt at a time. With each step on the target sequence, the output
    of the decoder is concatenated with all previous outputs. This results
    in a final output of shape (batch_size, seq_length, vocab_size). Hidden
    states are not useful for calculating loss, but each output hidden
    state is used as the initial hidden state for the following step.
    """
    def __init__(self, hidden_size, vocab_size, embeddings=None, embedding_size=50, teacher_forcing=1, device="cuda"):
        """
        Same Arguments as Encoder/ Decoder except for:
            - teacher_forcing: probability of enforcing teacher forcing.
                               Teacher forcing is used for the decoder
                               and is the probability that we give the 
                               correct next word to the decoder rather than 
                               the predicted next word
        """    
        super().__init__()
        self.encoder = Encoder(hidden_size, vocab_size, embeddings, embedding_size)
        self.decoder = Decoder(hidden_size, vocab_size, embeddings, embedding_size)
        self.teacher_forcing = teacher_forcing
        
    def forward(self, x, target):
        output, final_hidden = self.encoder(x)
        batch_size = x.shape[0]
                
        decoder_input = torch.zeros(batch_size, 1).long().cuda()  # intialize input as 0, index of <eos>
        decoder_cell = torch.zeros(1, batch_size, final_hidden.shape[2]).cuda()
        decoder_hidden = final_hidden


        decoder_output, (decoder_hidden, decoder_cell) = \
            self.decoder(decoder_input, decoder_hidden, decoder_cell)
        full_output = torch.clone(decoder_output)
        
        for idx in range(target.shape[1]):
            if self.teacher_forcing > np.random.random():
                next_words = target[:, idx].unsqueeze(1).long()  # LSTM needs sequence length dimension, even for 1 element
            else:
                next_words = torch.argmin(decoder_output, dim=2).long()
            
            decoder_output, (decoder_hidden, decoder_cell) = \
                self.decoder(next_words, decoder_hidden, decoder_cell)

            full_output = torch.cat([full_output, decoder_output], dim=1)  # save LogSoftmax calculations for loss
            
        return full_output