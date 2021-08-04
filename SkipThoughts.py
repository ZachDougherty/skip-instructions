import torch
import torch.nn as nn
import torch

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
            - seq_length: fixed length of input vectors (max instruction length)
            - vocab_size: number of unique words in all instructions
            - embeddings: if not None, these are pretrained word embeddings
                          for this dataset
            - embedding_size: if embeddings is None, this is size of embeddings
                              to train
        """
        super(self, Encoder).__init__()
        if embeddings:  # if using word2vec
            weights = embeddings.get_normed_vectors()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(weights), freeze=True)
        else:
            self.emb = nn.Embedding(vocab_size, embedding_size)
            
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
            - seq_length: fixed length of input vectors (max instruction length)
            - vocab_size: number of unique words in all instructions
            - embeddings: if not None, these are pretrained word embeddings
                          for this dataset
            - embedding_size: if embeddings is None, this is size of embeddings
                              to train
        """
        super(self, Decoder).__init__()
        if embeddings:  # if using word2vec
            weights = embeddings.get_normed_vectors()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(weights), freeze=True)
        else:
            self.emb = nn.Embedding(vocab_size, embedding_size)
            
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, proj_size=vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)  # the 2nd dimension contains vector of possible word predictions
        
    def forward(self, output, hidden):
        "Expects a batch where each element is a single element. Produces the output and hidden state after a single pass"
        x = self.emb(output)
        output, (hidden, cell) = self.lstm(output, hidden)  # proj size means we can project output to number of classes (words)
        output = self.softmax(output)  # apply softmax so we can calculate loss properly
        return output, hidden
    
    
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
    def __init__(self, hidden_size, seq_length, vocab_size=None, embeddings=None, embedding_size=50, teacher_forcing=1):
        """
        Same Arguments as EmbeddingLSTM except for:
            - teacher_forcing: probability of enforcing teacher forcing.
                               Teacher forcing is used for the decoder
                               and is the probability that we give the 
                               correct next word to the decoder rather than 
                               the predicted next word
        """    
        super().__init__()
        self.encoder = EmbeddingLSTM(hidden_size, vocab_size, embeddings, embedding_size)
        self.decoder = EmbeddingLSTM(hidden_size, vocab_size, embeddings, embedding_size)
        self.teacher_forcing = teacher_forcing
        self.seq_length = seq_length
        
    def forward(self, x, target):
        output, final_hidden = self.encoder(x)
        batch_size = x.shape[0]
                
        decoder_input = torch.zeros(batch_size, 1)  # intialize input as 0, index of <sos>
        
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        full_output = torch.clone(decoder_input)
        
        for idx in range(self.seq_length):
            if teacher_forcing > np.random.random():
                next_words = target[:, idx, :].unsqueeze(1)  # LSTM needs sequence length dimension, even for 1 element
            else:
                next_words = torch.argmin(decoder_output, dim=2)
            
            decoder_output, decoder_hidden = self.decoder(next_words, decoder_hidden)
            full_output = torch.cat([full_output, decoder_output], dim=1)  # save LogSoftmax calculations for loss
            
        return full_output

        