import torch
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

np.random.seed(0)

class InstructionDataset(Dataset):
    def __init__(self, sentences, word_dict, max_len):
        self.maxlen = max_len
        self.sentences = sentences
        self.word_dict = word_dict

    def __len__(self):
        return len(self.sentences)

    def sentence_encode(self, sentence):
        enc = [self.word_dict.get(w, self.word_dict["__unknown__"]) for w in sentence.split()][: self.maxlen - 1]
        enc += [self.word_dict["<eos>"]] * (self.maxlen - len(enc))  # add end of sentence
        enc = Variable(torch.from_numpy(np.array(enc)))
        return enc

    def __getitem__(self, idx):
        x = self.sentences[idx]
        y = self.sentences[idx+1]
        x = self.sentence_encode(x)
        y = self.sentence_encode(y)

        return x, y


def get_dataloader(text, word_dict, max_len, train_batch_size, train_shuffle):
    train_ds = InstructionDataset(text, word_dict, max_len)
    # valid_ds = InstructionDataset(text_file_path=text)

    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=train_shuffle)
    # valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=valid_shuffle)

    return train_dl
