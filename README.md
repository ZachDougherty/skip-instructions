# Skip-Instructions Model
### By: Zachary Dougherty and Wonseok Choi

## Goal
The goal of this project is to develop a PyTorch implementation of the _skip-instructions_ model from the [im2recipe-PyTorch](https://github.com/torralba-lab/im2recipe-Pytorch) repository. This repository and its [associated paper](http://pic2recipe.csail.mit.edu/) aim to develop a joint-embedding model for recipe-image pairs. The final result is a model which can produce a sequence of recipe instructions from an image of food. 

One essential part of their architecture is a _skip-instructions_ model, based on the _skip-thought_ architecture proposed [here](https://papers.nips.cc/paper/2015/file/f442d33fa06832082290ad8544a8da27-Paper.pdf). In this implementation, however, we will be using LSTM's for both the Encoder and Decoder as opposed to GRU's.

## Process
Here we detail the process of data processing, embedding, and model training.
### Data
The recipe data comes in `json` format:
```json
[
  {
    "ingredients": [
      {
        "text": "ingredient 1"
      },
      {
        "text": "ingredient 2"
      },
    ],
    "url": "source of recipe",
    "partition": "one of train, test, val",
    "title": "title of recipe",
    "id": "unique identifier",
    "instructions": [
      {
        "text": "step 1"
      },
      {
        "text": "step 2"
      },
    ]
  }
]
```
The [im2recipe-PyTorch](https://github.com/torralba-lab/im2recipe-Pytorch) repository luckily provides the tools for tokenizing the recipe and ingredient text. We have modified some and created our own vocabulary from the tokenized data in the case of custom embeddings.

After tokenization, we provide the option of using pretrained 100d gLoVe vectors or a custom embedding. The default custom embedding dimension is `50`.

Our PyTorch Dataset will consist of input - output pairs where:

input = an skip-instruction vector where each element represents the index of the associated word.

output = the skip-instruction vector for the next instruction in the recipe.

We use a vector of \<eos> or end of sentence tokens for the first input to the decoder.

## Model
The recipe embedding model consists of a 2 stage LSTM. Since recipe instruction lengths are fairly long, a single LSTM will run into the vanishing gradient problem. The proposed architecture is to first train a skip-thought LSTM on individual instructions. A skip-thought model has an encoder-decoder architecture and we will use an LSTM for both. The idea is to encode the sequence of word embeddings for a given instruction and decode the next instruction. Essentially, the model is learning the task of predicting the next instruction from the previous one.

After training, the final hidden state of the encoder for any given sentence becomes an embedding for that instruction. We can then use these embeddings as the inputs to a standard LSTM to produce recipe level embeddings.

## Training
In order to train the model, we first need to download the ingredient and recipe data from the MIT group. You can follow the instructions in their [article](http://pic2recipe.csail.mit.edu/) and download the `det_ingrs.json` and `layer1.json` files, placing them in the `data` directory.

Once the data is downloaded, run `python tokenize_instructions.py` to generate tokenized recipe instructions. This will create 3 files, `tokenized_train_text.txt`,`tokenized_test_text.txt` and `tokenized_val_text.txt` each corresponding to train, test and validation portions of the recipe data.

Once the data is created, we train the model with `python train.py`. We have included many command line options for customizing your training approach. These can be explored with `python parser.py -h`. The most important argument is `-c` or `--save-checkpoints`. Given this flag, the script will save a checkpoint of the model after each epoch in the `--outpath` directory (default is `models/`).

## Results
Unfortunately, we have not been able to get our model to run under the current architecture. We tried training the model with three different enviornments: locally with a Macbook with 16GB RAM, a Google Colab GPU runtime with 16GB RAM and finally a VM with 2x A6000 GPUs with 48GB RAM from [lambdalabs.com](https://lambdalabs.com/). For each environment, the RAM was maxed out when allocating memory for the output tensors from our SkipThought model. We believe that there are a few key reasons as to why our current design is so greedy for memory:

1) Since PyTorch LSTM's expect batch elements to be all the same length, we are forced to make every sequence as long as the longest single recipe instruction which is in our case, 165 words. We pad the remaining empty space with the \<eos> token. This also causes our model to suffer from the vanishing gradient problem. One way to avoid this issue is for each batch, find the longest sequence of non \<eos> tokens and simply remove 165 - this length from each batch element. This way our model only has to process sequences equal to the maximum length in the current batch. With shuffling of the Dataloader, our model would be much better efficient.

2) Our word vocabulary is massive, at 249,736 words, it is over 8x the size of the word2vec vocabulary. I think this number is highly overinflating the true number of unique words in our data and is most likely the result of insufficient word cleaning and tokenization. However, since our model is essentially a language generation model, we do need high and low frequency words which are typically removed from corpi in other NLP tasks. Further exploration into the techniques of the reference paper and repository are needed to solve this issue.

3) In order to bundle up our model into a single class, the output produced by our model is of shape `(batch_size, 249736, 165)`, an extremely large tensor even with small batch_sizes. Other implementations calculate the loss after the decoder LSTM predicts each next step in the sequence rather than our design which seeks to calculate the loss after each batch is processed. If we calculate loss in this way, we can remove the output tensors from memory after the loss is calculated and reduce the amount of space needed at any given time for our model.

With these adjustments, we could create a more efficient architecture and train the model to obtain our recipe instruction embeddings. We will continue to redesign and research methods to fix the issues present in the project.