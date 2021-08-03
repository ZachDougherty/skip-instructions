# Skip-Instructions Model
### By: Zachary Dougherty and Wonseok Choi

## Goal
The goal of this project is to develop a PyTorch implementation of the _skip_instructions_ model from the [im2recipe-PyTorch](https://github.com/torralba-lab/im2recipe-Pytorch) repository. This repository and its [associated paper](http://pic2recipe.csail.mit.edu/) aim to develop a joint-embedding model for recipe-image pairs. The final result is a model which can produce a sequence of recipe instructions from an image of food. 

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
The [im2recipe-PyTorch](https://github.com/torralba-lab/im2recipe-Pytorch) repository luckily provides the tools for tokenizing the recipe and ingredient text.

After tokenization, we will finetune pretrained Word2Vec embeddings on our recipe dataset.

Our PyTorch Dataset will consist of input - output pairs where

input = an instruction vector where each element represents the index of the associated word.

output = the instruction vector for the next instruction in the recipe.

We also prepend each instruction vector with the start of sentence, \<sos>, token and append it with the end of sentence token, \<eos>.

## Model
Our skip-instructions model consists of a 2 stage LSTM. Since recipe instruction lengths are fairly long, a single LSTM will run into the vanishing gradient problem. The proposed architecture is to first train a skip-thought LSTM on individual instructions. A skip-thought model has an encoder-decoder architecture, and we will use an LSTM for both. The idea is to encode the sequence of word embeddings for a given instruction and decode the next instruction. Essentially, the model is learning the task of predicting the next instruction from the previous one.

After training, the hidden state for any given sentence becomes an embedding for that instruction. We can then use these embeddings as the inputs to a standard LSTM to produce recipe level embeddings. In the end, we will be able to use these embeddings for the larger joint-embedding model or for other machine learning tasks.