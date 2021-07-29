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
		"ingredients" : [
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