import torch
import torch.nn as nn
import torch.optim as optim
from parser import get_parser
from SkipThoughts import SkipThought
import gensim.downloader


parser = get_parser()
opts = parser.parse_args()

def train(opts)

	embeddings = None
	word2idx = load_word_dict()

	if opts.pretrained:
		word2idx, embeddings = load_embeddings()

	model = SkipThought(opts.hidden_size, len(word2idx), embeddings, teacher_forcing=teacher_forcing)
	optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
	loss = nn.NLLLoss(reduction='sum')

	for epoch in range(opts.num_epochs):
		print(f"Epoch {epoch+1}")

		total_loss = 0
		for x, target in train_dl:
			model.train()

			output = model(x, target)
			B, L, V = output.shape
			
			output = output.view(B, V, L)  # reshape for the loss function
			target = target.flatten()

			optimizer.zero_grad()
			batch_loss = loss(output, target)
			total_loss += batch_loss

			batch_loss.backward()
			optimizer.step()

		if opts.checkpoints:
			save_checkpoint(model, optimizer, total_loss, epoch, opts)


def save_checkpoint(model, optimizer, loss, epoch, opts):
	"Saves checkpoint for model."
	emb_weights = "custom"
	if opts.pretrained:
		emb_weights = "pretrained"
	model_fname = opts.out_path + f"model_st_{emb_weights}_epoch{epoch+1}_val{loss:.2f}.pt"

	torch.save({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'loss': loss
	}, model_fname)

def load_embeddings():
	glove = gensim.downloader.load('glove-wiki-gigaword-100')
	weights = glove.get_normed_vectors()

	return glove.key_to_index, torch.tensor(weights)

def load_word_dict():
	"Needs to be implemented."
