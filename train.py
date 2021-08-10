import torch
import torch.nn as nn
import torch.optim as optim
from parser import get_parser
from SkipThoughts import SkipThought
from vocab import build_dictionary
from data_loader import get_dataloader
import gensim.downloader


parser = get_parser()
opts = parser.parse_args()

def train(opts):
	"Train the model"
	print("Creating vocabulary...", end='')
	word2idx, recipes, max_len = build_dictionary("./data/tokenized_train_text.txt")
	if opts.pretrained:  # if using gLoVe vectors
		emb_weights = "pretrained"
		word2idx, embeddings = load_embeddings()
	else:
		emb_weights = "custom"
		embeddings = None
	print("Done")
	print(f"Vocab size: {len(word2idx)}")

	print("Loading dataset...", end='')
	train_dl = get_dataloader(recipes, word2idx, max_len, opts.batch_size, opts.train_shuffle)
	print("Done")

	print("Constructing model...", end='')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	model = SkipThought(opts.hidden_size, len(word2idx), embeddings, teacher_forcing=opts.teacher_forcing)
	model.to(device)
	model.encoder.to(device)
	model.decoder.to(device)
	optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
	loss = nn.NLLLoss(reduction='sum')
	print("Done")

	print("Training model...")
	for epoch in range(opts.num_epochs):
		print(f"Epoch {epoch+1}...", end=' ')

		total_loss = 0
		for x, target in train_dl:
			x = x.to(device)
			target = target.to(device)

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

		if opts.save_checkpoints:
			print("checkpoint saved.")
			save_checkpoint(model, optimizer, total_loss, epoch, opts)


	final_fname = opts.out_path + f"finalmodel_st_{emb_weights}.pt"
	torch.save({
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, final_fname)


def save_checkpoint(model, optimizer, loss, epoch, emb_weights):
	"Saves checkpoint for model."
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

if __name__ == '__main__':
	train(opts)