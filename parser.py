import argparse

def get_parser():
	parser = argparse.ArgumentParser(description="Train the Skip-Thoughts model.")

	parser.add_argument("--num-epochs", default=5, type=int,
		help="Number of epochs to train.")
	parser.add_argument("--batch-size", default=2500, type=int,
		help="Batch size, sometimes called B in the code.")
	parser.add_argument("--learning-rate", default=0.001, type=float,
		help="Learning rate for the optimizer.")
	parser.add_argument("-o", "--out-path", default='models/', type=str,
		help="Path to directory where models will be stored.")
	parser.add_argument("--pretrained" default=True, type=bool,
		help="Boolean, whether or not to used pretrained word embeddings.")
	parser.add_argument("-c","--checkpoints", default=True, type=bool,
		help="Boolean, whether or not to save model checkpoints for each epoch.")
	parser.add_argument("-h","--hidden-size", default=150, type=int,
		help="Integer, size of hidden state in LSTMs")
	parser.add_argument("-t","--teacher-forcing", default=1.0, type=float,
		help="The probability of teacher forcing for each element of decoded sequence.")

	return parser