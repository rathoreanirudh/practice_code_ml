import os
import numpy as np

data_dir = "/Users/anirudh/Downloads/CSCI_5832_NLP/hw2"
train_data_file = "hw2_training_sets.txt"
test_data_file = "test_set.txt"

vocab_dict = {}
bigram_dict = {}

# Alpha value for additive smoothing
ALPHA = 1e-9


def bigram_probability(word_i_1, word_i, vocab_size):
	try:
		numerator = bigram_dict[(word_i_1, word_i)] + ALPHA
		denominator = vocab_dict[word_i_1] + (vocab_size*ALPHA)
		return numerator / denominator
	except Exception as e:
		numerator = 0 + ALPHA
		denominator = vocab_dict[word_i_1] + (vocab_size*ALPHA)
		return numerator / denominator


def sentence_probability(word_list, vocab_size):
	# Introducing Laplace smoothing or additive smoothing
	# https://en.wikipedia.org/wiki/Additive_smoothing
	log_prob = 0.0
	for i in range(1, len(word_list)):
		log_prob += np.log(bigram_probability(word_i_1=word_list[i-1], word_i=word_list[i], vocab_size=vocab_size))
	prob = 10**log_prob
	return prob


def calculate_sentence_perplexity(word_list, vocab_size, sentence_length):
	log_perplexity = 0.0
	for i in range(1, len(word_list)):
		prob_bigram = bigram_probability(word_i_1=word_list[i-1], word_i=word_list[i], vocab_size=vocab_size)
		log_perplexity += np.log10(prob_bigram)
	log_perplexity = -log_perplexity
	log_perplexity = log_perplexity/sentence_length
	perplexity = 10**log_perplexity
	return perplexity


def normalized_sentence_probability(word_list, vocab_size, sentence_length):
	log_prob = 0.0
	for i in range(1, len(word_list)):
		log_prob += np.log(bigram_probability(word_i_1=word_list[i-1], word_i=word_list[i], vocab_size=vocab_size))
	log_prob = log_prob/sentence_length
	normalized_prob = 10**log_prob
	return normalized_prob


def create_vocab(sent):
	for word in sent:
		try:
			vocab_dict[word] += 1.0
		except Exception as e:
			# print(f"Seeing word: {word} =| for the first time! Adding to vocab.")
			vocab_dict[word] = 1.0


def create_bigram_vocab(word_list):
	for i in range(1, len(word_list)):
		bigram = (word_list[i-1], word_list[i])
		try:
			bigram_dict[bigram] += 1.0
		except Exception as e:
			# print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
			bigram_dict[bigram] = 1.0


def add_unk_to_vocab():
	# Adding UNK to bigram matrix
	vocab_dict["<UNK>"] = 0
	for word in vocab_dict.keys():
		if word != "<s>" and word != "</s>":
			# Create bigram with [<UNK> word]
			bigram = ("<UNK>", word)
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				# print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			# Create bigram with [word <UNK>]
			bigram = (word, "<UNK>")
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				# print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			# Add two occurence of <UNK> in the vocab
			vocab_dict["<UNK>"] += 2.0
		elif word == "<s>":
			bigram = (word, "<UNK>")
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				# print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			vocab_dict["<UNK>"] += 1.0
		elif word == "</s>":
			bigram = ("<UNK>", word)
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				# print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			vocab_dict["<UNK>"] += 1.0


def create_model(data):
	for sentence in data:
		# print(f"Adding sentence: {sentence} to the model count")
		# Splitting the pre-tokenized training data wo create a word list and appending start and end token
		word_list = sentence.split()
		word_list.append("</s>")
		word_list.insert(0, "<s>")
		# print(word_list)
		create_vocab(word_list)
		create_bigram_vocab(word_list)
	add_unk_to_vocab()
	# print("Model Created and Loaded!")


def fill_unk_in_test_sentence(word_list):
	for i in range(len(word_list)):
		if word_list[i] not in vocab_dict.keys():
			word_list[i] = "<UNK>"
	return word_list


def evaluate_model(data, vocab_size):
	for sentence in data:
		print(f"Evaluating sentence: {sentence}")
		# Splitting the pre-tokenized sentence on space
		word_list = sentence.split()
		word_list.append("</s>")
		word_list.insert(0, "<s>")
		word_list = fill_unk_in_test_sentence(word_list=word_list)
		prob = sentence_probability(word_list=word_list, vocab_size=vocab_size)
		normalized_prob = normalized_sentence_probability(word_list=word_list, sentence_length=len(word_list),
														  vocab_size=len(word_list))
		perplexity = calculate_sentence_perplexity(word_list=word_list, sentence_length=len(word_list),
												   vocab_size=len(word_list))
		print(f"Probability: {prob}, Normalized probability: {normalized_prob}, Perplexity: {perplexity}")


if __name__ == "__main__":
	with open(os.path.join(data_dir, train_data_file)) as f:
		train_data = f.readlines()

	with open(os.path.join(data_dir, test_data_file)) as f:
		test_data = f.readlines()

	create_model(data=train_data)
	vocab_size = len(vocab_dict.keys())
	evaluate_model(data=test_data, vocab_size=vocab_size)
