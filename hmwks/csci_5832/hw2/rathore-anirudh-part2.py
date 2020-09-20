import os
import numpy as np

data_dir = "/Users/anirudh/Downloads/CSCI_5832_NLP/hw2"
train_data_file = "hw2_training_sets.txt"
test_data_file = "test_set.txt"

vocab_dict = {}
bigram_dict = {}


def bigram_probability(word_i_1, word_i):
	try:
		return bigram_dict[(word_i_1, word_i)] / vocab_dict[word_i_1]
	except Exception as e:
		print(e)
		return 0.0


def sentence_probability(word_list):
	prob = 1.0
	for i in range(1, len(word_list)):
		prob_bigram = bigram_probability(word_i_1=word_list[i-1], word_i=word_list[i])
		prob *= prob_bigram
	return prob


def calculate_sentence_perplexity(word_list):
	log_perplexity = 0.0
	for i in range(1, len(word_list)):
		prob_bigram = bigram_probability(word_i_1=word_list[i-1], word_i=word_list[i])
		log_perplexity += np.log10(prob_bigram) / len(word_list)
	log_perplexity = -log_perplexity
	perplexity = 10**log_perplexity
	return perplexity


def normalized_sentence_probability(prob, sentence_length):
	return prob/sentence_length


def create_vocab(sent):
	for word in sent:
		try:
			vocab_dict[word] += 1.0
		except Exception as e:
			print(f"Seeing word: {word} =| for the first time! Adding to vocab.")
			vocab_dict[word] = 1.0


def create_bigram_vocab(word_list):
	for i in range(1, len(word_list)):
		bigram = (word_list[i-1], word_list[i])
		try:
			bigram_dict[bigram] += 1.0
		except Exception as e:
			print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
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
				print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			# Create bigram with [word <UNK>]
			bigram = (word, "<UNK>")
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			# Add two occurence of <UNK> in the vocab
			vocab_dict["<UNK>"] += 2.0
		elif word == "<s>":
			bigram = (word, "<UNK>")
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			vocab_dict["<UNK>"] += 1.0
		elif word == "</s>":
			bigram = ("<UNK>", word)
			try:
				bigram_dict[bigram] += 1.0
			except Exception as e:
				print(f"Seeing bigram: {bigram} =| for the first time! Adding to vocab.")
				bigram_dict[bigram] = 1.0
			vocab_dict["<UNK>"] += 1.0


def create_model(data):
	for sentence in data:
		print(f"Adding sentence: {sentence} to the model count")
		# Splitting the pre-tokenized training data wo create a word list and appending start and end token
		word_list = sentence.split()
		word_list.append("</s>")
		word_list.insert(0, "<s>")
		print(word_list)
		create_vocab(word_list)
		create_bigram_vocab(word_list)
	add_unk_to_vocab()
	print("Model Created and Loaded!")


def fill_unk_in_test_sentence(word_list):
	for i in range(len(word_list)):
		if word_list[i] not in vocab_dict.keys():
			word_list[i] = "<UNK>"
	return word_list


def evaluate_model(data):
	for sentence in data:
		print(f"Evaluating sentence: {sentence} from the bigram language model")
		word_list = sentence.split()
		word_list.append("</s>")
		word_list.insert(0, "<s>")
		word_list = fill_unk_in_test_sentence(word_list=word_list)
		prob = sentence_probability(word_list=word_list)
		normalized_prob = normalized_sentence_probability(prob=prob, sentence_length=len(word_list))
		perplexity = calculate_sentence_perplexity(word_list=word_list)
		print(f"Probability: {prob} | Normalized probability: {normalized_prob} | Perplexity: {perplexity}")


if __name__ == "__main__":
	with open(os.path.join(data_dir, train_data_file)) as f:
		train_data = f.readlines()

	with open(os.path.join(data_dir, test_data_file)) as f:
		test_data = f.readlines()

	create_model(data=train_data)
	evaluate_model(data=test_data)


# https://en.wikipedia.org/wiki/Additive_smoothing