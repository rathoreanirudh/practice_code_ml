import os
import random

from hmwks.csci_5832.hw2.rathore_anirudh_part1 import calculate_sentence_perplexity, vocab_dict,\
	create_model, bigram_probability

data_dir = "/Users/anirudh/Downloads/CSCI_5832_NLP/hw2"
train_data_file = "hw2_training_sets.txt"
test_data_file = "test_set.txt"


def sample():
	# starting the sentence with <s>
	next_word = "<s>"
	word_list = []
	while next_word != "</s>":
		word_list.append(next_word)
		next_word = get_next_word(next_word)
		print(word_list)
	word_list.append(next_word)
	sentence = ' '.join(word_list)
	return word_list, sentence


def get_next_word(word):
	next_word = []
	# Take the 10 most probable words and randomly give one
	for new_word in vocab_dict.keys():
		# print(word, new_word)
		new_word_prob = bigram_probability(word_i_1=word, word_i=new_word, vocab_size=len(vocab_dict.keys()))
		next_word.append((new_word_prob, new_word))

	next_word = sorted(next_word, reverse=True)
	next_word = next_word[:10]
	print(next_word)
	next_word = random.choice(next_word)[1]
	return next_word


if __name__ == "__main__":
	with open(os.path.join(data_dir, train_data_file)) as f:
		train_data = f.readlines()

	create_model(data=train_data)

	word_list, sentence = sample()
	print(f"Sample sentence: {sentence}")
	perplexity = calculate_sentence_perplexity(word_list=word_list, vocab_size=len(vocab_dict.keys()),
											   sentence_length=len(word_list))
	print(f"Perplexity of the sample sentence: {perplexity}")