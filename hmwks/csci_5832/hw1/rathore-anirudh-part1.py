# from nltk.tokenize import word_tokenize
#
# # Read the data
# with open('hw1_training_sets.txt', 'r') as f:
# 	lines = f.readlines()
#
# # Using nltk tokenizer word_tokenize
# all_tokens = {}
# number_of_all_tokens = 0
# for line in lines:
# 	line_to_token = word_tokenize(line)
# 	for token in line_to_token:
# 		number_of_all_tokens += 1
# 		if token not in all_tokens.keys():
# 			all_tokens[token] = 1
# 		else:
# 			all_tokens[token] += 1
#
# # Number of tokens
# print(number_of_all_tokens)
#
# # Number of types
# print(len(all_tokens.keys()))
#
# # Printing a blank 3rd line
# print()
#
# # Sort the dictionary on the basis of values
# freq_tokens = sorted(list(all_tokens.items()), key=lambda x:x[1], reverse=True)
# # print(freq_tokens)
# for i in range(5):
# 	print(freq_tokens[i][0], freq_tokens[i][1])


data_dir = "/Users/anirudh/Downloads/CSCI_5832_NLP/hw1/"
train_data_filepath = "hw1_training_sets.txt"

import os
import nltk

with open(os.path.join(data_dir, train_data_filepath)) as f:
	train_data = f.readlines()


# Part 1
token_dict = {}
total_num_tokens = 0

for idx in range(len(train_data)):
	# print(train_data[idx])
	token_list = nltk.tokenize.word_tokenize(train_data[idx])
	# print(token_list)
	for token in token_list:
		total_num_tokens = total_num_tokens + 1
		if token in token_dict.keys():
			token_dict[token] = token_dict[token] + 1
		else:
			token_dict[token] = 1

num_types = len(token_dict.keys())
print(f"The total number of tokens are {total_num_tokens}")
print(f"The total number of types are {num_types}")

# Part 2
freq_of_words = []
for token in token_dict.keys():
	word_freq = (token_dict[token], token)
	# (14, the)
	freq_of_words.append(word_freq)

# print(freq_of_words)
sorted_freq_of_words = sorted(freq_of_words, reverse=True)
# print(sorted_freq_of_words)

num_frequent_words = 5
frequent_words = []

for idx in range(num_frequent_words):
	frequent_words.append(sorted_freq_of_words[idx][1])

print("The five most frequent words are :")
print(frequent_words)
