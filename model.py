import string
import dill as pickle

from tqdm import tqdm

import numpy as np


class NGramModel:
    def __init__(self, file_name: str = None):
        if file_name is None:
            self.__vocabulary_hash_to_word = dict()
            self.__vocabulary_word_to_hash = dict()
            self.__learned_heuristic = dict()
            self.__gram_length = 0
        else:
            with open(file_name + ".pkl", "rb") as file:
                model_data = pickle.load(file)
                self.__vocabulary_word_to_hash = model_data["vocabulary"]

                self.__vocabulary_hash_to_word = dict()
                for word in self.__vocabulary_word_to_hash.keys():
                    self.__vocabulary_word_to_hash[word] = self.__vocabulary_word_to_hash[word]
                    self.__vocabulary_hash_to_word[self.__vocabulary_word_to_hash[word]] = word

                self.__learned_heuristic = model_data["learned_heuristic"]
                self.__gram_length = model_data["gram_length"]

    def fit(self, text: str) -> None:
        if self.__gram_length <= 0:
            raise KeyError

        ######################### prepare text
        labeled_text = self.__prepare(text)

        ######################## learn from text
        contexts = np.lib.stride_tricks.sliding_window_view(labeled_text, self.__gram_length + 1)
        self.__learn(contexts)

    def generate(self, length: int, prefix: str = None) -> str:
        self.__to_probabilities()
        generated_text = list()
        context = np.ndarray([self.__gram_length])

        if prefix != None:
            prefix_labeled = self.__prepare(prefix)
            for word_label in prefix_labeled:
                generated_text.append(word_label)
                context = np.roll(context, -1)
                context[-1] = generated_text[-1]

        word_count = 0
        for i in tqdm(range(length)):
            new_word = np.random.choice(list(self.__vocabulary_word_to_hash.values()))
            for size in range(self.__gram_length):
                context_tuple = tuple(context[size:])

                if context_tuple in self.__learned_heuristic.keys():
                    new_word = np.random.choice(list(self.__learned_heuristic[context_tuple].keys()),
                                                p=list(self.__learned_heuristic[context_tuple].values()))
                    break

            generated_text.append(new_word)

            context = np.roll(context, -1)
            context[-1] = generated_text[-1]

        return self.__interpret_generated_text(generated_text)

    def dump(self, filename: str) -> None:
        with open(filename + ".pkl", "wb") as file:
            stored_data = dict()
            stored_data["gram_length"] = self.__gram_length
            stored_data["vocabulary"] = self.__vocabulary_word_to_hash
            stored_data["learned_heuristic"] = self.__learned_heuristic
            pickle.dump(stored_data, file)

    def set_gram_length(self, gram_length: int):
        self.__gram_length = gram_length

    def __interpret_generated_text(self, generated_text: list) -> str:
        result = str()
        for word_hash in generated_text:
            word = self.__vocabulary_hash_to_word[word_hash]
            if word not in string.punctuation + '\n' and len(result) != 0:
                result += " "
            result += word

        return result

    def __add_vocabulary(self, tokenized_text):
        for token in tokenized_text:
            if token not in self.__vocabulary_word_to_hash.keys():
                self.__vocabulary_word_to_hash[token] = len(self.__vocabulary_word_to_hash)
                self.__vocabulary_hash_to_word[len(self.__vocabulary_hash_to_word)] = token

    def __cleanse_text(text: str):
        copy = text.lower()
        return copy

    def __tokenize(text: str):
        # make commas and dots a word
        copy = str(text)
        for punctuation in string.punctuation + "\n":
            copy = copy.replace(punctuation, " " + punctuation + " ")

        copy = copy.replace('\n', "newlinesymbol")
        tokens = copy.split()
        for i in range(len(tokens)):
            if tokens[i] == "newlinesymbol":
                tokens[i] = '\n'

        return tokens

    def __label(self, text):
        labels = np.ndarray([len(text)])
        for idx, word in enumerate(text):
            labels[idx] = self.__vocabulary_word_to_hash[word]
        return labels

    def __prepare(self, text):
        text = NGramModel.__cleanse_text(text)
        text = NGramModel.__tokenize(text)
        self.__add_vocabulary(text)
        return self.__label(text)

    def __learn(self, contexts):
        for chunk in contexts:
            word = chunk[-1]
            for size in range(self.__gram_length):
                context = tuple(chunk[self.__gram_length - size:-1])

                if context not in self.__learned_heuristic.keys():
                    self.__learned_heuristic[context] = dict()

                if word not in self.__learned_heuristic[context].keys():
                    self.__learned_heuristic[context][word] = 0

                self.__learned_heuristic[context][word] += 1

    def __to_probabilities(self):
        for context in self.__learned_heuristic.keys():
            total_cases = np.sum(list(self.__learned_heuristic[context].values()))
            for successor in self.__learned_heuristic[context].keys():
                self.__learned_heuristic[context][successor] /= total_cases
