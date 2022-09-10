import numpy as np

from tqdm import tqdm

import dill as pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from pyfillet import WordEmbedder
from pyfillet import Tokenizer


class EmbeddedVocabulary:
    def __init__(self):
        self.__vocabulary = dict()
        self.__embedder = WordEmbedder()
        self.__eos_embedding = np.zeros(self.__embedder.dim)
        self.__eos_embedding[0] = 1

    @property
    def dim(self):
        return self.__embedder.dim

    @property
    def words(self) -> list:
        return list(self.__vocabulary.keys())

    def is_valid(self, word: str) -> bool:
        return (self.__embedder(word) is not None) or (word in ['.', '!', '?', ','])

    def learn(self, word: str):
        if word not in self.__vocabulary.keys():
            result = self.__embedder(word)
            if result is None:
                result = self.__eos_embedding
            self.__vocabulary[word] = result

    def word2vec(self, word: str):
        self.learn(word)
        return self.__vocabulary[word]

    def vec2word(self, word_vector):
        def softmax(x):
            e_x = np.exp(x)
            return e_x / e_x.sum()

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

        angle = np.array(
            [np.arccos(cosine_sim(word_vector, vec)) for vec in self.__vocabulary.values()], dtype=np.float32)

        probabilities = softmax(-angle)
        return np.random.choice(list(self.__vocabulary.keys()), p=probabilities)


class NNModel(nn.Module):
    def __init__(self, prefix_size: int, word_dim: int):
        super(NNModel, self).__init__()

        self.lin1 = nn.Linear(prefix_size * word_dim, 900)
        self.bn1 = nn.BatchNorm1d(900)

        self.lin2 = nn.Linear(900, 700)
        self.bn2 = nn.BatchNorm1d(700)

        self.lin3 = nn.Linear(700, 500)
        self.bn3 = nn.BatchNorm1d(500)

        self.lin4 = nn.Linear(500, 300)

        self.lin5 = nn.Linear(300, word_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)

        x = F.relu(self.lin3(x))
        x = self.bn3(x)

        x = F.relu(self.lin4(x))

        x = self.lin5(x)
        return x


class NeuralModel:
    def __init__(self, gram_length: int = 0, storage_location: str = None):
        self.__vocabulary = EmbeddedVocabulary()
        if storage_location is None:
            self.__gram_length = gram_length
            self.__model = NNModel(self.__gram_length, self.__vocabulary.dim)
        else:
            with open(storage_location + ".pkl", "rb") as file:
                model_data = pickle.load(file)
                self.__gram_length = model_data['gram length']
                self.__model = NNModel(self.__gram_length, self.__vocabulary.dim)
                self.__model.load_state_dict(model_data['model'])
                for word in model_data['known words']:
                    self.__vocabulary.learn(word)

        self.__tokenizer = Tokenizer()

    def fit(self, text: str, epochs: int, learning_rate: float = 3e-4):
        filtered_text = self.__prepare(text)
        contexts = np.lib.stride_tricks.sliding_window_view(
            filtered_text, (self.__gram_length + 1, self.__vocabulary.dim)) \
            .reshape(-1, self.__gram_length + 1, self.__vocabulary.dim)

        prefixes = torch.from_numpy(np.array([np.concatenate(context[:-1]) for context in contexts])).float()
        successors = torch.from_numpy(np.array([context[-1] for context in contexts])).float()

        train_dataset = TensorDataset(prefixes, successors)
        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        self.__train(epochs, train_dataloader, learning_rate)

    def generate(self, length: int, prefix: str = '') -> str:
        result_list = self.__parse(prefix)
        context = np.zeros((self.__gram_length, self.__vocabulary.dim), dtype=np.float32)
        for i in range(min(len(result_list), self.__gram_length)):
            context[-(i + 1)] = self.__vocabulary.word2vec(result_list[-(i + 1)])

        self.__model.eval()
        with torch.no_grad():
            for i in tqdm(range(length)):
                inp = torch.from_numpy(np.concatenate(context)).float().view(1, -1)
                word_vec = self.__model(inp).numpy()
                result_list.append(self.__vocabulary.vec2word(word_vec[0]))
                context = np.roll(context, -1, axis=0)
                context[-1] = self.__vocabulary.word2vec(result_list[-1])

        result_text = str()

        for word in result_list:
            if word not in ['.', ',', '?', '!']:
                result_text += ' '
            result_text += word

        return result_text

    def dump(self, filename: str):
        with open(filename + ".pkl", "wb") as file:
            model_data = dict()
            model_data['model'] = self.__model.state_dict()
            model_data['known words'] = self.__vocabulary.words
            model_data['gram length'] = self.__gram_length

            pickle.dump(model_data, file)

    def __parse(self, text: str) -> list:
        sentences = self.__tokenizer(text)
        parsed_text = []
        for sentence in sentences:
            for word in sentence:
                parsed_text.append(word)
        return parsed_text

    def __prepare(self, text: str) -> np.array:
        text = self.__parse(text)
        filtered_text = []
        for word in text:
            if self.__vocabulary.is_valid(word):
                filtered_text.append(self.__vocabulary.word2vec(word))

        return np.array(filtered_text, dtype=np.float32)

    def __train(self, epochs: int, data_loader: DataLoader, learning_rate):
        def cosine_loss(output, labels):
            cos = nn.CosineSimilarity()
            return torch.mean(1 - cos(output, labels))

        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} train_accuracy: {t_acc:0.4f}"
        with tqdm(desc="epoch", total=epochs) as pbar_outer:
            self.__model.train()
            optimizer = torch.optim.Adam(self.__model.parameters(), learning_rate)
            criterion = cosine_loss

            for epoch in range(epochs):
                train_loss, train_acc = self.__fit_epoch(data_loader, criterion, optimizer)

                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss, t_acc=train_acc))

    def __fit_epoch(self, train_loader: DataLoader, criterion, optimizer) -> tuple:
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for prefixes, successors in train_loader:
            optimizer.zero_grad()

            model_outputs = self.__model(prefixes)
            loss = criterion(model_outputs, successors)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * prefixes.size(0)
            running_corrects += torch.sum(nn.CosineSimilarity()(model_outputs, successors) > 0.70)
            processed_data += prefixes.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects / processed_data
        return train_loss, train_acc
