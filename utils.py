import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import params
from copy import copy
from collections import Counter


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def build_vocab(path, n_vocab):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []
        word_counter = Counter()
        vocab = dict()
        reverse_vocab = dict()
        vocab['<PAD>'] = params.PAD
        vocab['<UNK>'] = params.UNK
        vocab['<SOS>'] = params.SOS
        vocab['<EOS>'] = params.EOS
        vocab_idx = len(vocab)

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n")
                for word in k_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

            elif "__SILENCE__" not in line:
                X_line = " ".join(line.split("\t")[0].split()[1:])
                y_line = line.split("\t")[1].strip("\n")

                for word in X_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

                for word in y_line.split():
                    if word in vocab:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1

        for key, _ in word_counter.most_common(n_vocab - 2):
            vocab[key] = vocab_idx
            vocab_idx += 1

        for key, value in vocab.items():
            reverse_vocab[value] = key

    return vocab, reverse_vocab


def load_data(path, vocab):
    with open(path, errors="ignore") as file:
        X = []
        K = []
        y = []
        k = []

        for line in file:
            dialog_id = line.split()[0]
            if dialog_id == "1":
                k = []

            if "your persona:" in line:
                if len(k) == 3:
                    continue
                k_line = line.split("persona:")[1].strip("\n")
                k.append(k_line)

            elif "__SILENCE__" not in line:
                K.append(k)
                X_line = " ".join(line.split("\t")[0].split()[1:])
                y_line = line.split("\t")[1].strip("\n")
                X.append(X_line)
                y.append(y_line)

    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        X_temp = []
        for word in line.split():
            if word in vocab:
                X_temp.append(vocab[word])
            else:
                X_temp.append(vocab['<UNK>'])
        X_ind.append(X_temp)

    for line in y:
        y_temp = []
        for word in line.split():
            if word in vocab:
                y_temp.append(vocab[word])
            else:
                y_temp.append(vocab['<UNK>'])
        y_ind.append(y_temp)

    for lines in K:
        K_temp = []
        for line in lines:
            k_temp = []
            for word in line.split():
                if word in vocab:
                    k_temp.append(vocab[word])
                else:
                    k_temp.append(vocab['<UNK>'])
            K_temp.append(k_temp)
        K_ind.append(K_temp)

    return X_ind, y_ind, K_ind


def get_data_loader(X, y, K, n_batch):
    dataset = PersonaDataset(X, y, K)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=n_batch,
        shuffle=True
    )
    return data_loader


class PersonaDataset(Dataset):
    def __init__(self, X, y, K):
        X_len = max([len(line) for line in X])
        y_len = max([len(line) for line in y])
        k_len = 0
        for lines in K:
            for line in lines:
                if k_len < len(line):
                    k_len = len(line)

        src_X = list()
        src_y = list()
        src_K = list()
        tgt_y = list()

        for line in X:
            line.extend([params.PAD] * (X_len - len(line)))
            src_X.append(line)

        for line in y:
            src_line = copy(line)
            tgt_line = copy(line)
            src_line.insert(0, params.SOS)
            tgt_line.append(params.EOS)
            src_line.extend([params.PAD] * (y_len - len(src_line) + 1))
            tgt_line.extend([params.PAD] * (y_len - len(tgt_line) + 1))
            src_y.append(src_line)
            tgt_y.append(tgt_line)

        for lines in K:
            src_k = list()
            for line in lines:
                line.extend([params.PAD] * (k_len - len(line)))
                src_k.append(line)
            src_K.append(src_k)

        self.src_X = torch.LongTensor(src_X)
        self.src_y = torch.LongTensor(src_y)
        self.src_K = torch.LongTensor(src_K)
        self.tgt_y = torch.LongTensor(tgt_y)
        self.dataset_size = len(self.src_X)

    def __getitem__(self, index):
        src_X = self.src_X[index]
        src_y = self.src_y[index]
        tgt_y = self.tgt_y[index]
        src_K = self.src_K[index]
        return src_X, src_y, src_K, tgt_y

    def __len__(self):
        return self.dataset_size
