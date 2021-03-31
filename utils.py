import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import params
from copy import copy
import torch.backends.cudnn as cudnn
from collections import Counter
import nltk
import json


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


def init_model(net, restore=None):

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(), filename)
    print("save pretrained model to: {}".format(filename))


def save_models(model, filenames):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    for i in range(len(model)):
        net = model[i]
        filename = filenames[i]
        torch.save(net.state_dict(), filename)
        print("save pretrained model to: {}".format(filename))


def build_vocab(path, n_vocab):
    with open(path, errors="ignore") as file:
        datas = json.load(file)

        word_counter = Counter()
        vocab = Vocabulary()
        # vocab = dict()
        # reverse_vocab = dict()
        vocab.stoi['<PAD>'] = params.PAD
        vocab.stoi['<UNK>'] = params.UNK
        vocab.stoi['<SOS>'] = params.SOS
        vocab.stoi['<EOS>'] = params.EOS

        initial_vocab_size = len(vocab.stoi)
        vocab_idx = initial_vocab_size

        for data in datas:
            posts = data['posts']
            responses = data['responses']
            knowledges = data['knowledges']

            for i in range(len(posts)):
                post_tokens = nltk.word_tokenize(posts[i])
                response_tokens = nltk.word_tokenize(responses[i])
                knowledge_tokens = []
                for knowledge in knowledges[i]:
                    k = nltk.word_tokenize(knowledge)
                    knowledge_tokens = knowledge_tokens + k
                    
                for word in post_tokens + response_tokens + knowledge_tokens:
                    if word in vocab.itos:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1
            
        for key, _ in word_counter.most_common(n_vocab - initial_vocab_size):
            vocab.stoi[key] = vocab_idx
            vocab_idx += 1

        for key, value in vocab.stoi.items():
            vocab.itos.append(key)

    return vocab


def load_data(path, vocab):
    with open(path, errors="ignore") as file:
        datas = json.load(file)
        X = []
        K = []  # 二维，[轮数, 知识数]
        y = []

        for data in datas:  #每个对话
            posts = data['posts']
            responses = data['responses']
            knowledges = data['knowledges']
            # 每轮对话
            for i in range(len(posts)):
                # 并没有考虑对话历史，只对每轮对话进行知识选择和回复生成
                # ToDo：跟DiffKS一样可以只使用上一轮对话和当前轮的问题，拼接X：[x_t-1,y_t-1,x_t]
                # 如果当前是一次对话的开始，那么X不用拼接，否则可以在这里修改拼接上一轮对话内容
                X.append(posts[i])
                y.append(responses[i])
                K.append(knowledges[i]) # K的每个元素是一轮对话涉及到的所有知识

    # 将token转换为index
    X_ind = []
    y_ind = []
    K_ind = []

    for line in X:
        X_temp = []
        tokens = nltk.word_tokenize(line)
        for word in tokens:
            if word in vocab.stoi:
                X_temp.append(vocab.stoi[word])
            else:
                X_temp.append(vocab.stoi['<UNK>'])
        X_ind.append(X_temp)

    for line in y:
        y_temp = []
        tokens = nltk.word_tokenize(line)
        for word in tokens:
            if word in vocab.stoi:
                y_temp.append(vocab.stoi[word])
            else:
                y_temp.append(vocab.stoi['<UNK>'])
        y_ind.append(y_temp)

    for lines in K: # 每轮的所有知识
        K_temp = []
        for line in lines:  # 每个知识
            k_temp = []
            tokens = nltk.word_tokenize(line)
            for word in tokens:
                if word in vocab.stoi:
                    k_temp.append(vocab.stoi[word])
                else:
                    k_temp.append(vocab.stoi['<UNK>'])
            K_temp.append(k_temp)
        K_ind.append(K_temp)    #二维张量：[轮数, 知识数]

    return X_ind, y_ind, K_ind


def get_data_loader(X, y, K, n_batch):
    dataset = WizardDataset(X, y, K)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=n_batch,
        shuffle=True
    )
    return data_loader


class Vocabulary:
    def __init__(self):
        self.itos = list()
        self.stoi = dict()


class WizardDataset(Dataset):
    def __init__(self, X, y, K):
        X_len = max([len(line) for line in X])  # 最大的序列长度
        y_len = max([len(line) for line in y])
        k_len = 0
        num_k = max([len(line) for line in K])  # 最多知识数
        for lines in K:
            for line in lines:
                if k_len < len(line):
                    k_len = len(line)

        src_X = list()
        src_y = list()
        src_K = list()
        tgt_y = list()

        # 所有样本进行对齐操作
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

        for lines in K: # 每轮对话
            src_k = list()
            for line in lines: # 每个知识
                line.extend([params.PAD] * (k_len - len(line)))
                src_k.append(line)
            gap = num_k - len(src_k)
            knowledge_pad = [params.PAD] * k_len
            for i in range(gap):
                src_k.append(knowledge_pad)
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


def knowledgeToIndex(K, vocab):
    k1, k2, k3 = K
    K1 = []
    K2 = []
    K3 = []

    tokens = nltk.word_tokenize(k1)
    for word in tokens:
        if word in vocab.stoi:
            K1.append(vocab.stoi[word])
        else:
            K1.append(vocab.stoi["<UNK>"])

    tokens = nltk.word_tokenize(k2)
    for word in tokens:
        if word in vocab.stoi:
            K2.append(vocab.stoi[word])
        else:
            K2.append(vocab.stoi["<UNK>"])

    tokens = nltk.word_tokenize(k3)
    for word in tokens:
        if word in vocab.stoi:
            K3.append(vocab.stoi[word])
        else:
            K3.append(vocab.stoi["<UNK>"])

    K = [K1, K2, K3]
    seq_len = max([len(k) for k in K])

    K1.extend([0] * (seq_len - len(K1)))
    K2.extend([0] * (seq_len - len(K2)))
    K3.extend([0] * (seq_len - len(K3)))

    K1 = torch.LongTensor(K1).unsqueeze(0)
    K2 = torch.LongTensor(K2).unsqueeze(0)
    K3 = torch.LongTensor(K3).unsqueeze(0)
    K = torch.cat((K1, K2, K3), dim=0).unsqueeze(0).cuda()  # K: [1, 3, seq_len]
    return K