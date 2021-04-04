import sys

import json
import os
import params
from threading import main_thread
from utils import WizardDataset, build_vocab, get_data_loader
from tqdm import tqdm
from nltk import word_tokenize

file_names = ["test"]
save_names = ["test_seen"]

def cal_length():
    for name in save_names:
        print(name)
        f = json.load(open("./data/prepared_data/%s.json" % name, 'r', encoding='utf-8'))
        le = 0
        num = 0
        x_len = 0
        y_len = 0
        for data in tqdm(f, total=len(f)):
            knowledges = data['knowledges']
            for i in range(len(knowledges)):
                num = max(num, len(knowledges[i]))
                for j in range(len(knowledges[i])):
                    le = max(le, len(word_tokenize(knowledges[i][j])))
        for data in tqdm(f, total=len(f)):
            posts = data['posts']
            for i in range(len(posts)):
                x_len = max(x_len, len(word_tokenize(posts[i])))
        for data in tqdm(f, total=len(f)):
            responses = data['responses']
            for i in range(len(responses)):
                y_len = max(y_len, len(word_tokenize(responses[i])))
        print('知识的最大长度,',le)
        print('x的最大长度,',x_len)
        print('y的最大长度,',y_len)
        print('每轮对话的最大知识数量,',num)

def origin(tgt_y, vocab):
    tgts = tgt_y # [max_len]
    target = ''
    for tgt in tgts:
        if tgt.item() == params.EOS:
            break
        target += vocab.itos[tgt.item()] + " "
    target = target[:-1]
    return target

def origin_sentence():
    dataset = WizardDataset(params.test_samples_path)
    vocab = build_vocab(params.train_path, params.n_vocab)
    for i in range(3):
        src_X, src_y, src_K, tgt_y = dataset[i]
        print('X: ',origin(src_X, vocab))
        print('y: ',origin(src_y, vocab))
        print('K: ',origin(src_K[0], vocab))
        print('tgt_y: ',origin(tgt_y, vocab))

if __name__ == '__main__':
    # cal_length()
    # datas = json.load('./data/')
    origin_sentence()
    pass