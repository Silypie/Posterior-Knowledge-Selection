import sys

import json
import os
import params
from threading import main_thread
from utils import get_data_loader
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


if __name__ == '__main__':
    # cal_length()
    # datas = json.load('./data/')
    num = 0
    samples_dirs = os.listdir(params.train_samples_path)
    for dir in samples_dirs:
        num = num + len(os.listdir(params.train_samples_path + dir))
    print(num)
    pass