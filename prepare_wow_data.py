import sys

import json
import os
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk import word_tokenize

# file_names = ["test", "test_unseen", "train"]
# save_names = ["test_seen", "test_unseen", "train"]
file_names = ["test", "test_unseen", "valid"]
save_names = ["test_seen", "test_unseen", "valid"]

for key, name in zip(file_names, save_names):
    print(key)
    total_data = []

    d = json.load(open("./data/%s_collected.json" % key, 'r', encoding='utf-8'))
    for data in tqdm(d, total=len(d)):
        new_data = {}
        new_data['posts'] = data['post']
        new_data['responses'] = data['response']
        assert all(e[0] == 'no_passages_used __knowledge__ no_passages_used' for e in data['knowledge'])
        new_data['knowledges'] = [list(map(lambda x: x.split('__knowledge__')[1], e[1:])) for e in data['knowledge']]
        new_data['labels'] = data['labels']
        assert len(new_data['responses'])==len(new_data['knowledges']) and len(new_data['posts'])==len(new_data['responses'])
        total_data.append(new_data)

    json.dump(total_data, open('./data/prepared_data/%s.json' % name, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

# for name in save_names:
#     print(name)
#     f = json.load(open("./data/prepared_data/%s.json" % name, 'r'))
#     for data in tqdm(f, total=len(f)):
#         posts = data['posts']
#         responses = data['responses']
#         knowledges = data['knowledges']
#         assert len(responses)==len(knowledges)
#         for i in range(len(responses)):
            