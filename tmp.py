import sys

import json
import os
import params
from threading import main_thread
from utils import WizardDataset, build_vocab, get_data_loader
from tqdm import tqdm
from nltk import word_tokenize
import rouge
import re
from parlai.core.metrics import RougeMetric, BleuMetric
from torchtext.data.metrics import bleu_score

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

def cal_rouge():
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)

    hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
    references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians."]

    hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
    references_2 = ["Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"]

    all_hypothesis = [hypothesis_1, hypothesis_2]
    all_references = [references_1, references_2]

    print(evaluator.get_scores(normalize_answer(hypothesis_1), normalize_answer(references_1[0])))
    print('*'*40)
    s = RougeMetric.compute_many(hypothesis_1, references_1) # references必须是列表
    print(s)
    print('上面可以证明py-rouge与parl计算rouge的结果相同，parl只能一次计算一对')
    print('1 的结果：', evaluator.get_scores(hypothesis_1, references_1[0]))
    print('2 的结果：', evaluator.get_scores(hypothesis_2, references_2[0]))
    print('总的结果：', evaluator.get_scores(all_hypothesis, all_references))
    print('可以看出多组数据直接取平均即可')

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

def cal_bleu():
    hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
    references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians."]
    for k in range(1,5):
        score = BleuMetric.compute(hypothesis_1, references_1, k)
        print(float(score))
    print('*'*50)
    candidate_corpus = [hypothesis_1.split(' ')]
    references_corpus = [[references_1[0].split(' ')]]
    # print(candidate_corpus)
    # print(references_corpus)
    for k in range(1,5):
        weights = [1 / k for _ in range(k)]
        print(bleu_score(candidate_corpus, references_corpus, max_n=k,weights=weights))

if __name__ == '__main__':
    cal_length()
    # datas = json.load('./data/')
    # origin_sentence()
    # cal_rouge()
    # cal_bleu()
    pass