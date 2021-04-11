import os
import json
import torch
import torch.nn as nn
import params
import argparse
from utils import init_model, Vocabulary, build_vocab, load_data, get_data_loader
from model import PostKS, Seq2Seq, PostKS_Difference
from parlai.core.metrics import RougeMetric, BleuMetric, InterDistinctMetric, F1Metric
from rich import print


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-n_batch', type=int, default=128,
                   help='number of epochs for test')
    p.add_argument('-unseen', default=False, action='store_true',
                   help='whether test unseen dataset')
    p.add_argument('-output', type=str, default='output',
                    help='output file name')
    p.add_argument('-model_type', type=str, default='postks',
                    help='specify PostKS or Seq2Seq for testing')
    return p.parse_args()


def evaluate(model, test_loader, device, vocab, output_file_name, model_type):
    model.eval()
    rouge = {'rouge-1':0.0, 'rouge-2':0.0, 'rouge-L':0.0}
    bleu = {'bleu-1':0.0, 'bleu-2':0.0, 'bleu-3':0.0, 'bleu-4':0.0}
    knowledge_F1 = 0.0
    count = 0
    all_responses = ""

    with open(output_file_name, 'w') as f:
        for step, (src_X, _, src_K, tgt_y, last_select_knowledge) in enumerate(test_loader):
            src_X = src_X.to(device)
            if model_type != 'seq2seq':
                src_K = src_K.to(device) # [n_batch, n_knowledge, 100]
            tgt_y = tgt_y.to(device) # [n_batch, max_len]
            if model_type == 'postks_diff':
                last_select_knowledge = last_select_knowledge.to(device)

            if model_type == 'seq2seq':
                outputs = model.forward(src_X, tgt_y, 0)
            elif model_type == 'postks':
                outputs = model.forward(src_X, _, src_K, tgt_y, 0, False, True) # [max_len, n_batch, self.n_vocab]
            elif model_type == 'postks_diff':
                outputs = model.forward(src_X, _, src_K, tgt_y, 0, False, True, last_select_knowledge)
            outputs = outputs.transpose(0, 1).contiguous()  # [n_batch, max_len, self.n_vocab]

            for i in range(outputs.size(0)):
                output = outputs[i] # [max_len, self.n_vocab]
                idxs = output.max(dim=1)[1] # [max_len]

                response = ''
                for idx in idxs:
                    if idx.item() == params.EOS:
                        break
                    response += vocab.itos[idx.item()] + " "
                response = response[:-1]
                # 保留全部的回复，用于计算distinct
                all_responses += ' ' + response

                tgts = tgt_y[i] # [max_len]
                target = ''
                for tgt in tgts:
                    if tgt.item() == params.EOS:
                        break
                    target += vocab.itos[tgt.item()] + " "
                target = target[:-1]
                # 将预测句和目标句保存到文件
                f.write('response: ' + response + '\n' + 'target: ' + target + '\n\n')
                count +=1
                
                # 当前轮的所有知识字符串
                knowledges = []
                kns = src_K[i] # [n_knowledge, 100]
                for j in range(kns.size(0)):
                    kn = kns[j]
                    if kn[0] != params.EOS:
                        knowledge = ''
                        for k_idx in kn:
                            if k_idx.item() == params.PAD:
                                break
                            knowledge += vocab.itos[k_idx.item()] + ' '
                        knowledges.append(knowledge[:-1])

                # 计算rouge
                rouge_score = RougeMetric.compute_many(response, [target]) # references必须是列表
                rouge['rouge-1'] += float(rouge_score[0])
                rouge['rouge-2'] += float(rouge_score[1])
                rouge['rouge-L'] += float(rouge_score[2])

                # 计算bleu
                for k in range(1,5):
                    score = BleuMetric.compute(response, [target], k)
                    bleu['bleu-'+str(k)] += float(score)

                # 计算knowledge F1
                knowledge_F1 += float(F1Metric.compute(response, knowledges))

            # 每100个step打印一下
            if (step + 1) % 100 == 0:
                print("Step [%.4d/%.4d]: rouge-1=%.4f rouge-2=%.4f rouge-L=%.4f \
                        bleu-1=%.8f bleu-2=%.8f bleu-3=%.8f bleu-4=%.8f knowledge_F1=%.4f"
                    % (step+1, len(test_loader), rouge['rouge-1']/count, rouge['rouge-2']/count, rouge['rouge-L']/count, 
                    bleu['bleu-1']/count, bleu['bleu-2']/count, bleu['bleu-3']/count, bleu['bleu-4']/count, knowledge_F1/count))

    # 计算Distinct
    all_responses = all_responses.strip()
    distinct_1 = InterDistinctMetric.compute(text=all_responses, ngram=1).value()
    distinct_2 = InterDistinctMetric.compute(text=all_responses, ngram=2).value()
    # 最终结果
    print('Final Result: ')
    print("rouge-1=%.4f rouge-2=%.4f rouge-L=%.4f \
                bleu-1=%.8f bleu-2=%.8f bleu-3=%.8f bleu-4=%.8f \
                distinct-1=%.4f distinct-2=%.4f knowledge_F1=%.4f"
            % (rouge['rouge-1']/count, rouge['rouge-2']/count, rouge['rouge-L']/count, 
            bleu['bleu-1']/count, bleu['bleu-2']/count, bleu['bleu-3']/count, bleu['bleu-4']/count, 
            distinct_1, distinct_2, knowledge_F1/count))
    with open(output_file_name, 'a') as f:
        s = "rouge-1=%.8f rouge-2=%.8f rouge-L=%.8f \
                bleu-1=%.8f bleu-2=%.8f bleu-3=%.8f bleu-4=%.8f \
                distinct-1=%.8f distinct-2=%.8f knowledge_F1=%.8f" \
            % (rouge['rouge-1']/count, rouge['rouge-2']/count, rouge['rouge-L']/count, 
            bleu['bleu-1']/count, bleu['bleu-2']/count, bleu['bleu-3']/count, bleu['bleu-4']/count, 
            distinct_1, distinct_2, knowledge_F1/count)
        f.write(s)


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.unseen:
        test_path = params.test_unseen_path
        test_samples_path = params.test_unseen_samples_path
        output_path = params.test_unseen_output_path
    else:
        test_path = params.test_seen_path
        test_samples_path = params.test_seen_samples_path
        output_path = params.test_seen_output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_name = output_path + args.output + '.txt'

    print("loading_data...")

    if os.path.exists("vocab.json"):
        vocab = Vocabulary()
        with open('vocab.json', 'r') as fp:
            vocab.stoi = json.load(fp)

        for key, value in vocab.stoi.items():
            vocab.itos.append(key)
    else:
        train_path = params.train_path
        vocab = build_vocab(train_path, n_vocab)
    print("successfully build vocab")

    load_data(test_path, vocab, test_samples_path)
    test_sampler, test_loader = get_data_loader(test_samples_path, n_batch, nccl=False) # 暂时在单卡上测试
    print("successfully loaded")
    if args.model_type == 'postks':
        model = PostKS(n_vocab, n_embed, n_hidden, n_layer, temperature, vocab).to(device)

        # 测试时不使用数据并行模式，需对参数名进行转换，去除module. 前缀
        model = init_model(model, device, restore=params.integrated_restore, is_test=True)
        print('init model with saved parameter')

        print("start evaluating")
        evaluate(model, test_loader, device, vocab, output_file_name, model_type='postks')
    elif args.model_type == 'seq2seq':
        model = Seq2Seq(n_vocab, n_embed, n_hidden, n_layer, vocab).to(device)

        model = init_model(model, device, restore=params.seq2seq_restore, is_test=True)
        print('init model with saved parameter')

        print("start evaluating")
        evaluate(model, test_loader, device, vocab, output_file_name, model_type ='seq2seq')
    elif args.model_type == 'postks_diff':
        model = PostKS_Difference(n_vocab, n_embed, n_hidden, n_layer, temperature, vocab).to(device)

        # 测试时不使用数据并行模式，需对参数名进行转换，去除module. 前缀
        model = init_model(model, device, restore=params.difference_aware_restore, is_test=True)
        print('init model with saved parameter')

        print("start evaluating")
        evaluate(model, test_loader, device, vocab, output_file_name, model_type='postks_diff')


if __name__ == "__main__":
    main()