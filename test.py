import os
import json
import torch
import torch.nn as nn
import params
import argparse
from utils import init_model, Vocabulary, build_vocab, load_data, get_data_loader
from model import PostKS


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-n_batch', type=int, default=128,
                   help='number of epochs for test')
    return p.parse_args()


def evaluate(model, test_loader, device):
    model.eval()
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    total_loss = 0
    n_vocab = params.n_vocab

    for step, (src_X, _, src_K, tgt_y) in enumerate(test_loader):
        src_X = src_X.to(device)
        src_K = src_K.to(device)
        tgt_y = tgt_y.to(device)

        outputs = model.forward(src_X, _, src_K, tgt_y, 0, False, True)

        outputs = outputs.transpose(0, 1).contiguous()
        loss = NLLLoss(outputs.view(-1, n_vocab),
                           tgt_y.contiguous().view(-1))
        total_loss += loss.item()
        if step % 10 ==0:
            print("Step [%.4d/%.4d]: NLLLoss=%.4f" % (step, len(test_loader), loss.item()))

    total_loss /= len(test_loader)
    print("nll_loss=%.4f" % (total_loss))


def main():
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    test_path = params.test_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    load_data(test_path, vocab, params.test_samples_path)
    test_sampler, test_loader = get_data_loader(params.test_samples_path, n_batch, nccl=False) # 暂时在单卡上测试
    print("successfully loaded")

    model = PostKS(n_vocab, n_embed, n_hidden, n_layer, temperature, vocab).to(device)

    # 测试时不使用数据并行模式，需对参数名进行转换，去除module. 前缀
    model = init_model(model, device, restore=params.integrated_restore, is_test=True)
    print('init model with saved parameter')

    print("start evaluating")
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()