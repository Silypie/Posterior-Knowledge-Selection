import os
import random
import argparse
import json
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import params
from utils import init_model, save_models, \
    build_vocab, load_data, get_data_loader, Vocabulary, save_model
from model import EmbeddingLayer, Encoder, KnowledgeEncoder, Decoder, Manager, PostKS



def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-pre_epoch', type=int, default=5,
                   help='number of epochs for pre_train')
    p.add_argument('-n_epoch', type=int, default=15,
                   help='number of epochs for train')
    p.add_argument('-n_batch', type=int, default=128,
                   help='number of batches for train')
    p.add_argument('-lr', type=float, default=5e-4,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    p.add_argument('-tfr', type=float, default=0.5,
                   help='teacher forcing ratio')
    p.add_argument('-restore', default=False, action='store_true',
                   help='whether restore trained model')
    p.add_argument("--local_rank", type=int, default=0, help="multi gpu traning")
    return p.parse_args()


def pre_train(model, optimizer, train_loader, args, device, train_sampler):
    model.train()
    parameters = list(model.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)

    for epoch in range(args.pre_epoch):
        train_sampler.set_epoch(epoch)
        b_loss = 0
        for step, (src_X, src_y, src_K, _) in enumerate(train_loader):
            src_X = src_X.to(device)
            src_y = src_y.to(device)
            src_K = src_K.to(device)

            optimizer.zero_grad()
            n_vocab = params.n_vocab

            k_logits = model.pre_forward(src_X, src_y, src_K) # k_logits: [n_batch, n_vocab]

            seq_len = src_y.size(1) - 1
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab) # [n_batch*seq_len, n_vocab]
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))
            bow_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            b_loss += bow_loss.item()
            if args.local_rank == 0:
                print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f" % (epoch + 1, args.pre_epoch,
                                                                             step + 1, len(train_loader),
                                                                             b_loss))
            b_loss = 0

            # if (step + 1) % 50 == 0:
            #     b_loss /= 50
            #     if args.local_rank == 0:
            #         print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f" % (epoch + 1, args.pre_epoch,
            #                                                                  step + 1, len(train_loader),
            #                                                                  b_loss))
            #     b_loss = 0
    
    # save model
    if args.local_rank == 0:
        save_model(model, params.integrated_restore)


def train(model, optimizer, train_loader, args, device, train_sampler):
    model.train()
    parameters = list(model.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(args.n_epoch):
        train_sampler.set_epoch(epoch)
        b_loss = 0
        k_loss = 0
        n_loss = 0
        t_loss = 0
        for step, (src_X, src_y, src_K, tgt_y) in enumerate(train_loader):
            src_X = src_X.to(device)
            src_y = src_y.to(device)
            src_K = src_K.to(device)
            tgt_y = tgt_y.to(device)

            optimizer.zero_grad()
            n_vocab = params.n_vocab

            prior, posterior,k_logits, outputs = model.forward(src_X, src_y, src_K, tgt_y, args.tfr)

            kldiv_loss = KLDLoss(prior, posterior.detach())

            seq_len = src_y.size(1) - 1
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab)
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))
            
            outputs = outputs.transpose(0, 1).contiguous()
            nll_loss = NLLLoss(outputs.view(-1, n_vocab),
                               tgt_y.contiguous().view(-1))

            loss = kldiv_loss + nll_loss + bow_loss
            loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            b_loss += bow_loss.item()
            k_loss += kldiv_loss.item()
            n_loss += nll_loss.item()
            t_loss += loss.item()
            if args.local_rank == 0:
                    print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
                      % (epoch + 1, args.n_epoch,
                         step + 1, len(train_loader),
                         t_loss, k_loss, b_loss, n_loss))
            k_loss = 0
            n_loss = 0
            b_loss = 0
            t_loss = 0

            # if (step + 1) % 50 == 0:
            #     k_loss /= 50
            #     n_loss /= 50
            #     b_loss /= 50
            #     t_loss /= 50
            #     if args.local_rank == 0:
            #         print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
            #           % (epoch + 1, args.n_epoch,
            #              step + 1, len(train_loader),
            #              t_loss, k_loss, b_loss, n_loss))
            #     k_loss = 0
            #     n_loss = 0
            #     b_loss = 0
            #     t_loss = 0

        # save model
        if args.local_rank == 0:
            save_model(model, params.integrated_restore)


def main():
    torch.distributed.init_process_group("nccl", init_method='env://')
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
    temperature = params.temperature
    train_path = params.train_path

    if args.local_rank == 0:
        print("loading_data...")
    # 训练时加载处理好的词典（如果有的话）
    if os.path.exists("vocab.json"):
        vocab = Vocabulary()
        with open('vocab.json', 'r') as fp:
            vocab.stoi = json.load(fp)

        for key, value in vocab.stoi.items():
            vocab.itos.append(key)
    else:
        vocab = build_vocab(train_path, n_vocab)
        # save vocab
        with open('vocab.json', 'w') as fp:
            json.dump(vocab.stoi, fp)
    if args.local_rank == 0:
        print("successfully build vocab")

    load_data(train_path, vocab, params.train_samples_path)
    train_sampler, train_loader = get_data_loader(params.train_samples_path, n_batch, shuffle=False)
    if args.local_rank == 0:
        print("successfully loaded")

    device = torch.device(args.local_rank if torch.cuda.is_available() else "cpu")

    model = PostKS(n_vocab, n_embed, n_hidden, n_layer, temperature, vocab).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)

    if args.restore:
        model = init_model(model, device, restore=params.integrated_restore)
        if args.local_rank == 0:
            print('init model with saved parameter')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # pre_train knowledge manager
    if args.local_rank == 0:
        print("start pre-training")
    # pre_train(model, optimizer, train_loader, args, device, train_sampler)
    if args.local_rank == 0:
        print("start training")
    train(model, optimizer, train_loader, args, device, train_sampler)

    # save final model
    if args.local_rank == 0:
        save_model(model, params.integrated_restore)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

