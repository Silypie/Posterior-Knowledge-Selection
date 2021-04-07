import os
import random
import argparse
import json
import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import params
from utils import init_model, \
    build_vocab, load_data, get_data_loader, Vocabulary, save_model
from model import Seq2Seq


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
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


def train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name):
    model.train()
    parameters = list(model.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)

    for epoch in range(args.n_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        n_loss = 0
        n_loss_epoch = 0
        for step, (src_X, src_y, src_K, tgt_y) in enumerate(train_loader):
            src_X = src_X.to(device)
            tgt_y = tgt_y.to(device)

            optimizer.zero_grad()
            n_vocab = params.n_vocab

            outputs = model.forward(src_X, tgt_y, args.tfr)
            outputs = outputs.transpose(0, 1).contiguous()
            nll_loss = NLLLoss(outputs.view(-1, n_vocab),
                               tgt_y.contiguous().view(-1))

            nll_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            n_loss += nll_loss.item()
            n_loss_epoch += nll_loss.item()
            
            if args.local_rank == 0:
                    print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: nll_loss=%.4f"
                      % (epoch + 1, args.n_epoch,
                         step + 1, len(train_loader),
                         n_loss))
            n_loss = 0

            # if (step + 1) % 50 == 0:
            #     n_loss /= 50
            #     if args.local_rank == 0:
            #         print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: nll_loss=%.4f"
            #           % (epoch + 1, args.n_epoch,
            #              step + 1, len(train_loader),
            #              n_loss))
            #     n_loss = 0

        # save model
        if args.local_rank == 0:
            save_model(model, params.integrated_restore, n_loss_epoch/len(train_loader), loss_file_name)


def main():
    nccl_available = torch.distributed.is_nccl_available()
    if nccl_available:
        torch.distributed.init_process_group("nccl", init_method='env://')
    args = parse_arguments()
    n_vocab = params.n_vocab
    n_layer = params.n_layer
    n_hidden = params.n_hidden
    n_embed = params.n_embed
    n_batch = args.n_batch
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
        # 只在主进程保存vocab
        if args.local_rank == 0:
            with open('vocab.json', 'w') as fp:
                json.dump(vocab.stoi, fp)
    if args.local_rank == 0:
        print("successfully build vocab")
        # 只在主进程里处理数据（字符转索引，对齐，分样本保存）
        load_data(train_path, vocab, params.train_samples_path)
    
    # 进程同步，防止其他进程在数据还没处理完就读取
    torch.distributed.barrier()
    train_sampler, train_loader = get_data_loader(params.train_samples_path, n_batch, nccl=nccl_available)

    if args.local_rank == 0:
        print("successfully loaded")

    device = torch.device(args.local_rank if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(n_vocab, n_embed, n_hidden, n_layer, vocab).to(device)
    if nccl_available:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    if args.restore:
        model = init_model(model, device, restore=params.seq2seq_restore)
        if args.local_rank == 0:
            print('init model with saved parameter')
    # 进程同步，防止有些进程还没初始化完模型就开始训练
    torch.distributed.barrier()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练开始前删除就得loss.json文件
    loss_file_name = params.model_root + 'Seq2Seq_loss.json'
    if os.path.exists(loss_file_name):
        os.remove(loss_file_name)
    
    if args.local_rank == 0:
        print("start training")
    train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name)


if __name__ == "__main__":
    main()