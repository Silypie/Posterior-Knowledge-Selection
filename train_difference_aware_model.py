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
from model import PostKS_Difference
from torch.utils.tensorboard import SummaryWriter



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
    p.add_argument('-train_path', type=str, default="data/prepared_data/train.json")
    p.add_argument('-train_samples_path', type=str, default="data/prepared_data/diff_train_samples/")
    return p.parse_args()


def pre_train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name):
    model.train()
    parameters = list(model.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    step_num = 0

    for epoch in range(args.pre_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        b_loss = 0
        b_loss_epoch = 0
        for step, (src_X, src_y, src_K, _, last_select_knowledge) in enumerate(train_loader):
            src_X = src_X.to(device)
            src_y = src_y.to(device)
            src_K = src_K.to(device)
            last_select_knowledge = last_select_knowledge.to(device)
            step_num += 1

            optimizer.zero_grad()
            n_vocab = params.n_vocab

            k_logits = model.forward(src_X, src_y, src_K, _, args.tfr, True, False, last_select_knowledge) 
            # k_logits: [n_batch, n_vocab]

            seq_len = src_y.size(1) - 1
            k_logits = k_logits.repeat(seq_len, 1, 1).transpose(0, 1).contiguous().view(-1, n_vocab) # [n_batch*seq_len, n_vocab]
            bow_loss = NLLLoss(k_logits, src_y[:, 1:].contiguous().view(-1))
            bow_loss.backward()
            clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            b_loss += bow_loss.item()
            b_loss_epoch += bow_loss.item()
            # if args.local_rank == 0:
            #     print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f" % (epoch + 1, args.pre_epoch,
            #                                                                  step + 1, len(train_loader),
            #                                                                  b_loss))
            # b_loss = 0

            if (step + 1) % 100 == 0:
                b_loss /= 100
                if args.local_rank == 0:
                    print("Epoch [%.1d/%.1d] Step [%.4d/%.4d]: bow_loss=%.4f" % (epoch + 1, args.pre_epoch,
                                                                             step + 1, len(train_loader),
                                                                             b_loss))
                b_loss = 0
    
        # save model
        if args.local_rank == 0:
            save_model(model, params.difference_aware_restore, b_loss_epoch/len(train_loader), loss_file_name)


def train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name):
    if args.local_rank == 0:
        writer = SummaryWriter('./runs/postks')
    model.train()
    parameters = list(model.parameters())
    NLLLoss = nn.NLLLoss(reduction='mean', ignore_index=params.PAD)
    KLDLoss = nn.KLDivLoss(reduction='batchmean')
    step_num = 0

    for epoch in range(args.n_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        b_loss = 0
        k_loss = 0
        n_loss = 0
        t_loss = 0
        t_loss_epoch = 0
        for step, (src_X, src_y, src_K, tgt_y, last_select_knowledge) in enumerate(train_loader):
            src_X = src_X.to(device)
            src_y = src_y.to(device)
            src_K = src_K.to(device)
            tgt_y = tgt_y.to(device)
            last_select_knowledge = last_select_knowledge.to(device)
            step_num += 1

            optimizer.zero_grad()
            n_vocab = params.n_vocab

            prior, posterior,k_logits, outputs = model.forward(src_X, src_y, src_K, tgt_y, args.tfr, False, False, last_select_knowledge)

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
            t_loss_epoch += loss.item()
            # ToDo: KLDivLoss与另外两个loss相差太大，数量级不一样，需优化
            # Epoch [01/01] Step [0001/0258]: total_loss=14.0119 kldiv_loss=0.0183 bow_loss=6.9392 nll_loss=7.0545
            # if args.local_rank == 0:
            #         print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
            #           % (epoch + 1, args.n_epoch,
            #              step + 1, len(train_loader),
            #              t_loss, k_loss, b_loss, n_loss))
            # k_loss = 0
            # n_loss = 0
            # b_loss = 0
            # t_loss = 0

            if (step + 1) % 100 == 0:
                k_loss /= 100
                n_loss /= 100
                b_loss /= 100
                t_loss /= 100
                if args.local_rank == 0:
                    print("Epoch [%.2d/%.2d] Step [%.4d/%.4d]: total_loss=%.4f kldiv_loss=%.4f bow_loss=%.4f nll_loss=%.4f"
                      % (epoch + 1, args.n_epoch,
                         step + 1, len(train_loader),
                         t_loss, k_loss, b_loss, n_loss))
                    # 写入日志
                    writer.add_scalar('total_loss', t_loss, step_num)
                    writer.add_scalars('mini_loss', {'kldiv_loss':k_loss,
                                    'bow_loss':b_loss,
                                    'nll_loss': n_loss}, step_num)

                k_loss = 0
                n_loss = 0
                b_loss = 0
                t_loss = 0

        # save model
        if args.local_rank == 0:
            save_model(model, params.difference_aware_restore, t_loss_epoch/len(train_loader), loss_file_name)
    if args.local_rank == 0:
        writer.close()


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
    temperature = params.temperature
    train_path = args.train_path
    train_samples_path = args.train_samples_path

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
        load_data(train_path, vocab, train_samples_path)
    
    # 进程同步，防止其他进程在数据还没处理完就读取
    torch.distributed.barrier()
    train_sampler, train_loader = get_data_loader(train_samples_path, n_batch, nccl=nccl_available)

    if args.local_rank == 0:
        print("successfully loaded")

    device = torch.device(args.local_rank if torch.cuda.is_available() else "cpu")

    model = PostKS_Difference(n_vocab, n_embed, n_hidden, n_layer, temperature, vocab).to(device)
    if nccl_available:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[args.local_rank],
                                                        output_device=args.local_rank,
                                                        find_unused_parameters=True)

    if args.restore:
        model = init_model(model, device, restore=params.difference_aware_restore)
        if args.local_rank == 0:
            print('init model with saved parameter')
    # 进程同步，防止有些进程还没初始化完模型就开始训练
    torch.distributed.barrier()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_file_name = params.model_root + 'PostKS_difference_loss.json'

    # pre_train knowledge manager
    if args.local_rank == 0:
        print("start pre-training")
        # 训练开始前删除就得loss.json文件
        if os.path.exists(loss_file_name):
            os.remove(loss_file_name)

    pre_train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name)
    if args.local_rank == 0:
        print("start training")
    train(model, optimizer, train_loader, args, device, train_sampler, loss_file_name)


if __name__ == "__main__":
    main()