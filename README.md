## PostKS (Posterior Knowledge Selection)

#### Pytorch implementation of [Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/pdf/1902.04911.pdf)
For decoder, I apply Hierarchical Gated Fusion Unit (HGFU) [[Yao et al. 2017](https://www.aclweb.org/anthology/D17-1233)] and I only use three number of knowledges for the sake of code simplicity.

<p align="center">
  <img src="https://github.com/bzantium/PostKS/blob/master/image/architecture.PNG">
</p>

<br><br>
## Requirement
- pytorch
- pytorch-nlp
- nltk
- nltk.download('punkt')

## Train PostKS model
#### If you run train, vocab.json and trained parameters will be saved. Then you can play demo.
```
$ python train.py -pre_epoch 5 -n_epoch 15 -n_batch 128
$ python train_integrated_model.py -pre_epoch 1 -n_epoch 1 -n_batch 1
$ nohup CUDA_VISIBLE_DEVICES=4,5,6  python -m torch.distributed.launch --nproc_per_node=3 train_integrated_model.py -pre_epoch 5 -n_epoch 15 -n_batch 5 > postks_trainlog.txt 2>&1 &

# 训练差异感知的模型
$ CUDA_VISIBLE_DEVICES=0,1,2  python -m torch.distributed.launch --nproc_per_node=3 train_integrated_model.py -pre_epoch 1 -n_epoch 1 -n_batch 4 -difference -train_path "data/prepared_data/test_seen.json" -train_samples_path "data/prepared_data/tmp_test_samples/"
```
## Train Seq2Seq model
```
$ CUDA_VISIBLE_DEVICES=4,5,6  python -m torch.distributed.launch --nproc_per_node=3 train_seq2seq.py -n_epoch 1 -n_batch 5
```

<br><br>
## test PostKS model
```
# -unseen for wow unseen dataset
$ python test.py -n_batch 4 -output 'postks'
```
## test Seq2Seq model
```
$ python test.py -n_batch 8 -output 'seq2seq' -model 'seq2seq'
```

<br><br>
## Play demo
```
$ python demo.py
```
#### You need to type three knowledges and utterance. Then bot will reply!
```
# example
Type first Knowledge: i'm very athletic.
Type second Knowledge: i wear contacts.
Type third Knowledge: i have brown hair.

you: hi ! i work as a gourmet cook .
bot(response): i don't like carrots . i throw them away . # reponse can change based on training.
```

## DataSet
- I only use "self_original_no_cands" in Persona-chat released by ParlAI

## Distinct
Distinct-1 and distinct-2 are measured as the total number of distinct bigrams/unigrams in the responses divided by the total number of tokens.