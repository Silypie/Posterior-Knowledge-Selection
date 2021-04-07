PAD = 0
UNK = 1
SOS = 2
EOS = 3

n_vocab = 20000
n_layer = 2
n_hidden = 800
n_embed = 300
temperature = 0.8

train_path = "data/prepared_data/test_seen.json"
train_samples_path = "data/prepared_data/train_samples/"

test_seen_path = "data/prepared_data/test_seen.json"
test_seen_samples_path = "data/prepared_data/test_seen__samples/"
test_seen_output_path = "output/seen/"

test_unseen_path = "data/prepared_data/test_unseen.json"
test_unseen_samples_path = "data/prepared_data/test_unseen__samples/"
test_unseen_output_path = "output/unseen/"

model_root = "snapshots/"
emlayer_restore = "snapshots/PostKS-emlayer.pt"
encoder_restore = "snapshots/PostKS-encoder.pt"
Kencoder_restore = "snapshots/PostKS-Kencoder.pt"
manager_restore = "snapshots/PostKS-manager.pt"
decoder_restore = "snapshots/PostKS-decoder.pt"
all_restore=[emlayer_restore ,encoder_restore, Kencoder_restore, manager_restore, decoder_restore]

integrated_restore = "snapshots/PostKS-integrated.pt"