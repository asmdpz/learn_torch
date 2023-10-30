import json
import multiprocessing
import os
import torch
from torch import nn
from d2l import torch as d2l

from pytorch_ch15.append_15_7 import SNLIBERTDataset

# 15.7 自然语言推断：微调BERT
# 15.7.1 加载预训练的BERT
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # 定义空词表以加载预定义词表
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab


devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices
)





# 如果出现内存不足错误，请减小在原始BERT模型中的batch_size，max_len=512
batch_size, max_len, num_workers = 512, 128, 4

# data_dir = d2l.download_extract('SNLI')
# 斯坦福语料库地址已失效
data_dir = '..\\pytorch_ch15\\dataset\\snli'


# 15.7.3 微调BERT
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segment_X, valid_lens_X = inputs
        encoded_X = self.encoder(tokens_X, segment_X, valid_lens_X)
        return self.output(self.hidden(encoded_X[:, 0, :]))


if __name__ == '__main__':
    train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    # train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)
    # test_iter = torch.utils.data.DataLoader(test_set, batch_size)
    net = BERTClassifier(bert)
    lr, num_epochs = 1e-4, 1
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)









