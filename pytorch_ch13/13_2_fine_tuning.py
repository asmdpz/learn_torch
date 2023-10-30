import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader

# 13.2 微调
# 13.2.1 步骤
# 1 在源数据集上预训练模型
# 2 创建目标模型并复制源模型的设计与参数
# 3 为目标模型添加输出层
# 4 在目标数据集上训练目标模型

# 13.2.2 热狗识别
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL +'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i-1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

# 裁剪随机大小和长宽比的区域，并缩放为224x224，随机左右翻转
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])
# 缩放为256x256，中心裁剪224x224
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

finetune_net = torchvision.models.resnet18(weights=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    train_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=train_augs),
        batch_size=batch_size)

    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        # 将输出层和其它层的参数分开，输出层的学习率改为10倍
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ['fc.weight', 'fc.bias']]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(),
                                  lr=learning_rate, weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


train_fine_tuning(finetune_net, 5e-5)

scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)


