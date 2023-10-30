import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image
from d2l import torch as d2l
from matplotlib_inline import backend_inline
from torch.utils.data import DataLoader

# 13.1 图像增广
d2l.set_figsize()

backend_inline.set_matplotlib_formats('svg')
plt.rcParams['figure.figsize'] = (3.5, 2.5)
img = Image.open('./img/img.jpg')
plt.imshow(img)
plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

# 随机左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
plt.show()

# 随机上下翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
plt.show()

# 随机裁剪一个面积为原始面积 10% ~ 100% 的区域，宽高比在 0.5 ~ 2 随机取值
# 宽度和高度都被缩放为200像素
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)
plt.show()

# 随机改变图像的亮度，随机值为原始图像亮度的 50% ~ 150%
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0
))
plt.show()

# 随机改变图像的亮度、对比度、饱和度和色调，随机值为原始图像亮度的 50% ~ 150%
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)
apply(img, color_aug)
plt.show()

# 结合多种图像增广方法
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,
    shape_aug
])
apply(img, augs)
plt.show()

# 13.1.2 使用图像增广进行训练
all_images = torchvision.datasets.CIFAR10(train=True, root='./data', download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='./data',train=is_train, transform=augs, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader


# 多GPU训练
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """使用多GPU进行小批量训练"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
    """使用多GPU进行小批量训练"""
    timer, num_batchs = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0,1], legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batchs // 5) == 0 or i == num_batchs - 1:
                animator.add(epoch + (i + 1) / num_batchs,
                             (metric[0] / metric[2], metric[1] / metric[3], None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


batch_size = 256
devices = d2l.try_all_gpus()
net = d2l.resnet18(10, 3)
lr = 0.001


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)


train_with_data_aug(train_augs, test_augs, net, lr)



