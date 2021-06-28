# pytorch trick



## torchvision.models

**torchvision.models**中已包含以下模型，可以直接调用：

import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
googlenet = models.googlenet()
shufflenet = models.shufflenet_v2_x1_0()
mobilenet = models.mobilenet_v2()
resnext50_32x4d = models.resnext50_32x4d()
wide_resnet50_2 = models.wide_resnet50_2()
mnasnet = models.mnasnet1_0()

通过添加参数pretrainde=True，可以加载在imagenet上预训练的模型，例：resnet18 = models.resnet18(pretrained=True)

（**imagenet中输入图片都是3通道，并且输入图片的宽高不小于224像素，并且要求输入图片像素值的范围在0到1之间，然后做一个normalization标准化。**）



torchvision官方提供的不同模型在imagenet数据集上的错误率，可作为参考：

| 网络           | Top-1 error | Top-5 error |
| :------------- | :---------- | :---------- |
| AlexNet        | 43.45       | 20.91       |
| VGG-11         | 30.98       | 11.37       |
| VGG-13         | 30.07       | 10.75       |
| VGG-16         | 28.41       | 9.62        |
| VGG-19         | 27.62       | 9.12        |
| VGG-13 with BN | 28.45       | 9.63        |
| VGG-19 with BN | 25.76       | 8.15        |
| Resnet-18      | 30.24       | 10.92       |
| Resnet-34      | 26.70       | 8.58        |
| Resnet-50      | 23.85       | 7.13        |
| Resnet-101     | 22.63       | 6.44        |
| Resnet-152     | 21.69       | 5.94        |
| SqueezeNet 1.1 | 41.81       | 19.38       |
| Densenet-161   | 22.35       | 6.2         |



## tqdm是个骚道具（待补完）

`import tqdm`

导入后tqdm可用于所有可迭代对象，故而在pytorch的dataloader的使用场景下，可以这样使用：

```python
for data, target in tqdm(train_loader):`

	`......
```

（省略号中为循环内的各种操作，可以正常对batch内的data进行后续处理，并显示进度条）

原理：这种用法相当于在dataloader上对每个batch和batch总数做的进度条



## torch训练中的需注意的点

正向传播后的梯度是储存在前一step中的，在进行backward()前要先归零，不要搞反了！

```python
	......
	# Gradients stored in the parameters in the previous step should be cleared out first.
	optimizer.zero_grad()
	# Compute the gradients for parameters.
	loss.backward()
	# Update the parameters with computed gradients.
	optimizer.step()
    ......
```



## model.parameters()与model.state_dict()

*model.parameters()*与*model.state_dict()*是Pytorch中用于查看网络参数的方法。一般来说，前者**多见于优化器的初始化**，例如：

![img](https://pic4.zhimg.com/80/v2-5c9bbd19ac058c725550d6a800ca19b7_720w.jpg)

后者**多见于模型的保存**，如：

![img](https://pic1.zhimg.com/80/v2-a52f44627d28ae6339adae1950a0de34_720w.jpg)

## sklearn中 KFold 和 StratifiedKFold 差别

**KFold划分数据集**：根据n_split直接进行顺序划分，不考虑数据label分布
**StratifiedKFold划分数据集**：划分后的训练集和验证集中类别分布尽量和原数据集一样

**example：**

```python
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.array([[10, 1], [20, 2], [30, 3], [40, 4], [50, 5], [60, 6], [70, 7], [80, 8], [90, 9], [100, 10], [90, 9], [100, 10]])

# 两个类别：1:1

Y = np.array([1,1,1,1,1,1,2,2,2,2,2,2])

print("Start Testing KFold...")

# KFold划分数据集的原理：根据n_split直接进行顺序划分

`kfolds = KFold(n_splits=3, shuffle=False)`
`for (trn_idx, val_idx) in kfolds.split(X, Y):`
    `print((trn_idx, val_idx))`
    `print((len(trn_idx), len(val_idx)))`


`print('\n' + "Start Testing StratifiedKFold...")`

# `StratifiedKFold: 抽样后的训练集和验证集的样本分类比例和原有的数据集尽量是一样的`

`stratifiedKFolds = StratifiedKFold(n_splits=3, shuffle=False)`
`for (trn_idx, val_idx) in stratifiedKFolds.split(X, Y):`
    `print((trn_idx, val_idx))`
    `print((len(trn_idx), len(val_idx)))`

```

## 数据增强库「albumentations」----值得学习

1. 我的官方地址在 github链接：

   https://github.com/albumentations-team/albumentations

2. 我的API（文档）地址在

   https://albumentations.ai/docs/

3. 我是负责处理图像的一个库，可用于所有数据类型：图像（RBG图像，灰度图像，多光谱图像），分割mask，边界框和关键点

4. 我大概有70多种不同的图像处理方法,相比torch自带的，这个库函数有更多的对图像的预处理的办法

5. 我的特点就是**快**：在相同的对图像的处理下，我的速度就是比其他处理方式更快👍

   ![图片](https://mmbiz.qpic.cn/mmbiz_png/UgCGraybEsStqiaq02J7c8qvOdLHuD4EcCOMYoHbD9dzVxibiaapeudSaZBicibXZicFcQicdJrEYWJnb20xqU02KGX0g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

   这个图中，可以看到albumentations的处理方法中，很多都是速度最快的。

6. 我可以与流行的深度学习框架（例如PyTorch和TensorFlow）一起使用。顺便说一句，我还是PyTorch生态系统的一部分

7. 对Pytorch很友好，而且这个库函数是kaggle master制作的

8. 广泛用于工业，深度学习研究，机器学习竞赛和开源项目。就是大佬都爱用的一个库，在kaggle比赛上都有我的身影。



## tensorboard 无法连接问题

切换至log目录

在terminal中输入以下命令(例)：

```python
tensorboard --logdir=D:\XXX\log --host=127.0.0.1
```



将会得到一个地址

复制地址到浏览器即可



## TIMM库——torch.models之外的可选项

相较于torch.models，TIMM（py**T**orch-**IM**age-**M**odels）是一个优秀的可选项，拥有远大于models的各个用于图像分类的预训练模型(相比之下也比较新)
