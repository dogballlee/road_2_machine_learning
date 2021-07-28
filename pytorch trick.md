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



## MNIST数据集结构

训练数据集：train-images-idx3-ubyte.gz （9.45 MB，包含60,000个样本）。
训练数据集标签：train-labels-idx1-ubyte.gz（28.2 KB，包含60,000个标签）。
测试数据集：t10k-images-idx3-ubyte.gz（1.57 MB ，包含10,000个样本）。
测试数据集标签：t10k-labels-idx1-ubyte.gz（4.43 KB，包含10,000个样本的标签）。



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



## python中的self究竟指的是啥？

Answer：实例化后的对象本身





## nn.Conv2d与nn.functional.conv2d

torch.nn.Conv2d主要是在各种组合的torch.nn.Sequential中使用，构建CNN模型。torch.nn.functional.conv2d更多是在各种自定义中使用，需要明确指出输入与权重filters参数。





## 卷积操作后的尺寸怎样计算？

输入图片大小 **W×W**
卷积核大小 **F×F**
步长 **S**
padding的像素数 **P**
于是我们可以得出计算公式为：
**N = (W − F + 2P )/S+1**

输出图片大小为 **N×N**
以resnet50为例，输入为[1,3,224,224]，其中1为batchsize，3为通道数，224为height和width。

经过第一层卷积后，其大小为[1,64,112,112]

**例：**

```python
nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
```

**解释：**
1为batch size，不改变。
对于通道数，会生成与设定的输出通道个数相同个数的卷积核，对图片进行卷积，即卷积核的个数等于输出特征图的通道数。
得到最终输出大小为[1,64,112,112]
(W − F + 2P )相当于计算除了第一次卷积后剩下的可用来卷积的大小
(W − F + 2P )/S为按照S大小的步长在刚刚得到的大小上可以向后移动多少次，即还可以做几次卷积
因为不包括第一次卷积，所以再加上一个1，
即N = (W − F + 2P )/S+1
输出大小 = （图片宽或高 - 卷积核大小 + padding大小）/ 步长 + 1
对于宽和高不同的图片可分别用上述公式计算，得到最终的输出大小。

卷积动态图解参考：
https://cs231n.github.io/assets/conv-demo/index.html



## darknet-53得名原因

darknet-53作为提取特征的backbone被使用于YOLOv3中，其来源于darknet-19（首次使用于YOLOv2，思路源于resnet），53与19分别指该网络结构中卷积层的个数（2 + 1×2 + 1 + 2×2 + 1 + 8×2 + 1 + 8×2 + 1 + 4×2 + 1 = 53 按照顺序数，<u>最后的Connected是全连接层也算卷积层</u>，一共53个）

**图1：**

![img](https://img-blog.csdn.net/20180726102742325?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**图2：**

![img](https://img-blog.csdnimg.cn/2019040211084050.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70)

在**图2**中我们能够很清晰的看到三个预测层分别来自的什么地方，以及Concatenate层与哪个层进行拼接。**注意Convolutional是指Conv2d+BN+LeakyReLU，和Darknet53图中的一样，而生成预测结果的最后三层都只是Conv2d**



## embedding(嵌入层)究竟是个啥

为了解决特征稀疏造成的一系列问题(<u>稀疏表示存在一些问题，这些问题可能使模型难以有效学习。主要问题是构造的ont-hot vector太大以及vector之间距离刻画问题</u>)，而采取的将大型稀疏向量转换为保留语义关系的低维空间的方法。常用的方法有：

​				**主成分分析（PCA）**：已用于创建单词embedding。给定一组像词袋向量一样的实例，PCA试图找到可以折叠成单个维度的高度相关的维度。

​				**Word2vec**：依赖于分布假设(distributional hypothesis)来将语义相似的词映射到几何上相近的embedding向量。



## DL常用的激活函数

![img](https://img-blog.csdnimg.cn/20210522090207129.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzQ2NTEwMjQ1,size_16,color_FFFFFF,t_70)



# @staticmethod和@classmethod的用法

**一般来说，要使用某个类的方法，需要先实例化一个对象再调用方法。**
**而使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用。**
这有利于组织代码，把某些应该属于某个类的函数给放到那个类里去，同时有利于命名空间的整洁。

区别在于：

- @staticmethod不需要表示自身对象的self和自身类的cls参数，就跟使用函数一样。

- @classmethod也不需要self参数，但第一个参数需要是表示自身类的cls参数。

    如果在@staticmethod中要调用到这个类的一些属性方法，只能直接类名.属性名或类名.方法名。
    而@classmethod因为持有cls参数，可以来调用类的属性，类的方法，实例化对象等，避免硬编码。



## 交叉熵和相对熵（KL散度）？

KL散度是两个概率分布P和Q差别的非对称性的度量。典型情况下，P表示数据的真实分布，Q表示数据的理论分布，模型分布，或P的近似分布。

![[公式]](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28P%7C%7CQ%29%3D%5Csum_iP%28i%29%5Clog%5Cfrac%7BP%28i%29%7D%7BQ%28i%29%7D%3D%5Csum_iP%28i%29%5Clog%7BP%28i%29%7D-%5Csum_iP%28i%29%5Clog%7BQ%28i%29%7D)

这里注意：由于P和Q在公式中的地位不是相等的，所以 ![[公式]](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28P%7C%7CQ%29%5Cneq+D_%7BKL%7D%28Q%7C%7CP%29)。在机器学习中，由于真实的概率分布是固定的，前半部分是个常数。那么对KL散度的优化就相当于优化后半部分。

相对熵公式的后半部分就是交叉熵

![[公式]](https://www.zhihu.com/equation?tex=CrossEntropy%3D-%5Csum_iP%28i%29%5Clog%7BQ%28i%29%7D)



## 卷积特点总结

卷积网络联合了三个架构特征导致了转换、拉伸和扭曲的不变形：

​		（1）**局部感受野**（Local Receptive Fields）；

​		（2）**共享权重**（Shared Weights）；

​		（3）**时间和空间的二次抽样**（Spatial or Temporal Subsampling）。

 

卷积的作用：

　　（1）引入稀疏或局部连接，约减不必要的权值连接

　　（2）权值共享，减少参数量

　　（3）平移不变性

　　（4）避免过拟合现象

 

池化的作用：

　　（1）减少计算量，刻画平移不变性

　　（2）约减下一层输入的维数（参数量降低）

　　（3）避免过拟合
