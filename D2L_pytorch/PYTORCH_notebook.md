# **PYTORCH**

## autograd：自动求导

所有基于pytorch构建的神经网络的核心

autograd包为张量上的所有操作提供自动求导机制，define-by-run

## 张量--torch.Tensor

<u>autograd包的核心类</u>

 一个多维数组，支持诸如`backward()`等的自动求导操作，同时也保存了张量的梯度。

将其属性**.requires_grad**设置为True时，autograd包会追踪该张量的所有操作，计算完成后可通过调用**.backward()**来自动计算所有梯度，该张量的所有梯度将会自动累加到**.grad()**属性

调用**.detach()**方法可以阻止张量被跟踪，防止它在未来的计算中被跟踪

**with torch.no_grad():** 的作用：评估模型好坏时，模型中的.requires_grad可能为True，此时若不需要进行梯度计算，可将该代码块包装在其中，防止跟踪历史记录(同时避免占用内存)

**autograd.Function：**实现了自动求导前向和反向传播的定义，每个Tensor至少创建一个Function节点，该节点连接到创建Tensor的函数并对其历史进行编码。<u>**Tensor**与**Function**相互连接形成一个无圈图(acyclic graph)，其中记录了完整的计算历史过程。</u>每个张量经过运算后都获得一个**.grad_fn**属性，该属性<u>引用了创建Tensor自身的Function</u>，记录了该段计算方式

**.backward()**：可用于计算梯度--标量不需要指定参数，高维张量需要指定一个gradient参数

**.requires_grad_()**可以原地改变现有张量的requires_grad标志(默认值是False)，如：

```python
a.requires_grad_(True)
```

综上，torch.autograd可以看作是计算一段向量计算过程对应的Jacobian矩阵的工具：

流程归纳为：<u>设置正向计算过程，带入X，y，调用.backward()反向传播得出y对于X的导数</u>



**Tips：**torch.rand和torch.randn有什么区别？

torch.rand 均匀分布--区间[0, 1)的均匀分布中抽取的一组随机数

torch.randn标准正态分布--返回一组（均值为0，方差为1，即高斯白噪声）随机数





## 神经网络



一个神经网络的典型训练过程如下：

- 定义包含一些可学习参数(或者叫权重）的神经网络
- 在输入数据集上迭代
- 通过网络处理输入
- 计算loss(输出和正确答案的距离）
- 将梯度反向传播给网络的参数
- 更新网络的权重，一般使用一个简单的规则：`weight = weight - learning_rate * gradient`



*以上过程可简单概括为：*

<u>定义一个神经网络>>>处理输入>>>调用backward>>>计算损失>>>更新网络权重</u>



在pytorch中，可使用torch.nn包来构建神经网络，其依赖于autograd包来定义模型并对其求导。

**torch.nn包中需要着重记忆的模块（目前阶段）：**

**nn.Module：**神经网络模块。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能。

**nn.Parameter：**张量的一种，当它作为一个属性分配给一个Module时，它会被自动注册为一个参数。



**损失函数：**

nn包中有多种损失函数，nn.MSELoss是比较简单的一种，它计算输出与目标的均方误差(mean squared error)



**更新参数：**

使用torch.optim包，可以很方便地导入常用的优化器(如optim.SGD)进行参数更新



至此，一个神经网络的典型训练过程已结束



## 训练分类器

**数据读取：**

为了避免重复造轮子，可以使用python标准库读取各种数据：

图片：Pillow\OpenCV等

音频：scipy\librosa等

文本：NLTK\SpaCy等

特别的，对于视觉方面可以使用torchvision包，其中包含了针对Imagenet\CIFAR10\MNIST等常用数据集的加载器(data loaders)。另外还有对图像数据转换的操作，<u>torchvision.datasets</u>和<u>torch.utils.data.DataLoader</u>













