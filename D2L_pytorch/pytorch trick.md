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



## tqdm是个骚道具

`import tqdm`

导入后tqdm可用于所有可迭代对象，故而在pytorch的dataloader的使用场景下，可以这样使用：

```python
for data, target in tqdm(train_loader):`

	`......
```

（省略号中为循环内的各种操作，可以正常对batch内的data进行后续处理，并显示进度条）

原理：这种用法相当于在dataloader上对每个batch和batch总数做的进度条



torch训练中的需注意的点

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

