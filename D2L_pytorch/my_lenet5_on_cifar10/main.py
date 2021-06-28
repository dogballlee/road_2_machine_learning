import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from lenet5 import LeNet5
from tqdm import tqdm
from torch.utils import tensorboard as tb
from torchvision import models
# import tensorboard

log_dir_train='log_train'
train_writer = tb.SummaryWriter(log_dir=log_dir_train, comment='LeNet5_train')


def main():
    batch_siz = 32
    learning_rate = 1e-4
    weight_decay = 1e-3
    n_epochs = 20

    cifar_train = datasets.CIFAR10('cifar', True, transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size = batch_siz, shuffle = True)
    cifar_test = datasets.CIFAR10('cifar', False, transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_siz, shuffle=True)

    x, labels = iter(cifar_train).next()
    print('x:', x.shape, 'label:', labels.shape)

    device = torch.device('cuda')
    # model = LeNet5.to(device)
    model = models.resnet50(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    print(model)

    model_path = './pre_res_model.ckpt'
    best_acc = 0.0

    input_shape = (32, 32)
    #喂1个假样本(3通道，高宽和input_shape相同)给summary_writer，仅用于展示网络结构
    graph_inputs = torch.from_numpy(np.random.rand(1, 3, input_shape[0], input_shape[1])).type(torch.FloatTensor).cuda()
    train_writer.add_graph(model, (graph_inputs))
    # torch.save(model,'my_LeNet5.pt')
    torch.save(model,'resnet18.pt')


    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_accs = []
        for batch in tqdm(cifar_train):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_writer.add_scalar('train_loss', train_loss, epoch)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(cifar_test):
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(x)
            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        print(f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))

if __name__ == '__main__':
    main()