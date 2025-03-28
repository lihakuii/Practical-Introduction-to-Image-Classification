"""
该文件能够通过修改代码来实现修改后的vgg16和resnet模型的训练
"""
import os
import time

import torch
import torchvision.models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import VGG16_Weights, ResNet18_Weights
from torch import nn

device_glo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 数据预处理
def load_data(path="../homework_dataset/train/"):
    print("start load_data")
    x_data_img = list()
    y_data_label = list()
    label_name_list = list()
    label_name_dict = dict()
    data_root_path = path
    # test_data_root_path = "../homework_dataset/test/"
    list_root_dir = os.listdir(data_root_path)
    img_trans_fuc = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
    for label, label_name in enumerate(list_root_dir):
        label_name_dict[label] = label_name
        label_name_list.append(label_name)
        for filename in os.listdir(data_root_path + label_name):
            if not filename.endswith(".jpg"):
                continue
            image_input = Image.open(data_root_path + label_name + '/' + filename)
            image_output = img_trans_fuc(image_input)
            x_data_img.append(image_output)
            y_data_label.append(label)
    print("end load_data")

    return x_data_img, y_data_label, label_name_dict, label_name_list


# x_data_img_train, y_data_label_train, label_name_dict_train, label_name_list_train = load_data()
# 验证数据预处理是否正确
# print(len(x_data_img), len(y_data_label), x_data_img[0].size(), y_data_label[0], label_name_list)


# 利用Dataset类生成训练数据集
class MyData(Dataset):
    def __init__(self, x_data_img, y_data_label, label_name_list):
        self.x_data_img = x_data_img
        self.y_data_label = y_data_label
        self.label_name_list = label_name_list

    def __len__(self):
        return len(self.x_data_img)

    def __getitem__(self, item):
        return self.x_data_img[item], self.y_data_label[item]


# mydata_train = MyData(x_data_img_train, y_data_label_train, label_name_list_train)
# dataloader_train = DataLoader(mydata_train, 16, True, drop_last=True)

# x, y = dataloader[0]
# 验证MyData和dataloader创建是否正确
# x, y = next(iter(dataloader))
# print(len(dataloader), x.size(), y)


# 迁移学习方法构建网络模型vgg16或者ResNet，增加满足需求的分类层，但对于特征学习框架不在重新学习

# vgg16
def model_vgg16(only_train_classifier_seq_bool=True):
    print("开始vgg16模型搭建")
    vgg16_pretrained = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    # print(vgg16_pretrained.classifier)
    # print(vgg16_pretrained.classifier.parameters())
    only_train_classifier_seq = only_train_classifier_seq_bool
    if only_train_classifier_seq:
        for param in vgg16_pretrained.features.parameters():
            param.requires_grad_(False)
        for param in vgg16_pretrained.avgpool.parameters():
            param.requires_grad_(False)
    vgg16_pretrained.classifier.append(nn.ReLU(inplace=True))
    vgg16_pretrained.classifier.append(nn.Dropout(0.5, False))
    vgg16_pretrained.classifier.append(nn.Linear(1000, 5))
    print("vgg16模型搭建完成")
    return vgg16_pretrained
    # print(vgg16_pretrained.classifier[7])

    # 测试哪些层requires_grad为true
    # for i, j in vgg16_pretrained.named_parameters():
    #     if j.requires_grad:
    #         print(i)


# vgg16_pretrained_glo = model_vgg16()


# resnet18
def model_resnet18(pretrained_bool=True):
    print("resnet18模型开始构建")
    resnet_pretrained = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    only_train_classifier_seq = pretrained_bool
    if only_train_classifier_seq:
        for param in resnet_pretrained.parameters():
            param.requires_grad_(False)
    resnet_pretrained.fc = nn.Linear(512, 5)
    print("resnet18模型构建完成")
    return resnet_pretrained


# resnet_pretrained_glo = model_resnet18()


# 开始使用数据集训练vgg16或resnet


# 训练调整后的vgg16模型
def train_vgg16(model_value, dataloader, device=device_glo, is_pretrained=True):
    print("vgg16模型开始训练")

    if is_pretrained:
        writer_save_path = "../logs/logs_train/vgg16_pretrained/"
        model_data_save_path = "../model_data/vgg16_pretrained/"
    else:
        writer_save_path = "../logs/logs_train/vgg16/"
        model_data_save_path = "../model_data/vgg16/"

    writer = SummaryWriter(writer_save_path)
    vgg16_pretrained = model_value
    vgg16_pretrained.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg16_pretrained.parameters()))
    # optimizer = torch.optim.Adam(vgg16_pretrained.parameters())
    vgg16_pretrained.train()

    # 测试训练数据是否正常
    # x, y = next(iter(dataloader))
    # print(x.shape, y.shape)
    # print(y)
    # output = vgg16_pretrained(x.to(device))
    # print(output.shape)
    # print(output.is_cuda)
    # print(output.device.type)
    # acc = (output.to("cpu").argmax(dim=1) == y)
    # print(acc)
    # print(acc.type())
    # print(acc.sum())
    # print(acc.sum().dtype)

    start_time = time.time()
    total_step = 0
    for epoch in range(20):
        for step, (img, label) in enumerate(dataloader):
            outputs = vgg16_pretrained(img.to(device))
            loss = loss_fn(outputs, label.to(device))
            label = label.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step = total_step + 1
            if step % 30 == 0:
                end_time = time.time()
                need_time = end_time - start_time
                start_time = time.time()
                acc = (outputs.argmax(dim=1) == label).sum().item() / len(label)
                print(f"epoch:{epoch}轮, 每轮次数：{step+1:<5}, 损失:{loss.item():<20}, 准确率: {acc}")
                print(f"train/100steps needs {need_time}s")
                print(total_step, "总数")
                writer.add_scalar("vgg16_train_loss", loss.item(), total_step)
                writer.add_scalar("vgg16_train_acc", acc, total_step)
        torch.save(vgg16_pretrained.state_dict(), model_data_save_path+f"vgg16_epoch{epoch}_data_gpu.pth")
    writer.close()


# 训练调整后的resnet18模型
def train_resnet18(model_value, dataloader, device=device_glo, is_pretrained=True):
    print("resnet18模型开始训练")

    if is_pretrained:
        writer_save_path = "../logs/logs_train/resnet18_pretrained/"
        model_data_save_path = "../model_data/resnet18_pretrained/"
    else:
        writer_save_path = "../logs/logs_train/resnet18/"
        model_data_save_path = "../model_data/resnet18/"

    writer = SummaryWriter(writer_save_path)
    resnet18_pretrained = model_value
    resnet18_pretrained.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet18_pretrained.parameters()))
    resnet18_pretrained.train()

    # 测试训练数据是否正常
    # x, y = next(iter(dataloader))
    # print(x.shape, y.shape)
    # print(y)
    # output = resnet18_pretrained(x.to(device))
    # print(output.shape)
    # print(output.is_cuda)
    # print(output.device.type)
    # acc = (output.to("cpu").argmax(dim=1) == y)
    # print(acc)
    # print(acc.type())
    # print(acc.sum())
    # print(acc.sum().dtype)

    start_time = time.time()
    total_step = 0
    for epoch in range(20):
        for step, (img, label) in enumerate(dataloader):
            outputs = resnet18_pretrained(img.to(device))
            loss = loss_fn(outputs, label.to(device))
            label = label.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step = total_step + 1
            # 按照每一百轮频率输出loss和accuracy的可视化变化轨迹
            if step % 30 == 0:
                end_time = time.time()
                need_time = end_time - start_time
                start_time = time.time()
                acc = (outputs.argmax(dim=1) == label).sum().item() / len(label)
                print(f"epoch:{epoch}轮, 每轮次数：{step+1:<5}, 损失:{loss.item():<20}, 准确率: {acc}")
                print(f"train/100steps needs {need_time}s")
                writer.add_scalar("resnet18_train_loss", loss.item(), total_step)
                writer.add_scalar("resnet18_train_acc", acc, total_step)
        torch.save(resnet18_pretrained.state_dict(), model_data_save_path+f"resnet18_epoch{epoch}_data_gpu.pth")
    writer.close()

# train(vgg16_pretrained_glo)


if __name__ == "__main__":
    x_data_img_train, y_data_label_train, label_name_dict_train, label_name_list_train = load_data()
    mydata_train = MyData(x_data_img_train, y_data_label_train, label_name_list_train)
    dataloader_train = DataLoader(mydata_train, 16, True, drop_last=True)

# 下面每两个为一组运行，其余需要进行注释，分别训练不同情况下的模型

    # 训练 非迁移学习下的VGG16模型
    # vgg16_pretrained_glo = model_vgg16(only_train_classifier_seq_bool=False)
    # train_vgg16(vgg16_pretrained_glo, dataloader_train, is_pretrained=False)

    # 训练 非迁移学习下的ResNet18模型
    # resnet_pretrained_glo = model_resnet18(pretrained_bool=False)
    # train_resnet18(resnet_pretrained_glo, dataloader_train, is_pretrained=False)

    # 训练 迁移学习下的ResNet18模型
    resnet_pretrained_glo = model_resnet18(pretrained_bool=True)
    train_resnet18(resnet_pretrained_glo, dataloader_train)

    # 训练 迁移学习下的VGG16模型
    # vgg16_pretrained_glo = model_vgg16(only_train_classifier_seq_bool=True)
    # train_vgg16(vgg16_pretrained_glo, dataloader_train)
