import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from final_assignment_train import model_resnet18, load_data, MyData


def choose_dir(model_name):
    model_dir_name = model_name
    root_dir_path = "../model_data/"
    model_dir = root_dir_path + model_dir_name
    model_dir = os.path.normpath(model_dir)
    dir_name = os.listdir(root_dir_path)
    # result = False
    for dir_data in dir_name:
        dir_data = os.path.normpath(root_dir_path+dir_data)
        if os.path.samefile(dir_data, model_dir):
            model_data_list = os.listdir(model_dir)
            # result = True
            return model_data_list, model_name

    print("不包含输入模型名称的训练结果目录！")
    return None


# 对模型训练数据进行排序
def sort_mode_data(filename):
    num_str = filename.split('_')[1]
    num_str_int = int(num_str[5:])
    # print(num_str_int)
    return num_str_int


def model_test(model_data_list, model_name):
    # writer = SummaryWriter(f"../logs_test/{model_name_glo}")
    step = 0.0
    x_point_list = list()
    y_point_list = list()
    resnet18_pretrained = model_resnet18(False)
    resnet18_pretrained.to("cuda")
    resnet18_pretrained.eval()

    # 加载测试数据集
    x_data_img_test, y_data_label_test, label_name_dict_test, label_name_list_test = \
        load_data("../homework_dataset/test/")
    mydata_test = MyData(x_data_img_test, y_data_label_test, label_name_list_test)
    dataloader_test = DataLoader(mydata_test, 16, True, drop_last=True)

    for model_data in model_data_list:
        # resnet18_pretrained = model_resnet18(False)
        resnet18_pretrained.load_state_dict(torch.load(f"../model_data/{model_name}/" + model_data))
        # print(resnet18_pretrained)

        # 测试网络效果，部署到CPU上
        # resnet18_pretrained.to("cpu")
        total_correct_count = 0
        total_count = 0
        print("开始测试一个版本训练模型！")
        with torch.no_grad():
            for data in dataloader_test:
                img, label = data
                img = img.to("cuda")
                label = label.to("cuda")
                outputs = resnet18_pretrained(img)
                out = outputs.argmax(dim=1)
                correct_count = (out == label).sum().item()
                total_correct_count += correct_count
                total_count += len(label)
        acc = float(format(total_correct_count/total_count, ".4f"))
        print(f"准确率：{total_correct_count/total_count:.4f}")
        y_point_list.append(acc)
        x_point_list.append(step)
        step += 1.0
        # writer.add_scalar(f"{model_name_glo}_test_acc", acc, step)

    # writer.close()
    plt.plot(x_point_list, y_point_list, marker='o')
    plt.title("Changes in accuracy of different versions")
    plt.xlabel("train_model_version")
    plt.ylabel("accuracy")
    plt.show()
    # plt.savefig("../image/resnet18_test_acc.jpg", bbox_inches="tight", pad_inches=0)
    plt.savefig("../image/resnet18_pre_test_acc.jpg", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    # 加载训练后的模型和参数
    # 分别测试有无迁移学习训练模型的效果，需要注释其一
    # model_data_list_glo, model_name_glo = choose_dir("resnet18")
    model_data_list_glo, model_name_glo = choose_dir("resnet18_pretrained")

    model_data_list_glo.sort(key=sort_mode_data)
    model_test(model_data_list_glo, model_name_glo)
