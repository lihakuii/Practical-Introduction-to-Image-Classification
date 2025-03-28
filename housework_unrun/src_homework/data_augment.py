"""
    这是图片数据增强的代码，可以对图片实现：
    # 1. 尺寸放大缩小(不进行处理效果不大）
    2. 旋转（任意角度，如45°，90°，180°，270°）
    3. 翻转（水平翻转，垂直翻转）
    4. 明亮度改变（变亮，变暗）
    5. 像素平移（往一个方向平移像素，空出部分自动填补黑色）
    6. 添加噪声（椒盐噪声，高斯噪声）

    本文件只需要修改程序末尾的root_path_g的值即可运行自动进行目录下的图像增强处理
"""
import os
import cv2
import numpy as np

"""
    缩放
"""


# 放大缩小
def scale_f(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


'''
翻转
'''


# 水平翻转
def horizontal(image):
    return cv2.flip(image, 1, dst=None)  # 水平镜像


# 垂直翻转
def vertical(image):
    return cv2.flip(image, 0, dst=None)  # 垂直镜像


# 旋转，R可控制图片放大缩小
def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    # rotate matrix
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, m, (w, h))
    return image


'''  
明亮度 
'''


# 变暗
def darker(image, percentage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percentage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percentage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percentage)
    return image_copy


# 明亮
def brighter(image, percentage=1.1):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percentage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percentage), a_max=255, a_min=0)
    return image_copy


# 平移
def move(img, x, y):
    img_info = img.shape
    height = img_info[0]
    width = img_info[1]

    mat_translation = np.float32([[1, 0, x], [0, 1, y]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列
    # [[1,0,20],[0,1,50]]   表示平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离。
    dst = cv2.warpAffine(img, mat_translation, (width, height))  # 变换函数
    return dst


'''
增加噪声
'''


# 椒盐噪声
def salt_and_pepper(src, percentage=0.05):
    sp_noise_img = src.copy()
    sp_noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(sp_noise_num):
        rand_r = np.random.randint(0, src.shape[0] - 1)
        rand_g = np.random.randint(0, src.shape[1] - 1)
        rand_b = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            sp_noise_img[rand_r, rand_g, rand_b] = 0
        else:
            sp_noise_img[rand_r, rand_g, rand_b] = 255
    return sp_noise_img


# 高斯噪声
def gaussian_noise(image, percentage=0.05):
    g_noise_img = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    g_noise_num = int(percentage * image.shape[0] * image.shape[1])
    for i in range(g_noise_num):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        g_noise_img[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return g_noise_img


def blur_f(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    return blur

# 测试使用不需要管
def test_one_pic():
    test_jpg_loc = r"data/daisy/1.jpg"
    test_jpg = cv2.imread(test_jpg_loc)
    cv2.imshow("Show Img", test_jpg)
    # cv2.waitKey(0)
    img1 = blur_f(test_jpg)
    cv2.imshow("Img 1", img1)
    # cv2.waitKey(0)
    # img2 = gaussian_noise(test_jpg,0.01)
    # cv2.imshow("Img 2", img2)
    cv2.waitKey(0)

# 测试使用不需要管
def test_one_dir():
    root_path = "data/daisy"
    save_path = root_path
    for a, b, c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(file_i_path)
            img_i = cv2.imread(file_i_path)

            # img_scale = scale(img_i,1.5)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale.jpg"), img_scale)

            # img_horizontal = horizontal(img_i)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_horizontal.jpg"), img_horizontal)
            #
            # img_vertical = vertical(img_i)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_vertical.jpg"), img_vertical)
            #
            # img_rotate = rotate(img_i,90)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate90.jpg"), img_rotate)
            #
            # img_rotate = rotate(img_i, 180)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate180.jpg"), img_rotate)
            #
            # img_rotate = rotate(img_i, 270)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate270.jpg"), img_rotate)
            #
            # img_move = move(img_i,15,15)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_move.jpg"), img_move)
            #
            img_darker = darker(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_darker.jpg"), img_darker)
            #
            # img_brighter = brighter(img_i)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_brighter.jpg"), img_brighter)
            #
            # img_blur = blur(img_i)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_blur.jpg"), img_blur)
            #
            # img_salt = salt_and_pepper(img_i,0.05)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_salt.jpg"), img_salt)


def all_data(rootpath):
    root_path = rootpath
    # save_loc = root_path
    for a, b, c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(file_i_path)
            split = os.path.split(file_i_path)
            save_path = split[0]
            print(save_path)
            # print(split[0])
            # dir_loc = os.path.split(split[0])[1]
            # print(dir_loc)
            # save_path = os.path.join(save_loc, dir_loc)

            img_i = cv2.imread(file_i_path)
            # img_scale = scale(img_i, 1.5)
            # cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_scale.jpg"), img_scale)
            print(save_path)

            img_horizontal = horizontal(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_horizontal.jpg"), img_horizontal)

            img_vertical = vertical(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_vertical.jpg"), img_vertical)

            img_rotate = rotate(img_i, 90)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate90.jpg"), img_rotate)

            img_rotate = rotate(img_i, 180)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate180.jpg"), img_rotate)

            img_rotate = rotate(img_i, 270)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_rotate270.jpg"), img_rotate)

            img_move = move(img_i, 15, 15)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_move.jpg"), img_move)

            img_darker = darker(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_darker.jpg"), img_darker)

            img_brighter = brighter(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_brighter.jpg"), img_brighter)

            img_blur = blur_f(img_i)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_blur.jpg"), img_blur)

            img_salt = salt_and_pepper(img_i, 0.05)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_salt.jpg"), img_salt)

            img_gauss_noise = gaussian_noise(img_i, 0.05)
            cv2.imwrite(os.path.join(save_path, file_i[:-4] + "_gauss_noise.jpg"), img_gauss_noise)


if __name__ == "__main__":
    # test_one_dir()
    # test_one_pic()
    # 需要修改root_path_g为数据集的根目录，注意路径尾部需要加/
    root_path_g = "../homework_dataset/"
    all_data(root_path_g)
