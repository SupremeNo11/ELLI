import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_image(image, title=None):
    # 如果图像是 BGR 格式，将其转换为 RGB 格式
    if len(image.shape) == 3 and image.shape[2] == 3:  # 判断是否为彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    if title:
        plt.title(title)
    plt.show()


def show_all_images(images, labels=None, save=True):
    # 确定每行和每列显示多少张图片
    n = len(images)
    rows = int(np.sqrt(n))  # 计算行数
    cols = int(np.ceil(n / rows))  # 计算列数
    print('rows:', rows, 'cols:', cols)
    # 获取每张图片的尺寸（假设所有图片尺寸相同）
    img_height, img_width, _ = images[0].shape

    # 设置每个子图的大小
    fig, axes = plt.subplots(rows, cols, figsize=(cols * img_width / 100, rows * img_height / 100), dpi=100)

    # 显示所有图片
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                axes[i, j].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
                axes[i, j].axis('off')  # 隐藏坐标轴
                if labels:
                    axes[i, j].set_title(labels[idx], fontsize=12)  # 设置标签和字体大小
            else:
                axes[i, j].axis('off')  # 对于多余的子图，隐藏坐标轴

    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if save:
        plt.savefig('./output/pic/myplot.png', bbox_inches='tight')
    # 显示画布
    # plt.show()


if __name__ == '__main__':
    pass

