import matplotlib.pyplot as plt
from PIL import Image



def save_cropped_image(center_x, center_y, image):

    left = center_x - 128
    top = center_y - 128
    right = center_x + 128
    bottom = center_y + 128


    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image.width)
    bottom = min(bottom, image.height)


    cropped_image = image.crop((left, top, right, bottom))
    save_path= "/home/wubin/code/SamPose/data/Tless/color"

    cropped_image.save(f"{save_path}/output_image.png", 'png')
    print(f"Cropped image saved at: (left={left}, top={top}, right={right}, bottom={bottom})")


# 处理鼠标点击事件
def on_click(event, image):
    if event.button == 1:  # 只处理左键点击

        center_x, center_y = int(event.xdata), int(event.ydata)
        print(f"Clicked at: ({center_x}, {center_y})")


        save_cropped_image(center_x, center_y, image)
        plt.close()


# 打开图像
image_path = '/home/wubin/code/SamPose/data/Tless/color/000205.png'  # 输入图像路径
image = Image.open(image_path)

# 转换为numpy数组，以便matplotlib显示
image_np = image.convert('RGB')

# 创建图像窗口
fig, ax = plt.subplots()
ax.imshow(image_np)

# 绑定鼠标点击事件
fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, image))

# 显示图像并启用交互
plt.title('Click to select the center for a 256x256 crop')
plt.show()
