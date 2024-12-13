import torch
import clip
from PIL import Image
# An attempt to combine text and images
def generate_image_from_text(text):
    # 加载CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 准备输入文本
    text_input = clip.tokenize([text]).to(device)

    # 使用文本查询图像
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        # 使用文本特征和图像特征进行相似性匹配，选择最相似的图像
        similarity = (image_features @ text_features.T).squeeze(0)
        best_image_index = similarity.argmax().item()

        # 获取最相似的图像
        image_path = "your_image_dataset/image_" + str(best_image_index) + ".jpg"
        generated_image = Image.open(image_path)
        generated_image.show()

# 输入文本描述信息
text_description = "a cup of coffee on a wooden table"

# 生成图像
generate_image_from_text(text_description)
