import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

width, height = 48, 48
num_classes = 7
batch_size = 64
num_epochs = 100
#图像的宽度和高度，类别的数量，批处理的大小和训练的周期数。

def CRNO(df, dataName): # 定义一个函数，接受两个参数df（包含数据的DataFrame对象）和`dataName`（数据的名称）
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    # 将DataFrame中的'pixels'列中的每个像素序列字符串转换为整数列表
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, width, height, 1) / 255.0
    # 将'pixels'列转换为numpy数组，并将其形状改变为(width, height, 1)，然后将所有像素值除以255进行归一化
    data_Y = to_categorical(df['emotion'], num_classes)
    # 将'emotion'列转换为类别向量
    print(dataName, f"_X shape: {data_X.shape}, ", dataName, f"_Y shape: {data_Y.shape}")
    # 打印出数据的形状
    return data_X, data_Y # 返回处理后的数据

def generate_saliency_map(model, image):
    #该函数接受一个模型和一个图像，然后生成该图像的显著性图。
    # Reshape the image (because the model use 4-dimensional tensor (batch_size, channel, width, height))
    image = image.reshape(1, 48, 48, 1)
    # Set the requires_grad_ to the image for retrieving gradients
    image = tf.Variable(tf.cast(image, tf.float32), trainable=True)
    with tf.GradientTape() as tape:
        # Forward pass
        output = model(image)
        # Catch the output
        output_idx = tf.argmax(output[0])
        output_max = output[0, output_idx]
    # Do backpropagation to get the derivative of the output based on the image
    grads = tape.gradient(output_max, image)
    # Get the absolute values of the gradients
    grads = tf.abs(grads)
    # Get the maximum value of the gradients
    max_grads = np.max(grads)
    # Normalize the gradients
    grads = grads / max_grads
    # Convert the gradients to a numpy array
    grads = grads.numpy()
    # Reshape the gradients to the original image shape
    grads = grads.reshape(48, 48)
    # Plot the saliency map
    plt.imshow(grads, cmap='jet')
    plt.show()

datatrain0 = pd.read_csv('./dataset/train.csv')
datatest0 = pd.read_csv('./dataset/test.csv')

data_train, data_val = train_test_split(datatrain0, test_size=0.2, random_state=42, stratify=datatrain0['emotion'])
# Load the training and validation data

# Process the training and validation data
train_X, train_Y = CRNO(data_train, "train")  #training data
val_X, val_Y = CRNO(data_val, "val")  #validation data

# Load the trained model
model = tf.keras.models.load_model('FER_Model.h5')

# Generate the saliency map
generate_saliency_map(model, train_X[0])#emotion=0的第一个
generate_saliency_map(model, train_X[299])#emotion=1的第一个
generate_saliency_map(model, train_X[2])#emotion=2的第一个
generate_saliency_map(model, train_X[7])#emotion=3的第一个
generate_saliency_map(model, train_X[3])#emotion=4的第一个
generate_saliency_map(model, train_X[15])#emotion=5的第一个
generate_saliency_map(model, train_X[4])#emotion=6的第一个


# 图片样本示例（好像没必要）
def row2image_label(row): # 将一行数据转换为图像和标签
    pixels, emotion = row['pixels'], emotion_map[row['emotion']] # 从行数据中提取像素值和表情标签
    row2img = np.array(pixels.split()) # 将像素值字符串分割为数组
    row2img = row2img.reshape(48, 48) # 将像素值数组重塑为48x48的二维数组
    image = np.zeros((48, 48, 3)) # 创建一个48x48的三通道图像，所有像素值初始化为0
    image[:, :, 0] = row2img
    image[:, :, 1] = row2img
    image[:, :, 2] = row2img # 将像素值数组复制到图像的每个通道
    return image.astype(np.uint8), emotion # 返回图像和表情标签

# matplotlib inline
# config InlineBackend.figure_format = 'svg'
plt.figure(0, figsize=(16, 10)) # 创建图形窗口，大小为16x10
for i in range(1, 8): # 对每种表情执行以下操作
    face = datatrain0[datatrain0['emotion'] == i - 1].iloc[0] # 选择该表情的第一张图像
    img, label = row2image_label(face) # 将该行数据转换为图像和标签
    plt.subplot(2, 4, i) # 在2x4的子图网格中创建一个新的子图
    plt.imshow(img) # 显示图像
    plt.title(label) # 设置子图的标题为表情标签

plt.show() # 显示图形窗口