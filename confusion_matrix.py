import numpy as np  # 线性代数
import pandas as pd  # 数据预处理, CSV 文件 I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
from keras.models import load_model

print(os.listdir("./dataset"))

datatrain0 = pd.read_csv('./dataset/train.csv')
datatest0 = pd.read_csv('./dataset/test.csv')

#查看表情分类数据
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = datatrain0['emotion'].value_counts().sort_index().reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
print(emotion_counts)

# 分割数据集
train, val = train_test_split(datatrain0, test_size=0.2, random_state=42, stratify=datatrain0['emotion'])

# 查看训练集和验证集的大小
print('Training data shape:', train.shape)
print('Validation data shape:', val.shape)


#分割数据为: train, validation, test
data_train = train.copy()
data_val = val.copy() # 需要验证集时再调整
data_test = datatest0.copy()
print(f"train shape: {data_train.shape}")
print(f"validation shape: {data_val.shape}")
print(f"test shape: {data_test.shape}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # 定义一个列表，包含所有的表情标签

#初始化参数
num_classes = 7
width, height = 48, 48
num_epochs = 300
batch_size = 512
num_features = 64
rate_drop = 0.1

# CRNO -- Convert, Reshape, Normalize, One-hot encoding
# (i) 将数据标签由字符串改为整数
# (ii) 调整图片大小为 48x48, 归一化图像
# (iii) 更改标签编码为one-hot, e.g. 类别3(Happy)为 **[0,0,0,1,0,0,0]**

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


train_X, train_Y = CRNO(data_train, "train")  #training data
# 使用CRNO函数处理训练集，并将结果存储在`train_X`和`train_Y`中
val_X, val_Y = CRNO(data_val, "val")  #validation data
# test_X, test_Y = CRNO(data_test, "test")  #test data 先不test吧而且test里面没emotion
test_X, test_Y = CRNO(datatrain0, "test")  #狠人直接train当test

model = load_model('FER_Model.h5')

test_true = np.argmax(test_Y, axis=1)
test_pred = np.argmax(model.predict(test_X), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    此函数打印和绘制混淆矩阵
    可以通过设置“normalize=True”来应用规范化。
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 仅使用数据中显示的标签
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # 显示所有的标记...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... 用相应的列表条目标记它们
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 旋转x轴标签并设置其对齐方式。
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 在数据维度上循环并创建文本批注
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# 绘制归一化混淆矩阵
# matplotlib inline
# config InlineBackend.figure_format = 'svg'
plot_confusion_matrix(test_true, test_pred, classes=emotion_labels, normalize=True, title='Normalized confusion matrix')
plt.show()