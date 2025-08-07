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
print(os.listdir("./dataset"))

datatrain0 = pd.read_csv('./dataset/train.csv')
datatest0 = pd.read_csv('./dataset/test.csv')
# 查看数据集形状
datatrain0.shape

#预览前10行数据
datatrain0.head(10)

#查看数据集的分类情况
# 无这一项，不需要
# data.Usage.value_counts()

#查看表情分类数据
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = datatrain0['emotion'].value_counts().sort_index().reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
print(emotion_counts)

# 绘制类别分布条形图
# matplotlib inline
# config InlineBackend.figure_format = 'svg'
plt.figure(figsize=(6, 4))
sns.barplot(x=emotion_counts.emotion, y=emotion_counts.number)
plt.title('Class distribution')
plt.ylabel('Number', fontsize=12)
plt.xlabel('Emotions', fontsize=12)
plt.show()


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


# 绘制train, val, test的条形图
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # 定义一个列表，包含所有的表情标签

# 定义一个函数，用于设置条形图的属性。函数接受三个参数：`axe`（用于绘制条形图的坐标轴对象），`df`（包含数据的DataFrame对象）和`title`（条形图的标题）
def setup_axe(axe, df, title):
    df['emotion'].value_counts().sort_index().plot(ax=axe, kind='bar', rot=0,
                                                   color=['r', 'g', 'b', 'r', 'g', 'b', 'r'])
    # 计算每种表情的数量，然后按照表情的标签排序，最后在指定的坐标轴上绘制条形图。
    axe.set_xticklabels(emotion_labels) # 设置x轴的刻度标签为表情标签
    axe.set_xlabel("Emotions")
    axe.set_ylabel("Number")
    axe.set_title(title)
    # 分别设置x轴的标签、y轴的标签和条形图的标题

    # 使用上述列表设置单个条形标签
    for i in axe.patches: # 遍历每个条形
        # get_x pulls left or right; get_height pushes up or down
        axe.text(i.get_x() - .05, i.get_height() + 120,
                 str(round((i.get_height()), 2)), fontsize=14, color='dimgrey',rotation=0)
        # 在每个条形的顶部添加文本，显示该条形的高度（即表情的数量）

# matplotlib inline
# config InlineBackend.figure_format = 'svg'
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey='all')
# 创建一个新的图形窗口和两个子图的坐标轴
setup_axe(axes[0], data_train, 'Train')
setup_axe(axes[1], data_val, 'Validation')
# setup_axe(axes[2], data_test, 'Test') 无
# 分别为训练集、验证集和测试集设置条形图的属性
plt.show()

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


model = Sequential() # 创建一个Sequential模型，是多个网络层的线性堆叠

# ---------- Convolutional Stages 1 ----------
# ***** Conv Block a *****
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(width, height, 1),
                 data_format='channels_last', padding='same'))
# 向模型中添加卷积层，Conv2D是二维卷积层，这层会创建一组卷积核，用于在输入图像上滑动以捕获图像中的特征。
model.add(BatchNormalization())
# 向模型中添加批量标准化层，批量标准化层可以使神经网络的训练更快，同时还能提高模型的性能。
model.add(Activation('relu'))
# 向模型中添加激活层，并设置激活函数为ReLU，ReLU激活函数可以增加神经网络的非线性。
# ***** Conv Block b *****
model.add(Conv2D(64, kernel_size=(3, 3), padding='same')) # 卷积层
model.add(BatchNormalization()) # 批量标准化层
model.add(Activation('relu')) # 激活层
# max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# 向模型中添加最大池化层，最大池化层可以降低特征图的维度，同时保留最重要的特征。

# ---------- Convolutional Stages 2 ----------
# ***** Conv Block a *****
model.add(Conv2D(128, kernel_size=(3, 3), padding='same')) # 卷积层
model.add(BatchNormalization()) # 批量标准化层
model.add(Activation('relu')) # 激活层
# ***** Conv Block b *****
model.add(Conv2D(128, kernel_size=(3, 3), padding='same')) # 卷积层
model.add(BatchNormalization()) # 批量标准化层
model.add(Activation('relu')) # 激活层
# max pooling
model.add(MaxPooling2D(pool_size=(2, 2))) # 最大池化层

# ---------- Convolutional Stages 3 ----------
# ***** Conv Block a *****
model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# ***** Conv Block b *****
model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# ---------- Convolutional Stages 4 ----------
# ***** Conv Block a *****
model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# ***** Conv Block b *****
model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())
# 向模型中添加展平层，展平层会将多维输入一维化，常用在从卷积层到全连接层的过渡。

# Full connection
model.add(Dense(4096, activation='relu', kernel_regularizer=l2()))
# 向模型中添加全连接层，全连接层会学习输入特征的全局模式。
model.add(Dropout(rate_drop))
# 向模型中添加Dropout层，Dropout层可以在训练过程中随机关闭一部分神经元，以防止过拟合。
model.add(Dense(4096, activation='relu', kernel_regularizer=l2()))
model.add(Dropout(rate_drop))

#output layer
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2()))

model.save('FER_Model_untrained.h5')

# model.compile(loss=['categorical_crossentropy'],
#              optimizer=SGD(momentum=0.9, nesterov=True ,decay=1e-4),
#              metrics=['accuracy'])
# 配置模型的学习过程。这里设置了损失函数为交叉熵损失，优化器为带动量的随机梯度下降，评价指标为准确率。
# 因为在新版本的Keras优化器中，`decay`参数已经被弃用，可以使用`tf.keras.optimizers.schedules.ExponentialDecay`来代替，代码如下

# 创建学习率衰减计划
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

# 创建优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary() # 打印出模型的结构，包括每一层的名称、类型、输出形状和参数数量。

# 数据增强
data_generator = ImageDataGenerator(
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    featurewise_std_normalization=False,
    horizontal_flip=True)
# 创建一个图像数据生成器，用于进行数据增强。数据增强可以通过应用随机的图像变换（如缩放、平移、旋转和翻转）来增加模型的泛化能力。

es = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
# 创建一个早停回调。早停可以在验证损失不再下降时停止训练，以防止模型过拟合。

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=5, verbose=1)
# 创建一个学习率衰减回调。学习率衰减可以在验证准确率不再提高时降低学习率，以提高模型的性能。

history = model.fit(data_generator.flow(train_X, train_Y, batch_size),
                    # steps_per_epoch=len(train_X) / batch_size,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=[es, reduce_lr],
                    validation_data=(val_X, val_Y))
# 使用数据生成器和回调来训练模型。训练过程中的历史数据（如每个周期的训练损失和验证损失）会被保存在`history`对象中。

# matplotlib inline
# config InlineBackend.figure_format = 'svg'
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# 绘制训练和验证精度曲线
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

# 绘制训练和验证损失曲线
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save('FER_Model.h5')