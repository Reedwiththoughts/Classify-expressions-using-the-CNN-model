import pandas as pd
import numpy as np
from keras.models import load_model

# 加载模型
model = load_model('FER_Model.h5')

# 读取test.csv文件
test_df = pd.read_csv('./dataset/test.csv')

# 将pixels列中的每个像素序列字符串转换为整数列表
test_df['pixels'] = test_df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])

# 将'pixels'列转换为numpy数组，并将其形状改变为(width, height, 1)，然后将所有像素值除以255进行归一化
width, height = 48, 48
test_X = np.array(test_df['pixels'].tolist(), dtype='float32').reshape(-1, width, height, 1) / 255.0

# 使用模型进行预测
test_Y = model.predict(test_X)

# 将预测结果转换为0-6数字
test_Y = np.argmax(test_Y, axis=1)

# 将数字转换为对应的情感标签
emotion_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}
test_Y = [emotion_map[i] for i in test_Y]

# 将预测结果保存为csv文件
pd.DataFrame(test_Y).to_csv('test_result.csv', index=False, header=False)