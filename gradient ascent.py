import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 加载模型
model = load_model('FER_Model.h5')

# 选择要可视化的层和滤波器
layer_name = 'activation_24'
filter_index = 0  # 可以在0到`layer.output.shape[-1] - 1`之间选择任何值

# 创建一个新的模型，该模型的输出是所选层的输出
layer_output_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 初始化一个图像张量，并将其设置为可训练
input_img_data = tf.Variable(tf.random.uniform((1, 48, 48, 1)))

# 使用 GradientTape 来计算梯度
with tf.GradientTape() as tape:
    tape.watch(input_img_data)
    layer_output = layer_output_model(input_img_data)
    loss_value = tf.keras.backend.mean(layer_output[:, :, :, filter_index])

grads = tape.gradient(loss_value, input_img_data)

# 归一化梯度
grads /= (tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(grads))) + 1e-5)

# 运行梯度上升20步
step = 1.  # 这是梯度上升的步长
for i in range(20):
    with tf.GradientTape() as tape:
        tape.watch(input_img_data)
        layer_output = layer_output_model(input_img_data)
        loss_value = tf.keras.backend.mean(layer_output[:, :, :, filter_index])
    grads = tape.gradient(loss_value, input_img_data)
    input_img_data.assign_add(grads * step)

# 将张量转换为有效图像
img = input_img_data[0].numpy()
img -= img.mean()
img /= (img.std() + 1e-5)
img *= 0.1
img += 0.5
img = np.clip(img, 0, 1)
img *= 255
img = np.clip(img, 0, 255).astype('uint8')

# 显示图像
plt.imshow(img.reshape(48, 48), cmap='gray')
plt.show()