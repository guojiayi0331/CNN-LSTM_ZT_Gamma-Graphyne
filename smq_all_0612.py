# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 03:27:52 2024

@author: Gjy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

with open(r'D:\gjy_by\gjy_by\smq\graphyneQ_15_ZT_16512_change.csv') as f:
    df_Q_ch = pd.read_csv(f)

with open(r'D:\gjy_by\gjy_by\smq\graphyneQ_15_ZT_16512.csv') as f:
    df_Q = pd.read_csv(f)

with open(r'D:\gjy_by\gjy_by\smq\graphyne_15_ZT_16512_change.csv') as f:
    df_ch = pd.read_csv(f)

#with open(r'C:\Users\Administrator\Desktop\smq\smq_01\graphyne_15_01_ZT_16512.csv') as f:
with open(r'D:\gjy_by\gjy_by\smq\graphyne_15_ZT_16512.csv') as f:
    df = pd.read_csv(f)


test_ch = df_Q_ch['ZT']

df_all = df_Q.loc[:, 'T=300k':]
#df_all[['num_1_1', 'num_1_2', 'num_1_3', 'num_1_4m', 'num_0_1', 'num_0_2', 'num_0_3', 'num_0_4m', 'num_1_all']] = df_ch[['num_1_1', 'num_1_2', 'num_1_3', 'num_1_4m', 'num_0_1', 'num_0_2', 'num_0_3', 'num_0_4m', 'num_1_all']]

#y = test_ch.values.tolist()


########################获取人工特征
df_all_tz = df.loc[:, 'parameter1':]
df_all_tz[['T=300k']] = df[['T=300k']]

#num_1_1	num_1_2	num_1_3	num_1_4m	num_0_1	num_0_2	num_0_3	num_0_4m	num_1_all
#df_all[['num_1_1', 'num_1_2', 'num_1_3', 'num_1_4m', 'num_0_1', 'num_0_2', 'num_0_3', 'num_0_4m', 'num_1_all']] = df_ch[['num_1_1', 'num_1_2', 'num_1_3', 'num_1_4m', 'num_0_1', 'num_0_2', 'num_0_3', 'num_0_4m', 'num_1_all']]

data_old = df_all_tz.values.tolist()
#data_old = data_old[5000:5500]
data_tz = []
for row in data_old:
    binary_code = [int(bit) for bit in row[:-1]]
    zt_value = row[-1]
    data_tz.append((binary_code, zt_value))

#print(data_tz)

# 特征提取函数
def extract_pattern_features(binary_code, pattern_lengths=[1, 2, 3, 4, 5, 6]):
    features = {f'consecutive_1s_{length}': 0 for length in pattern_lengths}
    features.update({f'consecutive_0s_{length}': 0 for length in pattern_lengths})

    length = len(binary_code)
    def find_consecutive_sequences(binary_code):
        sequences = []
        i = 0
        while i < length:
            current_value = binary_code[i]
            start = i
            while i < length and binary_code[i] == current_value:
                i += 1
            end = i
            sequences.append((current_value, end - start))
        return sequences

    sequences = find_consecutive_sequences(binary_code)
    for value, seq_length in sequences:
        for pattern_length in pattern_lengths:
            if seq_length == pattern_length:
                if value == 1:
                    features[f'consecutive_1s_{pattern_length}'] += 1
                else:
                    features[f'consecutive_0s_{pattern_length}'] += 1
            if pattern_length == pattern_lengths[-1] and seq_length > pattern_length:
                if value == 1:
                    features[f'consecutive_1s_{pattern_length}'] += 1
                else:
                    features[f'consecutive_0s_{pattern_length}'] += 1

    # 复杂组合模式
    complex_patterns = {
        'pattern_10': [1, 0],
        'pattern_01': [0, 1],
        'pattern_101': [1, 0, 1],
        'pattern_010': [0, 1, 0],
        'pattern_1010': [1, 0, 1, 0],
        'pattern_0110': [0, 1, 1, 0],
        'pattern_1100': [1, 1, 0, 0],
        'pattern_0011': [0, 0, 1, 1]
    }
    
    for pattern_name, pattern in complex_patterns.items():
        pattern_length = len(pattern)
        count_pattern = 0
        i = 0
    
        while i <= length - pattern_length:
            # 检查当前窗口是否与模式匹配
            if binary_code[i:i + pattern_length] == pattern:
                count_pattern += 1
                # 跳过已匹配的模式，移动到模式末尾的下一个字符
                i += pattern_length
            else:
                # 如果不匹配，移动一个字符
                i += 1
    
        features[pattern_name] = count_pattern

    return features



# 提取所有样本的特征
pattern_lengths = [1, 2, 3, 4, 5, 6]
X = [extract_pattern_features(sample[0], pattern_lengths) for sample in data_tz]
y = [sample[1] for sample in data_tz]

# 转换为DataFrame
df_tz = pd.DataFrame(X)


# 使用pd.concat在列方向上合并数据框
df_all_new = pd.concat([df_all, df_tz], axis=1)

# 检查合并后的数据框
print(df_all_new)

#############################################

data = df_all_new.loc[:, 'T=300k':].values.tolist()

def get_data():
    train_set = random.sample(data, 500)
    for i in train_set:
        data.remove(i)
    test_set = data
    # print(len(train_set), len(test_set))
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    # print(train_set.shape, test_set.shape)

    x_train = train_set[:, 1:, np.newaxis]
    y_train = train_set[:, 0]
    x_test = test_set[:, 1:, np.newaxis]
    y_test = test_set[:, 0]

    print(x_train.shape, x_test.shape)
    # print(y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_data()



def Model_Single():
    ins = keras.layers.Input(shape=[15, 1])
    #a1 = keras.layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(ins)
    c = keras.layers.Conv1D(64, 4, padding='same',activation='relu')(ins)
    pool = keras.layers.AveragePooling1D(pool_size=1)(c)
    drop1 = keras.layers.Dropout(0.2)(pool)


    lstm = keras.layers.LSTM(50, return_sequences=True)(drop1)
    drop = keras.layers.Dropout(0.2)(lstm)

    z = keras.layers.Flatten()(drop)
    d = keras.layers.Dense(15, activation='tanh')(z)
    out = keras.layers.Dense(1,name='out')(d)
    
    return ins,out


class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, *args, **kwargs):
        output_tensor, attention_scores = self.attention(inputs, inputs, return_attention_scores=True)
        return output_tensor, tf.expand_dims(attention_scores, axis=-1)  # 扩展注意力分数的维度

    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], self.num_heads, input_shape[1], 1)


def Model_Multi():
    ins = keras.layers.Input(shape=[35, 1])
    #a1 = keras.layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(ins)
    c1 = keras.layers.Conv1D(64, 2, padding='same',activation='relu')(ins)
    pool1 = keras.layers.AveragePooling1D(pool_size=3)(c1)
    drop1 = keras.layers.Dropout(0.2)(pool1)

    c2 = keras.layers.Conv1D(64, 7, padding='same',activation='relu')(ins)
    pool2 = keras.layers.AveragePooling1D(pool_size=3)(c2)
    drop2 = keras.layers.Dropout(0.2)(pool2)

    c3 = keras.layers.Conv1D(64, 35, padding='same',activation='relu')(ins)
    pool3 = keras.layers.AveragePooling1D(pool_size=3)(c3)
    drop3 = keras.layers.Dropout(0.2)(pool3)

    Z1 = keras.layers.Concatenate()([drop1, drop2])
    Z2 = keras.layers.Concatenate()([drop2, drop3])
    z = keras.layers.Concatenate()([Z1, Z2])

    # 加入自定义的注意力层
    #z = keras.layers.Dense(64)(z)  # 增加维度以适应多头自注意力，如果需要的话

    # 加入多头自注意力层
    #attention_layer = CustomAttentionLayer(num_heads=2, key_dim=64)
    #attention_output, attention_scores = attention_layer(z)

    # 可选：使用额外的处理层来进一步处理注意力的输出
    #attention_output = keras.layers.Dense(128, activation="relu")(attention_output)

    lstm = keras.layers.LSTM(64, return_sequences=True)(z)
    drop = keras.layers.Dropout(0.2)(lstm)

    z = keras.layers.Flatten()(drop)
    d = keras.layers.Dense(35, activation='linear')(z)
    out = keras.layers.Dense(1,name='out')(d)
    
    return ins,out


def Train(ins,out):
    #ins,out = Model_Multi()
    reg_nn = keras.Model(inputs=ins, outputs=out)
    epoch = 300
    from tensorflow.keras.optimizers import Adam
    reg_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    history = reg_nn.fit(x_train, y_train, batch_size=64, epochs=epoch)
    reg_nn.evaluate(x_test, y_test)
    reg_nn.summary()
    y_pre = reg_nn.predict(x_test)
    y_pre_train = reg_nn.predict(x_train)

    r2 = metrics.r2_score(y_test, y_pre)
    mae = metrics.mean_absolute_error(y_test, y_pre)
    print('R2:', r2)
    print('MAE:', mae)
    #show_loss(epoch,history)
    show_mix(y_pre, y_test, y_train, y_pre_train, 'CNN-LSTM', 2.0, r2, mae, history)
    return x_train, y_train, x_test, y_test, y_pre,reg_nn

def show_loss(epoch,history):
    with plt.style.context(['science', 'no-latex']):
        plt.figure()
        plt.grid(linestyle='--')
        plt.plot(range(epoch), history.history['loss'], c='r', label='loss', linewidth=3)
        # plt.plot(range(epoch), history.history['val_loss'], c='b', label='val_loss', linewidth=1.5)
        plt.xticks(fontsize=30, fontweight='bold')  # 默认字体大小为10
        plt.yticks(fontsize=30, fontweight='bold')
        plt.xlabel('Epochs', fontsize=40, fontweight='bold')
        plt.ylabel('Loss', fontsize=40, fontweight='bold')
        # plt.legend()
        plt.legend(loc=0, numpoints=1)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
        plt.ylim(0, 0.1)
    plt.show()


def show_mix(y_pre, y_true, y_train, y_pre_train, title, lim, r2, mae, history):
    #with plt.style.context(['science']):
    plt.figure()
    #plt.grid(linestyle="--")  # 设置背景网格线为虚线  
    ax = plt.gca()
    ax.grid(which='major',axis='y',ls='--',c='gray',alpha=.7, linewidth = "2")
    #ax.set.axisbelow(True)
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
    ax.tick_params(bottom=True,direction='out',labelsize=28,width=3,length=4,
                   left=False)
    
    fontdict ={"size":30,"color":"k",'weight':'bold'}
    #ax.text(0.1,2.9,r'(a)',fontdict=fontdict)
    ax.text(2.1,0.9,r'$R^2=$'+str(round(r2,3)),fontdict=fontdict)
    ax.text(2.1,0.6,'MAE='+str(round(mae,3)),fontdict=fontdict)
    ax.text(2.1,0.3,r'MSE='+str(round(history.history['loss'][-1],5)),fontdict=fontdict)
    #ax.text(0.1,1.4,r'$N=$'+ str(N),fontdict=fontdict)
    
    from matplotlib.pyplot import MultipleLocator
    x_major_locator=MultipleLocator(0.3)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=MultipleLocator(0.3)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    #把y轴的主刻度设置为10的倍数
    
    # 绘制拟合线和阴影
    x = np.linspace(0, 4, 100)
    y = x  # 这是你的拟合线
    residuals = y_true - y_pre
    std_dev = np.std(residuals)  # 计算标准差
    plt.plot(x, y, c='#BD514A', linewidth=5, linestyle='-.', label='Fit line')
    plt.fill_between(x, y - std_dev, y + std_dev, color='red', alpha=0.2, label='Uncertainty interval')  # 阴影范围为0.1，可根据需要调整
    
    
    #plt.scatter(y_pre, y_true, edgecolors='#839DD1' , marker='o', s=200 , c='none', alpha=0.9, linewidths=3)
    #plt.scatter(y_pre, y_true, edgecolors='#304E7E' , marker='o', s=200 , c='#D2D6F5', alpha=1, linewidths=3)
    scatter_test = plt.scatter(y_pre, y_true, edgecolors='#3D3A99' , marker='o', s=200 , c='#7A7FFA', alpha=1, linewidths=2, label='Testing set')
    scatter_train = plt.scatter(y_pre_train, y_train, edgecolors='#E8BB50' , marker='o', s=200 , c='#FFF619', alpha=1, linewidths=2, label='Training set')
    #plt.plot(range(4), range(4), c='#BD514A', linewidth=5, linestyle=':')
    #plt.plot(range(4), range(4), c='#BD514A', linewidth=5, linestyle='-.')

    
    plt.xticks(fontsize=30, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=30, fontweight='bold')
    plt.title(title, fontsize=40, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("Predict", fontsize=40, fontweight='bold')
    plt.ylabel("Real", fontsize=40, fontweight='bold')
    plt.xlim((0, lim))
    plt.ylim((0.001, lim))
    #plt.legend(loc=0, numpoints=1)
    # 设置图例
    #legend = plt.legend(frameon=False, fontsize=50, loc='upper left')
    #legend = plt.legend(frameon=False, loc='upper left', fontsize=40, scatterpoints=1, markerscale=2)
    plt.legend(handles=[scatter_test, scatter_train], frameon=False, loc='upper left', fontsize=20, scatterpoints=1, markerscale=1.5)
    # 设置图例框的大小和位置，使其在图表内部
    #bbox_anchor = (0.05, 1)  # (x0, y0) 仅设置位置，不调整大小

    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=30, fontweight='bold')


    #plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用在线转换工具转为矢量图emf后插入word中

    plt.show()


#ins1,out1 = Model_Single()
ins2,out2 = Model_Multi()
a1 = Train(ins2,out2)

###################################
"""
x_train, y_train, x_test, y_test, y_pre, model = Train(ins2,out2)



y_pre = y_pre.reshape(y_test.shape)

# 总共需要选取的样本数量
total_samples = 500
# 每次迭代选取的样本数量
num_samples_per_iteration = 50
# 记录已选取的样本数量
selected_samples_count = 0

# 保持数据的 3D 形状，不要展平
while selected_samples_count < total_samples:
    # 计算预测误差
    errors = np.abs(y_pre - y_test)  # errors should be a 1D array with shape (16112,)

    # 选择误差最大的样本进行标记
    indices = np.argsort(errors)[-num_samples_per_iteration:]

    # 从测试集中选择最有信息量的样本
    X_selected = x_test[indices]
    y_selected = y_test[indices]

    # 将新标记的样本添加到训练集中
    x_train = np.vstack((x_train, X_selected))
    y_train = np.concatenate((y_train, y_selected))

    # 从测试集中移除已选取的样本
    x_test = np.delete(x_test, indices, axis=0)
    y_test = np.delete(y_test, indices, axis=0)
    print(x_train.shape, x_test.shape)
    
    # 确保保持3D形状进行训练
    history = model.fit(x_train, y_train, batch_size=64, epochs=100)

    # 再次评估模型性能
    y_pre = model.predict(x_test)
    y_pre_train = model.predict(x_train)
    mse = mean_squared_error(y_test, y_pre)
    mae = metrics.mean_absolute_error(y_test, y_pre)
    r2_old = 0
    r2 = r2_score(y_test, y_pre)
    print(f"Current MAE: {mae}")
    print(f"Current R² Score: {r2}")
    show_mix(y_pre, y_test, y_train, y_pre_train, 'CNN-LSTM', 2.0, r2, mae, history)
    if r2_old > r2:
        break  
    else:
        r2_old = r2
        # 更新已选取的样本数量
        selected_samples_count += num_samples_per_iteration
        y_pre = y_pre.reshape(y_test.shape)
"""