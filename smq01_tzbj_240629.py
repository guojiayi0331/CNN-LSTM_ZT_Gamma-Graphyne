# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:50:16 2024

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# 加载数据
with open(r'D:\gjy_by\gjy_by\smq\graphyneQ_15_ZT_16512_change.csv') as f:
    df_Q_ch = pd.read_csv(f)

with open(r'D:\gjy_by\gjy_by\smq\graphyneQ_15_ZT_16512.csv') as f:
    df_Q = pd.read_csv(f)

with open(r'D:\gjy_by\gjy_by\smq\graphyne_15_ZT_16512_change.csv') as f:
    df_ch = pd.read_csv(f)

with open(r'D:\gjy_by\gjy_by\smq\graphyne_15_ZT_16512.csv') as f:
    df = pd.read_csv(f)

test_ch = df_ch['T=300k']
df_all = df.loc[:, 'parameter1':]
df_all[['T=300k']] = df[['T=300k']]

data_old = df_all.values.tolist()
data = []
for row in data_old:
    binary_code = [int(bit) for bit in row[:-1]]
    zt_value = row[-1]
    data.append((binary_code, zt_value))

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
X_extracted = [extract_pattern_features(sample[0], pattern_lengths) for sample in data]
y = [sample[1] for sample in data]

# 转换为DataFrame
X_extracted_df = pd.DataFrame(X_extracted)
X_binary_df = df.loc[:, 'parameter1':'parameter15']

# 合并原始二进制编码特征和提取的特征
X_combined_df = pd.concat([X_binary_df, X_extracted_df], axis=1)
X_combined_df['zt'] = y

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_combined_df.drop(columns=['zt']), X_combined_df['zt'], test_size=0.9, random_state=42)

# 定义神经网络模型
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.AveragePooling1D(pool_size=1),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(64, 7, padding='same', activation='relu'),
        keras.layers.AveragePooling1D(pool_size=1),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(64, 35, padding='same', activation='relu'),
        keras.layers.AveragePooling1D(pool_size=1),
        keras.layers.Dropout(0.2),
        
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(35, activation='tanh'),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# 训练模型
model = create_model((X_train.shape[1], 1))
history = model.fit(np.expand_dims(X_train, axis=2), y_train, epochs=500, batch_size=32, validation_split=0.2)

# 预测结果
y_pred_test = model.predict(np.expand_dims(X_test, axis=2))
y_pred_train = model.predict(np.expand_dims(X_train, axis=2))

# 计算评估指标
r2 = r2_score(y_test, y_pred_test)
mae = mean_squared_error(y_test, y_pred_test, squared=False)
print('R2:', r2)
print('MAE:', mae)

# 将预测结果和真实数据结合
results_train_df = X_train.copy()
results_train_df['True ZT'] = y_train
results_train_df['Predicted ZT'] = y_pred_train.flatten()

results_test_df = X_test.copy()
results_test_df['True ZT'] = y_test
results_test_df['Predicted ZT'] = y_pred_test.flatten()

# 合并训练集和测试集的预测结果
results_combined_df = pd.concat([results_train_df, results_test_df])

# 真实数据
all_true_df = X_combined_df.copy()
all_true_df['True ZT'] = y

# 定义高ZT值和低ZT值的阈值
high_ZT_threshold = 2.0
low_ZT_threshold = 1.6

# 分组数据
high_ZT_pred = results_combined_df[results_combined_df['Predicted ZT'] >= high_ZT_threshold]
low_ZT_pred = results_combined_df[results_combined_df['Predicted ZT'] <= low_ZT_threshold]

high_ZT_true = all_true_df[all_true_df['True ZT'] >= high_ZT_threshold]
low_ZT_true = all_true_df[all_true_df['True ZT'] <= low_ZT_threshold]

print(high_ZT_pred,high_ZT_true)

"""
# 设置Seaborn和Matplotlib的样式以接近Nature期刊的风格
sns.set(style="whitegrid", palette="muted")

# 调整字体
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman']
})

# 比较关键特征分布
key_features = X_combined_df.drop(columns=['zt']).columns
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=high_ZT_true, x=feature, label='High True ZT', color='dodgerblue', kde=False, bins=1, alpha=0.6, stat="density", discrete=True)
    sns.histplot(data=low_ZT_true, x=feature, label='Low True ZT', color='red', kde=False, bins=1, alpha=0.6, stat="density", discrete=True)
    sns.histplot(data=high_ZT_pred, x=feature, label='High Predicted ZT', color='blue', kde=False, bins=1, alpha=0.6, stat="density", discrete=True, linestyle="--")
    sns.histplot(data=low_ZT_pred, x=feature, label='Low Predicted ZT', color='darkred', kde=False, bins=1, alpha=0.6, stat="density", discrete=True, linestyle="--")
    
    plt.title(f'Distribution of {feature}', fontsize=24, fontweight='bold')
    plt.xlabel(feature, fontsize=22)
    plt.ylabel('Density', fontsize=22)
    plt.ylim(0, 1)
    plt.xlim(-1, 7)
    plt.legend()
    sns.despine()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # 保存图片，设置高分辨率
    plt.savefig(f'picture\\{feature}_distribution.png', dpi=600)
    plt.show()
"""

# 设置Seaborn和Matplotlib的样式以接近Nature期刊的风格
sns.set(style="whitegrid", palette="muted")

# 调整字体
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman']
})

# 比较关键特征分布
key_features = X_combined_df.drop(columns=['zt']).columns
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=high_ZT_true, x=feature, label='High True ZT', fill=True, color='dodgerblue')
    sns.kdeplot(data=low_ZT_true, x=feature, label='Low True ZT', fill=True, color='red')
    sns.kdeplot(data=high_ZT_pred, x=feature, label='High Predicted ZT', fill=True, color='blue', linestyle="--")
    sns.kdeplot(data=low_ZT_pred, x=feature, label='Low Predicted ZT', fill=True, color='darkred', linestyle="--")
    plt.title(f'Distribution of {feature}', fontsize=24, fontweight='bold')
    plt.xlabel(feature, fontsize=22)
    plt.ylabel('Density', fontsize=22)
    #plt.ylim(0, 1)
    #plt.legend()
    sns.despine()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # 保存图片，设置高分辨率
    plt.savefig(f'picture\\1\\{feature}_distribution.png', dpi=1200)
    plt.show()
