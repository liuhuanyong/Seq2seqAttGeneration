# coding=utf-8
import os
import json
import numpy as np
from keras.layers import Input, Lambda, Embedding, LSTM, LeakyReLU, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from att_seq import *

class SeqAttModel:
    def __init__(self):
        self.data_path = "data/data.txt"
        self.min_count = 32
        self.maxlen = 400
        self.batch_size = 256
        self.epochs = 50
        self.char_size = 128
        self.z_dim = 128
        self.model_path = "model/model.weights"
        self.config_path = "model/seq2seq_config.json"
        self.image_path = "image/network.png"
        self.inputs, self.outputs = self.load_data(self.data_path)
        self.char2id, self.id2char, self.chars = self.build_config(self.inputs, self.outputs)
        self.model = self.load_data(self.model_path)
        return

    """对序列进行平均池化操作"""
    def seq_avgpool(self, x):
        # seq是[None, seq_len, s_size]的格式，mask是[None, seq_len, 1]的格式，先除去mask部分， 然后再做avgpooling。
        seq, mask = x
        return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)

    """对序列进行最大池化操作"""
    def seq_maxpool(self, x):
        # seq是[None, seq_len, s_size]的格式，mask是[None, seq_len, 1]的格式，先除去mask部分，然后再做maxpooling。
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return K.max(seq, 1)

    """加载训练语料"""
    def load_data(self, path):
        inputs = []
        outputs = []
        with open(path, 'r', errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.split('\t')
                inputs.append(line[-1])
                outputs.append(line[0])
        return inputs[:10000], outputs[:10000]

    """保存词典配置信息"""
    def build_config(self, inputs, outputs):
        chars = {}
        for t in inputs:
            for w in t:
                chars[w] = chars.get(w, 0) + 1
        for c in outputs:
            for w in c:
                chars[c] = chars.get(w, 0) + 1

        chars = {i: j for i, j in chars.items()}  # 过滤低频次
        # 0:mask, 1:unk, 2:start, 3:end
        id2char = {i + 4: j for i, j in enumerate(chars)}
        char2id = {j: i for i, j in id2char.items()}
        print(char2id)
        print(id2char)
        json.dump([chars, id2char, char2id], open(self.config_path, 'w+'))
        return char2id, id2char, chars

    """将字符转换成id"""
    def str2id(self, s, start_end=False):
        if start_end:  # 若是开始或结束补上<start>和<end>标记
            ids = [self.char2id.get(c, 1) for c in s[:self.maxlen - 2]]  # 转语料为id
            ids = [2] + ids + [3]  # 加头尾
        else:
            ids = [self.char2id.get(c, 1) for c in s[:self.maxlen]]
        return ids

    """将id转换成字符"""
    def id2str(self, ids):
        # 数字转汉字，没有填充''
        return ''.join([self.id2char.get(i, '') for i in ids])

    """对序列进行补全"""
    def padding(self, x):
        # padding到batch内最大长度
        ml = max([len(i) for i in x])
        return [i + [0] * (ml - len(i)) for i in x]  # 长度不够这填充0

    """针对数据进行样本生成"""
    def data_generator(self):
        # 数据生成器
        X, Y = [], []
        while True:
            for t, c in zip(self.inputs, self.outputs):
                X.append(self.str2id(c))
                Y.append(self.str2id(t, start_end=True))  # 只需给标题加开始和结尾
                if len(X) == self.batch_size:
                    X = np.array(self.padding(X))
                    Y = np.array(self.padding(Y))
                    yield [X, Y], None
                    X, Y = [], []

    """将文本转换成one-hot形式"""
    def to_one_hot(self, x):
        # 转one_hot  输出一个词表大小的向量，来标记该词是否在文章出现过
        x, x_mask = x
        x = K.cast(x, 'int32')  # 相当于转换类型
        x = K.one_hot(x, len(self.chars) + 4)  # 转one_hot
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')  # 相当于那个先验知识
        return x

    """搭建模型"""
    def build_model(self):
        # 搭建seq2seq模型
        x_in = Input(shape=(None,))
        y_in = Input(shape=(None,))
        x, y = x_in, y_in

        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

        x_one_hot = Lambda(self.to_one_hot)([x, x_mask])
        x_prior = ScaleShift()(x_one_hot)  # 学习输出的先验分布（标题的字词很可能在文章出现过）

        embedding = Embedding(len(self.chars) + 4, self.char_size)
        x = embedding(x)
        y = embedding(y)

        # encoder，双层双向LSTM
        x = LayerNormalization()(x)
        x = OurBidirectional(LSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x = LayerNormalization()(x)
        x = OurBidirectional(LSTM(self.z_dim // 2, return_sequences=True))([x, x_mask])
        x_max = Lambda(self.seq_maxpool)([x, x_mask])

        # decoder，双层单向LSTM
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = LSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])
        y = LSTM(self.z_dim, return_sequences=True)(y)
        y = SelfModulatedLayerNormalization(self.z_dim // 4)([y, x_max])

        # attention交互
        xy = Attention(8, 16)([y, x, x, x_mask])
        xy = Concatenate()([y, xy])

        # 输出分类
        xy = Dense(self.char_size)(xy)
        xy = LeakyReLU(0.2)(xy)
        xy = Dense(len(self.chars) + 4)(xy)
        xy = Lambda(lambda x: (x[0] + x[1]) / 2)([xy, x_prior])  # 与先验结果平均
        xy = Activation('softmax')(xy)

        # 交叉熵作为loss，但mask掉padding部分
        cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
        cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])
        model = Model([x_in, y_in], xy)
        model.add_loss(cross_entropy)
        model.compile(optimizer=Adam(1e-3))
        return model

    """训练模型"""
    def train_model(self):
        model = self.build_model()
        model.summary()
        model.fit_generator(self.data_generator(),
                            steps_per_epoch=1000,
                            epochs=self.epochs)
        model.save_weights(self.model_path)
        return

if __name__ == "__main__":
    handler = SeqAttModel()
    handler.train_model()