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
        self.min_count = 32
        self.maxlen = 400
        self.batch_size = 256
        self.epochs = 100
        self.char_size = 128
        self.z_dim = 128
        self.model_path = "model/model.weights"
        self.config_path = "model/seq2seq_config.json"
        self.char2id, self.id2char, self.chars = self.load_config()
        self.model = self.load_model()
        return

    """加载基础词典信息"""
    def load_config(self):
        chars, id2char, char2id = json.load(open(self.config_path))
        id2char = {int(i): j for i, j in id2char.items()}
        return chars, id2char, char2id

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

    """对文档进行paddle处理"""
    def padding(self, x):
        # padding到batch内最大长度
        ml = max([len(i) for i in x])
        return [i + [0] * (ml - len(i)) for i in x]  # 长度不够这填充0

    """将句子应设成为一个onehot向量"""
    def to_one_hot(self, x):
        # 转one_hot  输出一个词表大小的向量，来标记该词是否在文章出现过
        x, x_mask = x
        x = K.cast(x, 'int32')  # 相当于转换类型
        x = K.one_hot(x, len(self.chars) + 4)  # 转one_hot
        x = K.sum(x_mask * x, 1, keepdims=True)
        x = K.cast(K.greater(x, 0.5), 'float32')  # 相当于那个先验知识
        return x

    """加载模型权重"""
    def load_model(self):
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
        model.load_weights(self.model_path)
        return model

    """使用beamsearch算法进行模型预测"""
    def gen_sent(self, s, topk=3, maxlen=64):
        # beam search解码 :每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        xid = np.array([self.str2id(s)] * topk)  # 输入转id
        yid = np.array([[2]] * topk)  # 解码均以<start>开头，这里<start>的id为2
        scores = [0] * topk  # 候选答案分数
        for i in range(maxlen):  # 强制要求输出不超过maxlen字
            proba = self.model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
            log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
            arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _yid = []  # 暂存的候选目标序列
            _scores = []  # 暂存的候选目标序列得分
            if i == 0:
                for j in range(topk):
                    _yid.append(list(yid[j]) + [arg_topk[0][j] + 3])
                    _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
            else:
                for j in range(topk):
                    for k in range(topk):  # 遍历topk*topk的组合
                        _yid.append(list(yid[j]) + [arg_topk[j][k] + 3])
                        _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
                _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
                _yid = [_yid[k] for k in _arg_topk]
                _scores = [_scores[k] for k in _arg_topk]
            yid = np.array(_yid)
            scores = np.array(_scores)
            ends = np.where(yid[:, -1] == 3)[0]
            if len(ends) > 0:
                k = ends[scores[ends].argmax()]
                return self.id2str(yid[k])
        # 如果maxlen字都找不到<end>，直接返回
        return self.id2str(yid[np.argmax(scores)])

    """利用模型进行预测"""
    def predict(self, sent):
        return self.gen_sent(sent)

if __name__ == "__main__":
    handler = SeqAttModel()
    while 1:
        sent = input('enter an sent:').strip()
        res = handler.predict(sent)
        print(res)