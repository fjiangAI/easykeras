from keras.layers import Embedding, Conv1D, Dense
from keras.utils import to_categorical
#from keras.utils.multi_gpu_utils import multi_gpu_model   #导入keras多GPU函数
import re
import numpy as np

__author__ = 'jf'
import keras
import sys

sys.path.append("../../")
from easykeras.example.imdb_util import Config
from keras.callbacks import Callback
from tqdm import tqdm
from easykeras.layers.crf import CRF

data_source = "http://sighan.cs.uchicago.edu/bakeoff2005/"


def get_config():
    return {"vocab_size": 10000,
            "word_embedding_size": 128,
            "n_tags": 5,
            "max_len": 600,
            }


class CNNCRFModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(CNNCRFModel, self).__init__(name='cnncrf_model')
        self.embedding_layer = Embedding(input_dim=config.vocab_size, output_dim=config.word_embedding_size)
        self.cnn1 = Conv1D(config.word_embedding_size, 3, activation='relu', padding='same')
        self.cnn2 = Conv1D(config.word_embedding_size, 3, activation='relu', padding='same')
        self.cnn3 = Conv1D(config.word_embedding_size, 3, activation='relu', padding='same')
        self.dense_layer = Dense(config.n_tags)
        self.crf = CRF(True)

    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = self.embedding_layer(inputs)
        cnn1_output = self.cnn1(embedding_output)
        cnn2_output = self.cnn2(cnn1_output)
        cnn3_output = self.cnn3(cnn2_output)
        dense_output = self.dense_layer(cnn3_output)  # 变成了5分类，第五个标签用来mask掉
        pred = self.crf(dense_output)
        return pred


# 自定义Callback类
class Evaluate(Callback):
    def __init__(self, valid_sents, tag2id, char2id, config):
        self.highest = 0.
        self.tag2id = tag2id
        self.char2id = char2id
        self.valid_sents = valid_sents
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        _ = self.model.get_weights()[-1][:4, :4]  # 从训练模型中取出最新得到的转移矩阵
        #获取状态转移概率
        trans = {}
        for i in 'sbme':
            for j in 'sbme':
                trans[i + j] = _[self.tag2id[i], self.tag2id[j]]
        right = 0.
        total = 0.
        for s in tqdm(iter(self.valid_sents), desc=u'验证模型中'):
            result = self.cut(''.join(s), trans)
            total += len(set(s))
            right += len(set(s) & set(result))  # 直接将词集的交集作为正确数。该指标比较简单，
            # 也许会导致估计偏高。读者可以考虑自定义指标
        acc = right / total
        if acc > self.highest:
            self.highest = acc
        print('val acc: %s, highest: %s' % (acc, self.highest))

    def cut(self, s, trans):  # 分词函数，也跟前面的HMM基本一致
        if not s:  # 空字符直接返回
            return []
        # 字序列转化为id序列。注意，经过我们前面对语料的预处理，字符集是没有空格的，
        # 所以这里简单将空格的id跟句号的id等同起来
        sent_ids = [self.char2id.get(c, 0) if c != ' ' else self.char2id[u'。'] for c in s]
        sent_ids = np.array([sent_ids])
        # 补足长度
        sent_ids = keras.preprocessing.sequence.pad_sequences(sent_ids, maxlen=self.config.max_len, dtype='int32',
                                                              padding='post', truncating='post', value=0.)
        #获取概率
        probas = self.model.predict(sent_ids)[0]  # 模型预测
        # 去除空字符的结果
        nodes = [dict(zip('sbme', i)) for i in probas[:, :self.config.n_tags - 1]]  # 只取前4个
        # 增加约束条件
        nodes[0] = {i: j for i, j in nodes[0].items() if i in 'bs'}  # 首字标签只能是b或s
        nodes[-1] = {i: j for i, j in nodes[-1].items() if i in 'es'}  # 末字标签只能是e或s
        # 获取最优的标签序列
        tags = self.viterbi(nodes, trans)[0]
        # 从这开始获取结果
        result = [s[0]]
        for i, j in zip(s[1:], tags[1:]):
            if j in 'bs':  # 词的开始
                result.append(i)
            else:  # 接着原来的词
                result[-1] += i
        return result

    def viterbi(self, nodes, trans):  # viterbi算法，跟前面的HMM一致
        def max_in_dict(d):  # 定义一个求字典中最大值的函数
            key, value = list(d.items())[0]
            for i, j in list(d.items())[1:]:
                if j > value:
                    key, value = i, j
            return key, value

        paths = nodes[0]  # 初始化起始路径
        for l in range(1, len(nodes)):  # 遍历后面的节点
            paths_old, paths = paths, {}
            for n, ns in nodes[l].items():  # 当前时刻的所有节点
                max_path, max_score = '', -1e10
                for p, ps in paths_old.items():  # 截止至前一时刻的最优路径集合
                    score = ns + ps + trans[p[-1] + n]  # 计算新分数
                    if score > max_score:  # 如果新分数大于已有的最大分
                        max_path, max_score = p + n, score  # 更新路径
                paths[max_path] = max_score  # 储存到当前时刻所有节点的最优路径
        return max_in_dict(paths)


def train_generator(train_sents, char2id):  # 定义数据生成器
    X, Y = [], []
    for i, s in enumerate(train_sents):  # 遍历每个句子
        sx, sy = [], []
        for w in s:  # 遍历句子中的每个词
            sx.extend([char2id.get(c, 0) for c in w])  # 遍历词中的每个字
            if len(w) == 1:
                sy.append(0)  # 单字词的标签
            elif len(w) == 2:
                sy.extend([1, 3])  # 双字词的标签
            else:
                sy.extend([1] + [2] * (len(w) - 2) + [3])  # 多于两字的词的标签
        X.append(sx)
        Y.append(sy)
    maxlen = max([len(x) for x in X])  # 找出最大字数
    X = [x + [0] * (maxlen - len(x)) for x in X]  # 不足则补零
    Y = [y + [4] * (maxlen - len(y)) for y in Y]  # 不足则补第五个标签
    return np.array(X), to_categorical(Y, 5), maxlen


def get_data(filepath='./icwb2-data/training/msr_training.utf8'):
    sents = open(filepath, encoding="utf-8").read()
    sents = sents.strip()
    sents = sents.split('\n')  # 这个语料的换行符是\n
    sents = [re.split(' +', s) for s in sents]  # 词之间以空格隔开
    sents = [[w for w in s if w] for s in sents]  # 去掉空字符串
    sents = sents[:10000]  # 只用前10000个做小样本训练
    np.random.shuffle(sents)  # 打乱语料，以便后面划分验证集

    chars = {}  # 统计字表
    for s in sents:
        for c in ''.join(s):
            if c in chars:
                chars[c] += 1
            else:
                chars[c] = 1

    min_count = 2  # 过滤低频字
    chars = {i: j for i, j in chars.items() if j >= min_count}  # 过滤低频字
    id2char = {i + 1: j for i, j in enumerate(chars)}  # id到字的映射
    char2id = {j: i for i, j in id2char.items()}  # 字到id的映射
    id2tag = {0: 's', 1: 'b', 2: 'm', 3: 'e'}  # 标签（sbme）与id之间的映射
    tag2id = {j: i for i, j in id2tag.items()}
    train_sents = sents[:-2000]  # 留下2000个句子做验证，剩下的都用来训练
    valid_sents = sents[-2000:]
    return train_sents, valid_sents, tag2id, char2id


if __name__ == "__main__":
    # 主程序
    model_config = Config(get_config())
    cnncrf_model = CNNCRFModel(model_config)
    train_sents, valid_sents, tag2id, char2id = get_data()
    x_train, y_train, max_len = train_generator(train_sents, char2id)
    model_config.vocab_size = len(char2id) + 1
    model_config.max_len = max_len
    # 设定编译参数
    loss = cnncrf_model.crf.loss
    optimizer = 'adam'
    metrics = [cnncrf_model.crf.accuracy]
    #parallel_model = multi_gpu_model(cnncrf_model, gpus=2)
    cnncrf_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # 设置训练超参数
    batch_size = 128
    epochs = 10
    evaluator = Evaluate(valid_sents, tag2id, char2id, model_config)  # 建立Callback类
    # 训练过程
    cnncrf_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[evaluator])
