#! -*- coding: utf-8 -*-
# 追一科技2019年NL2SQL挑战赛的一个Baseline（个人作品，非官方发布，基于Bert）
# 比赛地址：https://tianchi.aliyun.com/competition/entrance/231716/introduction
# 目前全匹配率大概是50%左右

import json
import uniout
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from tqdm import tqdm
import jieba
import editdistance
import re


maxlen = 160
num_agg = 7 # agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
num_op = 5 # {0:">", 1:"<", 2:"==", 3:"!=", 4:"不被select"}
num_cond_conn_op = 3 # conn_sql_dict = {0:"", 1:"and", 2:"or"}
learning_rate = 5e-5
min_learning_rate = 1e-5


config_path = '../../kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../../kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'


def read_data(data_file, table_file):
    data, tables = [], {}
    with open(data_file) as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file) as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables


train_data, train_tables = read_data('../datasets/train.json', '../datasets/train.tables.json')
valid_data, valid_tables = read_data('../datasets/val.json', '../datasets/val.tables.json')
test_data, test_tables = read_data('../datasets/test.json', '../datasets/test.tables.json')


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])


def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]


def most_similar_2(w, s):
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)


class data_generator:
    def __init__(self, data, tables, batch_size=32):
        self.data = data
        self.tables = tables
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(idxs)
            X1, X2, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                t = self.tables[d['table_id']]['headers']
                x1, x2 = tokenizer.encode(d['question'])
                xm = [0] + [1] * len(d['question']) + [0]
                h = []
                for j in t:
                    _x1, _x2 = tokenizer.encode(j)
                    h.append(len(x1))
                    x1.extend(_x1)
                    x2.extend(_x2)
                hm = [1] * len(h)
                sel = []
                for j in range(len(h)):
                    if j in d['sql']['sel']:
                        j = d['sql']['sel'].index(j)
                        sel.append(d['sql']['agg'][j])
                    else:
                        sel.append(num_agg - 1) # 不被select则被标记为num_agg-1
                conn = [d['sql']['cond_conn_op']]
                csel = np.zeros(len(d['question']) + 2, dtype='int32') # 这里的0既表示padding，又表示第一列，padding部分训练时会被mask
                cop = np.zeros(len(d['question']) + 2, dtype='int32') + num_op - 1 # 不被select则被标记为num_op-1
                for j in d['sql']['conds']:
                    if j[2] not in d['question']:
                        j[2] = most_similar_2(j[2], d['question'])
                    if j[2] not in d['question']:
                        continue
                    k = d['question'].index(j[2])
                    csel[k + 1: k + 1 + len(j[2])] = j[0]
                    cop[k + 1: k + 1 + len(j[2])] = j[1]
                if len(x1) > maxlen:
                    continue
                X1.append(x1) # bert的输入
                X2.append(x2) # bert的输入
                XM.append(xm) # 输入序列的mask
                H.append(h) # 列名所在位置
                HM.append(hm) # 列名mask
                SEL.append(sel) # 被select的列
                CONN.append(conn) # 连接类型
                CSEL.append(csel) # 条件中的列
                COP.append(cop) # 条件中的运算符（同时也是值的标记）
                if len(X1) == self.batch_size:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    XM = seq_padding(XM, maxlen=X1.shape[1])
                    H = seq_padding(H)
                    HM = seq_padding(HM)
                    SEL = seq_padding(SEL)
                    CONN = seq_padding(CONN)
                    CSEL = seq_padding(CSEL, maxlen=X1.shape[1])
                    COP = seq_padding(COP, maxlen=X1.shape[1])
                    yield [X1, X2, XM, H, HM, SEL, CONN, CSEL, COP], None
                    X1, X2, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], []


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, n]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, n, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    return K.tf.batch_gather(seq, idxs)


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

for l in bert_model.layers:
    l.trainable = True


x1_in = Input(shape=(None,), dtype='int32')
x2_in = Input(shape=(None,))
xm_in = Input(shape=(None,))
h_in = Input(shape=(None,), dtype='int32')
hm_in = Input(shape=(None,))
sel_in = Input(shape=(None,), dtype='int32')
conn_in = Input(shape=(1,), dtype='int32')
csel_in = Input(shape=(None,), dtype='int32')
cop_in = Input(shape=(None,), dtype='int32')

x1, x2, xm, h, hm, sel, conn, csel, cop = (
    x1_in, x2_in, xm_in, h_in, hm_in, sel_in, conn_in, csel_in, cop_in
)

hm = Lambda(lambda x: K.expand_dims(x, 1))(hm) # header的mask.shape=(None, 1, h_len)

x = bert_model([x1_in, x2_in])
x4conn = Lambda(lambda x: x[:, 0])(x)
pconn = Dense(num_cond_conn_op, activation='softmax')(x4conn)

x4h = Lambda(seq_gather)([x, h])
psel = Dense(num_agg, activation='softmax')(x4h)

pcop = Dense(num_op, activation='softmax')(x)

x = Lambda(lambda x: K.expand_dims(x, 2))(x)
x4h = Lambda(lambda x: K.expand_dims(x, 1))(x4h)
pcsel_1 = Dense(1)(x)
pcsel_2 = Dense(1)(x4h)
pcsel = Lambda(lambda x: x[0] + x[1])([pcsel_1, pcsel_2])
pcsel = Lambda(lambda x: x[0][..., 0] - (1 - x[1]) * 1e10)([pcsel, hm])
pcsel = Activation('softmax')(pcsel)

model = Model(
    [x1_in, x2_in, h_in, hm_in],
    [psel, pconn, pcop, pcsel]
)

train_model = Model(
    [x1_in, x2_in, xm_in, h_in, hm_in, sel_in, conn_in, csel_in, cop_in],
    [psel, pconn, pcop, pcsel]
)

xm = xm # question的mask.shape=(None, x_len)
hm = hm[:, 0] # header的mask.shape=(None, h_len)
cm = K.cast(K.not_equal(cop, num_op - 1), 'float32') # conds的mask.shape=(None, x_len)

psel_loss = K.sparse_categorical_crossentropy(sel_in, psel)
psel_loss = K.sum(psel_loss * hm) / K.sum(hm)
pconn_loss = K.sparse_categorical_crossentropy(conn_in, pconn)
pconn_loss = K.mean(pconn_loss)
pcop_loss = K.sparse_categorical_crossentropy(cop_in, pcop)
pcop_loss = K.sum(pcop_loss * xm) / K.sum(xm)
pcsel_loss = K.sparse_categorical_crossentropy(csel_in, pcsel)
pcsel_loss = K.sum(pcsel_loss * xm * cm) / K.sum(xm * cm)
loss = psel_loss + pconn_loss + pcop_loss + pcsel_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


def nl2sql(question, table):
    """输入question和headers，转SQL
    """
    x1, x2 = tokenizer.encode(question)
    h = []
    for i in table['headers']:
        _x1, _x2 = tokenizer.encode(i)
        h.append(len(x1))
        x1.extend(_x1)
        x2.extend(_x2)
    hm = [1] * len(h)
    psel, pconn, pcop, pcsel = model.predict([
        np.array([x1]),
        np.array([x2]),
        np.array([h]),
        np.array([hm])
    ])
    R = {'agg': [], 'sel': []}
    for i, j in enumerate(psel[0].argmax(1)):
        if j != num_agg - 1: # num_agg-1类是不被select的意思
            R['sel'].append(i)
            R['agg'].append(j)
    conds = []
    v_op = -1
    for i, j in enumerate(pcop[0, :len(question)+1].argmax(1)):
        # 这里结合标注和分类来预测条件
        if j != num_op - 1:
            if v_op != j:
                if v_op != -1:
                    v_end = v_start + len(v_str)
                    csel = pcsel[0][v_start: v_end].mean(0).argmax()
                    conds.append((csel, v_op, v_str))
                v_start = i
                v_op = j
                v_str = question[i - 1]
            else:
                v_str += question[i - 1]
        elif v_op != -1:
            v_end = v_start + len(v_str)
            csel = pcsel[0][v_start: v_end].mean(0).argmax()
            conds.append((csel, v_op, v_str))
            v_op = -1
    R['conds'] = set()
    for i, j, k in conds:
        if re.findall('[^\d\.]', k):
            j = 2 # 非数字只能用等号
        if j == 2:
            if k not in table['all_values']:
                # 等号的值必须在table出现过，否则找一个最相近的
                k = most_similar(k, list(table['all_values']))
            h = table['headers'][i]
            # 然后检查值对应的列是否正确，如果不正确，直接修正列名
            if k not in table['content'][h]:
                for r, v in table['content'].items():
                    if k in v:
                        i = table['header2id'][r]
                        break
        R['conds'].add((i, j, k))
    R['conds'] = list(R['conds'])
    if len(R['conds']) <= 1: # 条件数少于等于1时，条件连接符直接为0
        R['cond_conn_op'] = 0
    else:
        R['cond_conn_op'] = 1 + pconn[0, 1:].argmax() # 不能是0
    return R


def is_equal(R1, R2):
    """判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op']) &\
    (set(zip(R1['sel'], R1['agg'])) == set(zip(R2['sel'], R2['agg']))) &\
    (set([tuple(i) for i in R1['conds']]) == set([tuple(i) for i in R2['conds']]))


def evaluate(data, tables):
    right = 0.
    pbar = tqdm()
    F = open('evaluate_pred.json', 'w')
    for i, d in enumerate(data):
        question = d['question']
        table = tables[d['table_id']]
        R = nl2sql(question, table)
        right += float(is_equal(R, d['sql']))
        pbar.update(1)
        pbar.set_description('< acc: %.5f >' % (right / (i + 1)))
        d['sql_pred'] = R
        s = json.dumps(d, ensure_ascii=False, indent=4)
        F.write(s.encode('utf-8') + '\n')
    F.close()
    pbar.close()
    return right / len(data)


def test(data, tables, outfile='result.json'):
    pbar = tqdm()
    F = open(outfile, 'w')
    for i, d in enumerate(data):
        question = d['question']
        table = tables[d['table_id']]
        R = nl2sql(question, table)
        pbar.update(1)
        s = json.dumps(R, ensure_ascii=False)
        F.write(s.encode('utf-8') + '\n')
    F.close()
    pbar.close()

# test(test_data, test_tables)


class Evaluate(Callback):
    def __init__(self):
        self.accs = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.accs.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print 'acc: %.5f, best acc: %.5f\n' % (acc, self.best)
    def evaluate(self):
        return evaluate(valid_data, valid_tables)


train_D = data_generator(train_data, train_tables)
evaluator = Evaluate()

if __name__ == '__main__':
    train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=15,
        callbacks=[evaluator]
    )
else:
    train_model.load_weights('best_model.weights')
