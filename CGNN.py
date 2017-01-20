import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback,ModelCheckpoint
import time

embed_dim = 300
maxlen = 8

#nb_filter1 = 512
#filter_length1 = 2
#pool_length1 = maxlen - filter_length1 + 1
addSize = 256

#nb_epoch = 100
#batch_size = 128

#train_file = 'train.csv'
train_file = 'CoNLL2009-ST-English-train.txt'
#test_file = 'train.csv'
#test_file = 'test.csv'
test_file = 'test.wsj.closed.GOLD'
#embed_file = 'glove.6B.50d.txt'
embed_file = 'glove.840B.300d.txt'

vocab = set()
word_index = {}

X_train_1 = []
X_train_2 = []
Y_train = []

def data_proc(file):
    f = open(file)
    sentence = []
    now = []
    for line in f:
        line = line.strip('\n')
        tmp = line.split('\t')
        if tmp[0] == '':
            sentence.append(now)
            now = []
            continue
        now.append(tmp)
    f.close()
    if file == train_file:
        length = len(sentence)
        length /= 10
        print 'number of training sentence', length
        return sentence[ : length]
    else:
        length = len(sentence)
        #print 'number of testing sentence', length
        return sentence[ : length]

def vocab_proc(file):
    sentence = data_proc(file)
    for i in range(len(sentence)):
        for j in range(len(sentence[i])):
            vocab.add(sentence[i][j][1].lower())
    vocab.add(' ')

def read_pre_train(file):
    embeddings_index = {}
    f = open(file)
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1 : ], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    embeddings_index[' '] = list(np.zeros(embed_dim))
    return embeddings_index

def WE_proc():
    vocab_proc(train_file)
    vocab_proc(test_file)
    idx = 1
    for w in vocab:
        word_index[w] = idx
        idx += 1
    WE = np.zeros((len(vocab) + 1, embed_dim), dtype = 'float32')
    pre_trained = read_pre_train(embed_file)
    for x in vocab:
        if pre_trained.has_key(x):
            WE[word_index[x], : ] = pre_trained[x]
        else:
            WE[word_index[x], : ] = np.array(np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (embed_dim,)),dtype='float32')
    return WE

def CGNN(nb_filter1, filter_length1, pool_length1, nb_class):
    input1 = Sequential()
    input1.add(Embedding(input_dim = len(vocab) + 1, input_length = maxlen, weights = [WE], output_dim = embed_dim))
    input1.add(Convolution1D(nb_filter = nb_filter1,
                                        filter_length = filter_length1,
                                        border_mode = 'valid',
                                        activation = 'tanh',
                                        subsample_length = 1))

    input2 = Sequential()
    input2.add(Embedding(input_dim = len(vocab) + 1, input_length = maxlen, weights = [WE], output_dim = embed_dim))
    input2.add(Convolution1D(nb_filter = nb_filter1,
                                        filter_length = filter_length1,
                                        border_mode = 'valid',
                                        activation = 'tanh',
                                        subsample_length = 1))


    merge = Merge([input1, input2], mode='concat')

    model = Sequential()
    model.add(merge)
    model.add(MaxPooling1D(pool_length = pool_length1))
    model.add(LSTM(addSize))
    model.add(Dense(nb_class, activation='softmax'))

    # ada=Adagrad(lr=lr, epsilon=1e-06)
    # sgd = SGD(lr=lr)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    model.summary()
    return model

def embed_proc(now, x):
    root = 0
    for i in range(len(now)):
        if fa[i] == len(now):
            root = i
            break
    tmp = []
    for i in range(8):
        tmp.append(word_index[' '])
    if x != len(now):
        for i in range(x - 2, x + 3):
            if i >= 0 and i < len(now):
                tmp[i - x + 2] = word_index[now[i][1].lower()]
    if x != root and x != len(now):
        tmp[5] = word_index[now[fa[x]][1].lower()]
    if len(edge[x]):
        tmp[6] = word_index[now[edge[x][0]][1].lower()]
        tmp[7] = word_index[now[edge[x][-1]][1].lower()]
    return tmp

#train
def out(now, x, y, label):
    X_train_1.append(embed_proc(now, x))
    X_train_2.append(embed_proc(now, y))
    Y_train.append(label)

#test
def test_result(now, x, y):
    X_test_1 = []
    X_test_2 = []
    X_test_1.append(embed_proc(now, x))
    X_test_2.append(embed_proc(now, y))
    X_test_1 = np.array(X_test_1)
    X_test_2 = np.array(X_test_2)

    global mod
    Y_test = mod.predict([X_test_1, X_test_2])
    result = encoder.inverse_transform(np.argmax(Y_test))
    return result

# predicate_label
def get_predicate_label(st):
    if st == '_':
        return '_'
    tmp = 0
    for i in range(len(st)):
        if st[i] == '.':
            tmp = i + 1
            break
    return st[tmp : ]

#check noun & verb
def check_N_V(st):
    return st[0] == 'N' or st[0] == 'V'

def train_traverse(now, x, z, rest):
    #constant
    Predicate_label = 13

    for y in edge[x]:
        if rest <= 0:
            #print sentence[i][z][1], sentence[i][y][1], 'NoMoreArg'
            out(now, z, y, 'NoMoreArg')
            return
        else:
            #print sentence[i][z][1], sentence[i][y][1], sentence[i][y][Predicate_label + 1 + predicate[z]]
            out(now, z, y, now[y][Predicate_label + 1 + predicate[z]])
        if now[y][Predicate_label + 1 + predicate[z]] != '_':
            rest -= 1
    if (x != len(now)):
        train_traverse(now, fa[x], z, rest)

def test_traverse(now, x, z, rest):
    # constant
    Predicate_label = 10

    global num_predict_argument, num_correct
    for y in edge[x]:
        result = test_result(now, z, y)
        if result != '_':
            #print sentence[i][z][1], sentence[i][y][1], result
            num_predict_argument += 1
            if result == 'NoMoreArg':
                if rest == 0:
                    num_correct += 1
                return
            else:
                if predicate.has_key(z):
                    if result == now[y][Predicate_label + 1 + predicate[z]]:
                        num_correct += 1
        if predicate.has_key(z):
            if now[y][Predicate_label + 1 + predicate[z]] != '_' or rest == 0:
                rest -= 1
    if (x != len(now)):
        test_traverse(now, fa[x], z, rest)

def count(now, x, z, rest):
    # constant
    Predicate_label = 10

    global num_require_argument
    for y in edge[x]:
        if rest == 0:
            num_require_argument += 1
            return
        if now[y][Predicate_label + 1 + predicate[z]] != '_':
            rest -= 1
    if (x != len(now)):
        count(now, fa[x], z, rest)


#training
def train():
    sentence = data_proc(train_file)

    #constant
    Offset = 0  # 0 for goleden and 1 for predict
    Predicate_label = 13
    Father = 8 + Offset
    Postag = 4 + Offset

    for i in range(len(sentence)):
        # clear global array
        global predicate, fa, edge
        predicate = {}
        fa = []
        edge = []
        for j in range(len(sentence[i]) + 1):
            edge.append([])
        #local var
        cnt = 0
        # deal
        for j in range(len(sentence[i])):
            #construct tree
            fa.append(int(sentence[i][j][Father]) - 1)
            if fa[j] == -1:
                fa[j] = len(sentence[i])
            edge[fa[j]].append(j)
            #correspond position of predicate
            if sentence[i][j][Predicate_label] != '_':
                predicate[j] = cnt
                cnt += 1
        for j in range(len(sentence[i])):
            if check_N_V(sentence[i][j][Postag]):
                # print sentence[i][j][1], get_label(sentence[i][j][Predicate_label])
                out(sentence[i], len(sentence[i]), j, get_predicate_label(sentence[i][j][Predicate_label]))
            if sentence[i][j][Predicate_label] != '_':
                rest = 0
                for k in range(len(sentence[i])):
                    if sentence[i][k][Predicate_label + 1 + predicate[j]] != '_':
                        rest += 1
                train_traverse(sentence[i], j, j, rest)
    global X_train_1, X_train_2, Y_train
    X_train_1 = np.array(X_train_1)
    X_train_2 = np.array(X_train_2)
    print 'number of training pairs', X_train_1.shape[0]

    # encode class values as integers
    global encoder
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    C = np.size(np.unique(encoded_Y))
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    nb_epoch = 50

    Nb_filter1 = [256, 512, 1024]
    Filter_length1 = [2, 4]
    Batch_size = [20, 128]
    for nb_filter1 in Nb_filter1:
        for filter_length1 in Filter_length1:
            pool_length1 = maxlen - filter_length1 + 1
            for batch_size in Batch_size:
                print 'nb_filter1 = ', nb_filter1
                print 'filter_length1 = ', filter_length1
                print 'batch_size = ', batch_size

                global mod
                mod = CGNN(nb_filter1, filter_length1, pool_length1, C)

                checkpointer = ModelCheckpoint(filepath = "loss_weights.h5", verbose=1, save_best_only=True)

                mod.fit([X_train_1, X_train_2], dummy_y,
                        batch_size = batch_size, shuffle = True,
                        nb_epoch = nb_epoch, callbacks = [scorer(), checkpointer])

#testing
def test():
    sentence = data_proc(test_file)

    # constant
    Predicate_label = 10
    Father = 8
    Postag = 3  # confuse here

    global num_correct, num_predict_argument, num_require_argument
    num_correct = 0
    num_predict_predicate = 0
    num_predict_argument = 0
    num_require_predicate = 0
    num_require_argument = 0

    for i in range(len(sentence)):
        # clear global array
        global predicate, fa, edge
        predicate = {}
        fa = []
        edge = []
        for j in range(len(sentence[i]) + 1):
            edge.append([])
        #local var
        cnt = 0
        # deal
        for j in range(len(sentence[i])):
            fa.append(int(sentence[i][j][Father]) - 1)
            if fa[j] == -1:
                fa[j] = len(sentence[i])
            edge[fa[j]].append(j)
            if sentence[i][j][Predicate_label] != '_':
                predicate[j] = cnt
                cnt += 1
                num_require_predicate += 1
        for j in range(len(sentence[i])):
            if fa[j] != len(sentence[i]):
                for k in range(Predicate_label + 1, len(sentence[i][j])):
                    if sentence[i][j][k] != '_':
                        num_require_argument += 1
            if check_N_V(sentence[i][j][Postag]):
                result = test_result(sentence[i], len(sentence[i]), j)
                rest = 0
                if sentence[i][j][Predicate_label] != '_':
                    for k in range(len(sentence[i])):
                        if sentence[i][k][Predicate_label + 1 + predicate[j]] != '_':
                            rest += 1
                    count(sentence[i], j, j, rest)
                if result != '_':
                    num_predict_predicate += 1
                    if result == get_predicate_label(sentence[i][j][Predicate_label]):
                        num_correct += 1
                    test_traverse(sentence[i], j, j, rest)

    #print 'predict predicate', num_predict_predicate
    #print 'predict argument', num_predict_argument
    #print 'require predicate', num_require_predicate
    #print 'require argument', num_require_argument
    #print 'correct', num_correct

    num_predict = num_predict_predicate + num_predict_argument
    num_require = num_require_predicate + num_require_argument
    f = 2. * num_correct / (num_predict + num_require)
    #print f * 100
    acc = float(num_correct) / num_require
    return f, acc

class scorer(Callback):
    best_f = 0
    best_acc = 0
    epoch_num = 0
    global mod
    def on_epoch_end(self, epoch, logs = {}):
        f, acc = test()
        print('accuracy:', acc)
        print('f value:', f)
        if (f > scorer.best_f):
            scorer.best_f = f
            scorer.best_acc = acc
            scorer.epoch_num = epoch
            print('saving the best-model : ', 'epoch : ', epoch, 'acc : ', acc, 'f : ', f)
            mod.save_weights('best_model_weights.h5', overwrite = True)

    def on_train_end(self, logs = {}):
        print('the best model is:', 'epoch', scorer.epoch_num, 'best_acc', scorer.best_acc, 'best_f', scorer.best_f)

time1 = time.time()
WE = WE_proc()
time2 = time.time()

train()
time3 = time.time()

#test()
#time4 = time.time()

print 'WE_proc time', time2 - time1
print 'train time', time3 - time2
#print 'test time', time4 - time3