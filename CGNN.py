import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Masking, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Merge

embed_dim = 50
#embed_dim = 300

#train_file = 'train.csv'
train_file = 'CoNLL2009-ST-English-train.txt'
#test_file = 'train.csv'
#test_file = 'test.csv'
test_file = 'test.wsj.closed.GOLD'
embed_file = 'glove.6B.50d.txt'
#embed_file = 'glove.840B.300d.txt'

embeddings_index = {}
word_index = {}
postag_index = {}
deprel_index = {}

word_num = 0
postag_num = 0
deprel_num = 0

max_features = 92

X_train = []
Y_train = []

#page_num = 0
#page_cnt = 0
#page_lim = 128 * 100
#X_train_file = 'data/X_train_'
#Y_train_file = 'Y_train_'
#f_X = open(X_train_file + str(page_num) + '.txt', 'w')
#f_Y = open(Y_train_file + str(page_num) + '.txt', 'w')

#best_f = 0
#best_acc = 0
#epoch_num = 0

def output(tmp, label):
    global f_X#, f_Y
    mask(tmp)
    #print np.array(tmp).shape
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            f_X.write(str(tmp[i][j]))
            if i != len(tmp) - 1 or j != len(tmp[i]) - 1:
                f_X.write('\t')
    f_X.write('\n')
    #f_Y.write(label+'\n')

    global page_num, page_cnt, page_lim
    page_cnt += 1
    #print 'page_cnt', page_cnt
    if page_cnt == page_lim:
        #print 'page', page_num, 'done'
        f_X.close()
        #f_Y.close()
        page_cnt = 0
        page_num += 1
        f_X = open(X_train_file + str(page_num) + '.txt', 'w')
        #f_Y = open(Y_train_file + str(page_num) + '.txt', 'w')

def get_data(num):
    ret_X = []

    f = open(X_train_file + str(num) + '.txt')
    for line in f:
        line = line.strip('\n')
        tmp = line.split('\t')
        #print np.array(tmp).shape
        tmp = np.array(tmp).reshape((max_features, maxlen))
        ret_X.append(tmp.tolist())
    f.close()
    ret_X = np.array(ret_X)
    #print ret_X.shape

    global dummy_y
    ret_Y = dummy_y[num * page_lim : num * page_lim + ret_X.shape[0]]
    #ret_Y = np.array(ret_Y)
    #print ret_Y.shape

    return ret_X, ret_Y

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
        #length /= 10
        #print 'number of training sentence', length
        return sentence[ : length]
    else:
        length = len(sentence)
        #print 'number of testing sentence', length
        return sentence[ : length]

def read_pre_train(file):
    f = open(file)
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1 : ], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    embeddings_index[' '] = list(np.zeros(embed_dim))

def vocab_pos_deprel_proc(file):
    sentence = data_proc(file)

    #constant
    Word = 1
    Postag = 4
    Deprel = 10
    if file == test_file:
        Postag = 3
        Deprel = 9

    global word_num, postag_num, deprel_num

    for i in range(len(sentence)):
        for j in range(len(sentence[i])):
            #if not embeddings_index.has_key(sentence[i][j][Word].lower()):
                #embeddings_index[sentence[i][j][Word].lower()] = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (embed_dim,))
            if not word_index.has_key(sentence[i][j][Word].lower()):
                word_index[sentence[i][j][Word].lower()] = word_num
                word_num += 1
            if not postag_index.has_key(sentence[i][j][Postag]):
                postag_index[sentence[i][j][Postag]] = postag_num
                postag_num += 1
            if not deprel_index.has_key(sentence[i][j][Deprel]):
                deprel_index[sentence[i][j][Deprel]] = deprel_num
                deprel_num += 1

    word_index[' '] = word_num
    word_num += 1
    postag_index[' '] = postag_num
    postag_num += 1

def prepare():
    read_pre_train(embed_file)
    vocab_pos_deprel_proc(train_file)
    vocab_pos_deprel_proc(test_file)

    global word_empty, postag_empty, deprel_empty
    word_empty = np.zeros(embed_dim)
    postag_empty = np.zeros(postag_num)
    deprel_empty = np.zeros(deprel_num)

    global maxlen
    maxlen = embed_dim + postag_num + deprel_num
    print 'maxlen', maxlen

    global WE_Word
    WE_Word = np.zeros((word_num + 2, embed_dim))
    for x, y in word_index.items():
        if embeddings_index.has_key(x):
            WE_Word[y][:] = embeddings_index[x]
        else:
            WE_Word[y][:] = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (embed_dim,))
    WE_Word[word_num + 1][:] = (-1.) * np.ones(embed_dim)

    global WE_Postag
    WE_Postag = np.zeros((postag_num + 2, postag_num))
    for x, y in postag_index.items():
        WE_Postag[y][y] = 1
    WE_Postag[postag_num + 1][:] = (-1.) * np.ones(postag_num)
    #print postag_num
    #print WE_Postag

    global WE_Deprel
    WE_Deprel = np.zeros((deprel_num + 2, deprel_num))
    for x, y in deprel_index.items():
        WE_Deprel[y][y] = 1
    WE_Deprel[deprel_num + 1][:] = (-1.) * np.ones(deprel_num)
    #print deprel_num
    #print WE_Deprel

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

def word_proc(now):
    ret = []

    #ret.extend(embeddings_index[now])
    #ret.extend(postag_empty)
    #ret.extend(deprel_empty)

    ret.append(word_index[now])
    ret.append(postag_num)
    ret.append(deprel_num)

    return ret

def postag_proc(now):
    ret = []

    #ret.extend(word_empty)
    #ret.extend(np_utils.to_categorical([postag_index[now]], len(postag_index)).reshape(-1).tolist())
    #ret.extend(deprel_empty)

    ret.append(word_num)
    ret.append(postag_index[now])
    ret.append(deprel_num)

    return ret


def deprel_proc(now):
    ret = []

    #ret.extend(word_empty)
    #ret.extend(postag_empty)
    #ret.extend(np_utils.to_categorical([deprel_index[now]], len(deprel_index)).reshape(-1).tolist())

    ret.append(word_num)
    ret.append(postag_num)
    ret.append(deprel_index[now])

    return ret

def out(x, y):
    tmp = []

    for i in range(len(y)):
        tmp.append(postag_proc(y[i][0]))
        tmp.append(word_proc(y[i][1]))
        tmp.append(deprel_proc(y[i][-1]))

    for i in reversed(range(len(x))):
        if i != len(x) - 1:
            tmp.append(deprel_proc(x[i][-1]))
        tmp.append(postag_proc(x[i][0]))
        tmp.append(word_proc(x[i][1]))

    #global max_features
    #max_features = max(max_features, len(tmp))

    return tmp

def mask(now):
    #minus = (-1.) * np.ones(maxlen)
    minus = [word_num + 1, postag_num + 1, deprel_num + 1]
    tmp = max_features - len(now)
    for j in range(tmp):
        now.append(minus)

#train
def train_argument(x, y, label):
    tmp = out(x, y)
    #output(tmp, label)
    X_train.append(tmp)
    Y_train.append(label)

#test
def test_argument(x, y):
    X_test = []
    X_test.append(out(x, y))
    mask(X_test[0])
    X_test = np.array(X_test)
    X_test_Word = X_test[:, :, 0]
    X_test_Postag = X_test[:, :, 1]
    X_test_Deprel = X_test[:, :, 2]

    global mod
    Y_test = mod.predict([X_test_Word, X_test_Postag, X_test_Deprel])
    result = encoder.inverse_transform(np.argmax(Y_test))
    return result

def train_predicate(now, x, label):
    #constant
    Word = 1
    Postag = 4
    Deprel = 10

    tmp = []

    while x != len(now):
        tmp.append(postag_proc(now[x][Postag]))
        tmp.append(word_proc(now[x][Word].lower()))
        tmp.append(deprel_proc(now[x][Deprel]))
        x = fa[x]

    tmp.append(postag_proc(' '))
    tmp.append(word_proc(' '))

    #global max_features
    #max_features = max(max_features, len(tmp))

    #output(tmp, label)
    X_train.append(tmp)
    Y_train.append(label)

def test_predicate(now, x):
    # constant
    Word = 1
    #Postag = 4
    Postag = 3
    #Deprel = 10
    Deprel = 9

    tmp = []

    while x != len(now):
        tmp.append(postag_proc(now[x][Postag]))
        tmp.append(word_proc(now[x][Word].lower()))
        tmp.append(deprel_proc(now[x][Deprel]))
        x = fa[x]

    tmp.append(postag_proc(' '))
    tmp.append(word_proc(' '))

    X_test = []
    X_test.append(tmp)
    mask(X_test[0])
    X_test = np.array(X_test)
    X_test_Word = X_test[:, :, 0]
    X_test_Postag = X_test[:, :, 1]
    X_test_Deprel = X_test[:, :, 2]

    global mod
    Y_test = mod.predict([X_test_Word, X_test_Postag, X_test_Deprel])
    result = encoder.inverse_transform(np.argmax(Y_test))
    return result

def test_traverse(now, x, z, rest, path):
    # constant
    Word = 1
    #Predicate_label = 13
    Predicate_label = 10
    #Postag = 4
    Postag = 3
    #Deprel = 10
    Deprel = 9

    if x != len(now):
        path.append([now[x][Postag], now[x][Word].lower()])

    global num_predict_argument, num_correct
    for y in edge[x]:
        result = test_argument(path, [[now[y][Postag], now[y][Word].lower(), now[y][Deprel]]])
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
        path[-1].append(now[x][Deprel])
        test_traverse(now, fa[x], z, rest, path)

def count(now, x, z, rest):
    #constant
    #Predicate_label = 13
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

#testing
def test():
    sentence = data_proc(test_file)

    # constant
    Father = 8
    #Predicate_label = 13
    Predicate_label = 10
    #Postag = 4
    Postag = 3

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
                result = test_predicate(sentence[i], j)
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
                    test_traverse(sentence[i], j, j, rest, [])

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

def train_traverse(now, x, z, rest, path):
    #constant
    Word = 1
    Predicate_label = 13
    Postag = 4
    Deprel = 10

    if x != len(now):
        path.append([now[x][Postag], now[x][Word].lower()])

    for y in edge[x]:
        if rest <= 0:
            #print sentence[i][z][Word], sentence[i][y][Word], 'NoMoreArg'
            train_argument(path, [[now[y][Postag], now[y][Word].lower(), now[y][Deprel]]], 'NoMoreArg')
            return
        else:
            #print sentence[i][z][Word], sentence[i][y][Word], sentence[i][y][Predicate_label + 1 + predicate[z]]
            train_argument(path, [[now[y][Postag], now[y][Word].lower(), now[y][Deprel]]],
                         now[y][Predicate_label + 1 + predicate[z]])
        if now[y][Predicate_label + 1 + predicate[z]] != '_':
            rest -= 1

    if (x != len(now)):
        path[-1].append(now[x][Deprel])
        train_traverse(now, fa[x], z, rest, path)

#training
def train():
    sentence = data_proc(train_file)
    print 'number of sentence', len(sentence)

    #constant
    Offset = 0  # 0 for goleden and 1 for predict
    Predicate_label = 13
    Father = 8 + Offset
    Postag = 4 + Offset

    for i in range(len(sentence)):
        #print 'deal with sentence', i, len(sentence[i])
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
                ##print sentence[i][j][1], get_label(sentence[i][j][Predicate_label])
                train_predicate(sentence[i], j, get_predicate_label(sentence[i][j][Predicate_label]))
            if sentence[i][j][Predicate_label] != '_':
                rest = 0
                for k in range(len(sentence[i])):
                    if sentence[i][k][Predicate_label + 1 + predicate[j]] != '_':
                        rest += 1
                train_traverse(sentence[i], j, j, rest, [])

    #global f_X#, f_Y
    #f_X.close()
    #f_Y.close()

    global X_train#, Y_train
    for i in range(len(X_train)):
        mask(X_train[i])
    X_train = np.array(X_train)
    print X_train.shape
    X_train_Word = X_train[:, :, 0]
    X_train_Postag = X_train[:, :, 1]
    X_train_Deprel = X_train[:, :, 2]
    #print X_train_Word.shape
    #print X_train_Postag.shape
    #print X_train_Deprel.shape
    #print max_features

    # encode class values as integers
    global encoder
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    global dummy_y
    dummy_y = np_utils.to_categorical(encoded_Y)

    Filter_num = [512]
    Filter_length = [16]
    AddSize = [1024]
    Pool_size = [2, 4, 8, 16, 32]
    class_num = np.size(np.unique(encoded_Y))

    for filter_num in Filter_num:
        for filter_length in Filter_length:
            for addSize in AddSize:
				for pool_size in Pool_size:
                    global mod
                    mod = model(filter_num, filter_length, class_num, addSize, pool_size)

                    batch_size = 128
                    nb_epoch = 15

                    checkpointer = ModelCheckpoint(filepath="loss_weights.h5", verbose=1, save_best_only=True)

                    mod.fit([X_train_Word, X_train_Postag, X_train_Deprel], dummy_y, batch_size = batch_size, shuffle = True, nb_epoch = nb_epoch, callbacks = [scorer(), checkpointer])


                    print 'filter_num', filter_num
                    print 'filter_length', filter_length
                    print 'addSize', addSize
                    print 'epoch', nb_epoch
                    print 'batch size', batch_size

                '''global best_f, best_acc, epoch_num
                best_f = 0
                best_acc = 0
                epoch_num = 0

                for i in range(nb_epoch):
                    for j in range(page_num + 1):
                        X_train_on_batch, Y_train_on_batch = get_data(page_num)
                        #mod.train_on_batch(X_train_on_batch, Y_train_on_batch)
                        mod.fit(X_train_on_batch, Y_train_on_batch,
                                batch_size = batch_size,
                                nb_epoch = 1,
                                #callbacks=[scorer(), checkpointer],
                                shuffle=True)
                    on_epoch_end(i)

                on_train_end()'''

def model(filter_num, filter_length, class_num, addSize, pool_size):
    '''model = Sequential()

    #model.add(LSTM(256, input_shape = (max_features, maxlen, )))
    model.add(Convolution1D(input_shape = (max_features, maxlen),
                            nb_filter = filter_num,
                            filter_length = filter_length,
                            border_mode = 'valid',
                            activation = 'tanh',
                            subsample_length = 1))
    model.add(MaxPooling1D(pool_length = max_features - filter_length + 1))
    model.add(LSTM(addSize))
    model.add(Dense(output_dim = class_num,
                    activation = 'softmax'))'''

    input_Word = Sequential()
    input_Word.add(Embedding(input_dim = word_num + 2, input_length = max_features, weights = [WE_Word], output_dim = embed_dim))

    input_Postag = Sequential()
    input_Postag.add(Embedding(input_dim = postag_num + 2, input_length = max_features, weights = [WE_Postag], output_dim = postag_num))

    input_Deprel = Sequential()
    input_Deprel.add(Embedding(input_dim = deprel_num + 2, input_length = max_features, weights = [WE_Deprel], output_dim = deprel_num))

    merge = Merge([input_Word, input_Postag, input_Deprel], mode='concat')

    model = Sequential()
    model.add(merge)
    model.add(Convolution1D(nb_filter=filter_num,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='tanh',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=3))#max_features - filter_length - pool_size))
    model.add(LSTM(addSize))
    model.add(Dense(output_dim=class_num,
                    activation='softmax'))

    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam')

    model.summary()

    return model

def on_epoch_end(epoch):
    global mod
    f, acc = test()
    print('at epoch:', epoch)
    print('accuracy:', acc)
    print('f value:', f)
    if (f > scorer.best_f):
        scorer.best_f = f
        scorer.best_acc = acc
        scorer.epoch_num = epoch
        print('saving the best-model : ', 'epoch : ', epoch, 'acc : ', acc, 'f : ', f)
        mod.save_weights('best_model_weights.h5', overwrite = True)

def on_train_end():
    print('the best model is:', 'epoch', scorer.epoch_num, 'best_acc', scorer.best_acc, 'best_f', scorer.best_f)

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

prepare()
print 'word done'
train()
