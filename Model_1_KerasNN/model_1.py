import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import plot_model
import keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

# Just disables the warning about AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_tr = numpy.loadtxt(
    './project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]


def euclidean_distance_loss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))


def train_and_learningcurve(x_tr, y_tr, x_ts, y_ts, eta=0.015, alpha=0.7, nEpoch=350, lambda_param=0.01, nUnitPerLayer=20,
                            nLayer=3,
                            batch_size=32):
    model = Sequential()
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer - 3*i, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
                        activation='relu'))
    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))
    sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss)
    history = model.fit(x_tr, y_tr, validation_data=(
        x_ts, y_ts), epochs=nEpoch, batch_size=batch_size, verbose=0)
    return history


def cross_validation1(eta, alpha, lambda_param, batch_size, nUnitLayer, nEpoc):
    fig, (plt1, plt2) = plt.subplots(2, 1)
    kfold = KFold(n_splits=10, random_state=None, shuffle=True)
    cvscores = []
    nFold = 0
    forLegend = []
    for train_index, test_index in kfold.split(X):
        x_tr = X[train_index]
        y_tr = Y[train_index]
        x_ts = X[test_index]
        y_ts = Y[test_index]
        history = train_and_learningcurve(
            x_tr, y_tr, x_ts, y_ts, eta, alpha, nEpoc, lambda_param, nUnitLayer, 3, batch_size)
        #score = [history.history['val_loss'][-1], history.history['val_mse'][-1], history.history['val_mae'][-1], history.history['val_coeff_determination'][-1]]
        cvscores.append([history.history['loss'][-1],
                         history.history['val_loss'][-1]])
        # Plot training loss values (just half of them)
        if nFold % 3 == 0:
            #plt.subplot(2, 1, 1)
            plt1.plot(history.history['loss'])
            plt1.plot(history.history['val_loss'])
            #plt.subplot(2, 1, 2)
            plt2.plot(range(25, nEpoc), history.history['loss'][25:])
            plt2.plot(range(25, nEpoc), history.history['val_loss'][25:])
            forLegend.append('Train ' + str(nFold))
            forLegend.append('Validation ' + str(nFold))
        nFold += 1
    averageLoss = 0
    averageLossTS = 0
    for score in cvscores:
        averageLoss += score[0]
        averageLossTS += score[1]
    averageLoss /= len(cvscores)
    averageLossTS /= len(cvscores)
    print(str(averageLoss)+'_________'+str(averageLossTS))
    fig.legend(forLegend, loc='center right')
    fig.suptitle('Model loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
        lambda_param) + '_' + str(batch_size))
    fig.savefig('./Keras_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
        lambda_param) + '_' + str(batch_size) + '_' + str(averageLossTS) + '_'+str(nUnitLayer)+'.png', dpi=600)
    plt.close()
    return averageLossTS


def best_model1(cross_validation):
    best_eta = 0.001
    best_alpha = 0.84
    best_lambda = 0.0006
    best_batch_size = 64
    best_nUnitLayer = 25
    nEpoch = 170
   
    if(cross_validation):
        min_loss = float('inf')
        nUnitLayers = [25]
        etas = [0.001]
        alphas = [0.84]
        lambdas = [0.0006]
        batch_sizes = [64]
        for nUnitLayer in nUnitLayers:
            for eta in etas:
                for alpha in alphas:
                    for _lambda in lambdas:
                        for batch_size in batch_sizes:
                            tmp = cross_validation1(
                                eta, alpha, _lambda, batch_size, nUnitLayer, nEpoch)
                            if(tmp < min_loss):
                                min_loss = tmp
                                best_alpha = alpha
                                best_batch_size = batch_size
                                best_lambda = _lambda
                                best_eta = eta
                                best_nUnitLayer = nUnitLayer

    return make_prediction1(best_eta, best_alpha, best_lambda, best_batch_size, best_nUnitLayer, nEpoch)


def make_prediction1(eta, alpha, lambda_param, batch_size, nUnitPerLayer, nEpoch, nLayer=3):
    model = Sequential()
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer - 3*i, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
                        activation='relu'))
    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))
    sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss)
    history = model.fit(X, Y, validation_split=0,
                        epochs=nEpoch, batch_size=batch_size, verbose=0)

    dataset_bs = numpy.genfromtxt(
        './project/ML-CUP19-TS.csv', delimiter=',', dtype=numpy.float64)
    data_test = numpy.genfromtxt(
        './project/ML-our_test_set.csv', delimiter=',', dtype=numpy.float64)
    X_test = data_test[:, 1:-2]
    Y_test = data_test[:, -2:]
    y_pred_final = model.predict(X_test)
    to_return = euclidean_distance_loss(Y_test, y_pred_final)

    return (model.predict(dataset_bs[:, 1:]), K.eval(to_return))
