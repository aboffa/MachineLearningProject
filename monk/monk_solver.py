import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
import matplotlib.pyplot as plt


dataset_train1 = loadtxt('./dataset/monks-1.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test1 = loadtxt('./dataset/monks-1.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train2 = loadtxt('./dataset/monks-2.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test2 = loadtxt('./dataset/monks-2.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train3 = loadtxt('./dataset/monks-3.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test3 = loadtxt('./dataset/monks-3.test',
                        delimiter=' ', usecols=range(1, 8))


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='acc', value=1.0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current >= self.value:
            self.model.stop_training = True
            print('Finded best accuracy!')


#early_stopping = EarlyStoppingByAccuracy(
#    monitor='acc', value=1.0, verbose=1)


def encoding(x):
    result = []
    if(x[0] == 1):
        result.append(0)
        result.append(0)
        result.append(1)
    if(x[0] == 2):
        result.append(0)
        result.append(1)
        result.append(0)
    if(x[0] == 3):
        result.append(1)
        result.append(0)
        result.append(0)
    if(x[1] == 1):
        result.append(0)
        result.append(0)
        result.append(1)
    if(x[1] == 2):
        result.append(0)
        result.append(1)
        result.append(0)
    if(x[1] == 3):
        result.append(1)
        result.append(0)
        result.append(0)
    if(x[2] == 1):
        result.append(0)
        result.append(1)
    if(x[2] == 2):
        result.append(1)
        result.append(0)
    if(x[3] == 1):
        result.append(0)
        result.append(0)
        result.append(1)
    if(x[3] == 2):
        result.append(0)
        result.append(1)
        result.append(0)
    if(x[3] == 3):
        result.append(1)
        result.append(0)
        result.append(0)
    if(x[4] == 1):
        result.append(0)
        result.append(0)
        result.append(0)
        result.append(1)
    if(x[4] == 2):
        result.append(0)
        result.append(0)
        result.append(1)
        result.append(0)
    if(x[4] == 3):
        result.append(0)
        result.append(1)
        result.append(0)
        result.append(0)
    if(x[4] == 4):
        result.append(1)
        result.append(0)
        result.append(0)
        result.append(0)
    if(x[5] == 1):
        result.append(0)
        result.append(1)
    if(x[5] == 2):
        result.append(1)
        result.append(0)
    return result


def monk_solve_plot(train, test, plotted, eta, alpha, batch_size, nUnit, nEpoch, lambda_param):
    x = train[:, 1:7]
    x_test = test[:, 1:7]
    new_x = []
    new_x_test = []
    for i in range(len(x)):
        new_x.append(encoding(x[i]))
    for i in range(len(x_test)):
        new_x_test.append(encoding(x_test[i]))
    x = numpy.array([numpy.array(xi) for xi in new_x])
    x_test = numpy.array([numpy.array(xi) for xi in new_x_test])
    y = train[:, 0]
    y_test = test[:, 0]

    model = Sequential()
    model.add(Dense(nUnit, input_dim=17, kernel_initializer="glorot_normal",
                    activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x, y, validation_data=(x_test, y_test), epochs=nEpoch, batch_size=batch_size, verbose=0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'], '--')
    plt.legend(['Loss on training set', 'Accuracy on training set'],
               loc='center right')
    plt.savefig('./' + str(plotted)+'_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(batch_size) +
                '_' + str(nUnit) + '_' + str(history.history['loss'][-1])+'_'+str(history.history['acc'][-1])+'.png', dpi=600)
    plt.close()
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_acc'], '--')
    plt.legend(['Loss on "test" set', 'Accuracy on "test" set'],
               loc='center right')
    plt.savefig('./' + str(plotted)+'_learning_curve_TEST_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(batch_size) +
                '_' + str(nUnit) + '_' + str(history.history['val_loss'][-1])+'_'+str(history.history['val_acc'][-1])+'.png', dpi=600)
    plt.close()


monk_solve_plot(dataset_train1, dataset_test1, 1, 0.25, 0.85, 25, 4, 90, 0)
monk_solve_plot(dataset_train2, dataset_test2, 2, 0.2, 0.75, 25, 4, 70, 0)
monk_solve_plot(dataset_train3, dataset_test3, 3, 0.2, 0.75, 25, 4, 120, 0)
monk_solve_plot(dataset_train3, dataset_test3, 3.1, 0.4, 0.75, 25, 4, 120, 0.0001)
