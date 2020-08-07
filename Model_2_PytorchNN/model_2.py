import torch
import numpy
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
import sys

# inizialization
dataset_tr = numpy.genfromtxt(
    './project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)
X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]
D_in = 20
D_out = 2
nFold = 0
splits_kfold = 10


class Model(torch.nn.Module):
    def __init__(self, D_in, nUnitLayer, D_out):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Linear(D_in, nUnitLayer)
        self.hidden3 = torch.nn.Linear(nUnitLayer, nUnitLayer)
        self.output = torch.nn.Linear(nUnitLayer, D_out)

    def forward(self, x):
        h_relu = F.relu(self.hidden1(x))
        h_relu3 = F.relu(self.hidden3(h_relu))
        y_pred = self.output(h_relu3)
        return y_pred


def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.requires_grad = True
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)


def loss_fn(y_real, y_pred): 
    return torch.div(
    torch.sum(F.pairwise_distance(y_real, y_pred, p=2)), len(y_real))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device.type == 'cuda'):
    print(torch.cuda.get_device_name(torch.cuda.current_device()),
          'enabled:', torch.backends.cudnn.enabled)


def cross_validation2(eta, alpha, lambda_param, batch_size, nUnitLayer,nEpoch):
                        fig, (plt1, plt2) = plt.subplots(2, 1)
                        nFold = 0
                        forLegend = []
                        kfold = KFold(n_splits=splits_kfold,
                                      random_state=None, shuffle=True)
                        print("Working on")
                        print("Eta: " + str(eta) + "  Alpha: " + str(alpha) + " nEpoch: " + str(nEpoch) + " Lambda: " + str(
                            lambda_param) + " nUnitPerLayer: " + str(nUnitLayer) + " Batch size: " + str(
                            batch_size))

                        print('Started at', datetime.now())
                        print('Executing cross-validation')

                        last_loss_tr = []
                        last_loss_ts = []
                        step = 1

                        for traing_index, test_index in kfold.split(X):
                            x_tr = X[traing_index]
                            y_tr = Y[traing_index]
                            x_ts = X[test_index]
                            y_ts = Y[test_index]
                            score_tr = []
                            score_ts = []
                            model = Model(D_in, nUnitLayer, D_out)
                            model.apply(init_weights)
                            model.cuda()
                            optimizer = optimizer = torch.optim.SGD(
                                model.parameters(), lr=eta, momentum=alpha, weight_decay=lambda_param)
                            loss = torch.zeros(1)
                            loss_ts = torch.zeros(1)
                            for epoch in range(nEpoch):
                                for i in range(int(len(x_tr) / batch_size)):
                                    optimizer.zero_grad()
                                    l = i * batch_size
                                    r = (i + 1) * batch_size
                                    # Slow TODO: work with tensors from the beginning
                                    x = torch.tensor(list(
                                        x_tr[l:r]), dtype=torch.float, requires_grad=True).cuda(device.type)
                                    y = torch.tensor(list(
                                        y_tr[l:r]), dtype=torch.float, requires_grad=True).cuda(device.type)

                                    y_pred = model(x)
                                    loss = loss_fn(y, y_pred)
                                    loss.backward()
                                    optimizer.step()
                                score_tr.append(loss.item())
                                y_pred_ts = model(torch.tensor(
                                    list(x_ts), dtype=torch.float, requires_grad=True).cuda(device.type))
                                loss_ts = loss_fn(torch.tensor(
                                    list(y_ts), dtype=torch.float, requires_grad=True).cuda(device.type), y_pred_ts)
                                score_ts.append(loss_ts.item())
                            last_loss_tr.append(loss.item())
                            last_loss_ts.append(loss_ts.item())
                            print('...Ended phase', nFold +
                                  1, 'of ', splits_kfold)
                            print(last_loss_tr[-1], '-', last_loss_ts[-1])
                            if nFold % 3 == 0:
                                plt1.plot(score_tr)
                                plt1.plot(score_ts)
                                plt2.plot(range(25, nEpoch), score_tr[25:])
                                plt2.plot(range(25, nEpoch), score_ts[25:])
                                forLegend.append('Train ' + str(nFold))
                                forLegend.append('Validation ' + str(nFold))
                            nFold += 1
                        averageLoss = 0
                        for cv_value in last_loss_tr:
                                averageLoss += cv_value
                        averageLoss /= len(last_loss_tr)
                        averageLossTs = 0
                        for cv_value2 in last_loss_ts:
                                averageLossTs += cv_value2
                        averageLossTs /= len(last_loss_ts)
                        print('Cross-Validation ended successfully!',
                              datetime.now())
                        print('Average TR:', averageLoss,
                              '- Average TS:', averageLossTs)

                        
                        print('Creating plot...')
                        fig.legend(forLegend, loc='center right')
                        fig.suptitle('Model loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                            lambda_param) + '_' + str(batch_size))
                        fig.savefig('./plots_final/Pytorch_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                            lambda_param) + '_' + str(batch_size) + '_' + str(nUnitLayer) +
                            '_' + str(
                            averageLossTs) + '.png', dpi=500)
                        plt.close()
                        print('Completed!')
                        return averageLossTs


def best_model2(cross_validation):

    best_eta = 0.0015
    best_alpha = 0.9
    best_lambda = 0.002
    best_batch_size = 64
    nUnitLayer = 40
    nEpoch=120
    if(cross_validation):
        min_loss=float('inf')
        nUnitLayers = [40]
        etas = [0.0015]
        alphas = [0.9]
        lambdas = [0.002]
        batch_sizes = [64]
        for nUnitLayer in nUnitLayers:
            for eta in etas:
                for alpha in alphas:
                    for _lambda in lambdas:
                        for batch_size in batch_sizes:
                                
                                tmp = cross_validation2(
                                    eta, alpha, _lambda, batch_size, nUnitLayer,nEpoch)
                                if(tmp < min_loss):
                                    min_loss = tmp
                                    best_alpha = alpha
                                    best_batch_size = batch_size
                                    best_lambda = _lambda
                                    best_eta = eta
                                    best_nUnitLayer = nUnitLayer
    return make_prediction2(best_eta, best_alpha, best_lambda, best_batch_size, nUnitLayer,nEpoch)



def make_prediction2(eta, alpha,  lambda_param, batch_size, _nUnits,nEpoch):
    model = Model(D_in, _nUnits, D_out)
    model.apply(init_weights)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=alpha, weight_decay=lambda_param)
    all_loss = []
    for epoch in range(nEpoch):
        loss = torch.zeros(1)
        for i in range(int(len(X) / batch_size)):
                    # TODO requires_grad=True ???
                    optimizer.zero_grad()
                    l = i * batch_size
                    r = (i + 1) * batch_size
                    x = torch.tensor(list(
                        X[l:r]), dtype=torch.float, requires_grad=True).cuda(device.type)
                    y = torch.tensor(list(
                        Y[l:r]), dtype=torch.float, requires_grad=True).cuda(device.type)
                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    loss.backward()
                    optimizer.step()
        all_loss.append(loss.item())
    plt.plot(all_loss) 
    plt.title("Final model2 Pytorch")
    plt.legend(["Learning curve"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('./plots_final/FinalPytorch_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                            lambda_param) + '_' + str(batch_size) + '_' + str(_nUnits) +
                            '_' + str(
                            all_loss[-1]) + '.png', dpi=500)   
    plt.close()                     
    dataset_bs = numpy.genfromtxt('./project/ML-CUP19-TS.csv', delimiter=',', dtype=numpy.float64)
    data_test = numpy.genfromtxt(
        './project/ML-our_test_set.csv', delimiter=',', dtype=numpy.float64)
    X_test = data_test[:, 1:-2]
    Y_test = torch.tensor(list(data_test[:, -2:]), dtype=torch.float, requires_grad=True).cuda(device.type)
    to_preditct = torch.tensor(list(dataset_bs[:,1:]), dtype=torch.float, requires_grad=True).cuda(device.type)
    to_preditct_test = torch.tensor(list(X_test), dtype=torch.float, requires_grad=True).cuda(device.type)
    y_predicted = model(to_preditct_test)
    return (model(to_preditct).cpu().detach().numpy(), loss_fn(y_predicted,Y_test).item())
