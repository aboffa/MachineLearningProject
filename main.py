from Model_2_PytorchNN.model_2 import best_model2
from Model_1_KerasNN.model_1 import best_model1
from Model_3_ScikitSVM.model_3 import best_model3
import matplotlib.pyplot as plt
import numpy
import time
import pandas as pd


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def numbers_to_Models(argument): 
    switcher = { 
        1: "Model Keras", 
        2: "Model Pytorch", 
        3: "Model SVR", 
    } 
    return switcher.get(argument, "nothing") 



def create_plot(y_pred,model_index):
    s = 50
    a = 0.4
    plt.figure()
    plt.scatter(y_pred[:,0],y_pred[:,1], c="cornflowerblue", s=s, alpha=a)
    plt.xlabel("Target 1")
    plt.ylabel("Target 2")
    plt.title(numbers_to_Models(model_index))
    labels = ['Data point predicted by model: '+str(model_index)]
    plt.legend(labels)
    if(model_index==1): plt.savefig('./Keras_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    if(model_index==2):
        plt.savefig('./Pytorch_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    if(model_index==3): plt.savefig('./SVM_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    plt.close()
loss_final = []

print('Using Neural Network in Keras...')
start = time.time()
returnBestModel1 = best_model1(False)
loss_final.append(returnBestModel1[1])
print("TEST ERROR 1 = ",loss_final[len(loss_final)-1])
create_plot(returnBestModel1[0],1)  
end = time.time()
print('Prediction done!')  
print('Ended in: ',end-start,'seconds') 


print('Using Neural Network in Pytorch...')
start = time.time()
returnBestModel2 = best_model2(False)
loss_final.append(returnBestModel2[1])
print("TEST ERROR 2 = ",loss_final[len(loss_final)-1])
create_plot(returnBestModel2[0],2)  
end = time.time()
print('Prediction done!')  


print('Ended in: ',end-start,'seconds') 
print('Using Support Vectors Machine..') 
start = time.time()
returnBestModel3 = best_model3(False)
loss_final.append(returnBestModel3[1])
print("TEST ERROR 3 = ",loss_final[len(loss_final)-1])
create_plot(returnBestModel3[0],3)  
end = time.time()
print('Prediction done!')  
print('Ended in: ',end-start,'seconds') 


print('Losses: ',loss_final)
print(numbers_to_Models(loss_final.index(min(loss_final))))
print('Best model is: ',numbers_to_Models(loss_final.index(min(loss_final))+1))

best_model = loss_final.index(min(loss_final))

final_name="./BOFMON_ML-CUP19-TS.csv"
if(best_model == 0):
    df = pd.DataFrame(returnBestModel1[0])
    df.index = numpy.arange(1, len(df)+1)
    df.to_csv(final_name,index=True,header=False)
if(best_model==1): 
    df = pd.DataFrame(returnBestModel2[0])
    df.index = numpy.arange(1, len(df)+1)
    df.to_csv(final_name,index=True,header=False)
if(best_model==2): 
    df = pd.DataFrame(returnBestModel3[0])
    df.index = numpy.arange(1, len(df)+1)
    df.to_csv(final_name,index=True,header=False)

line_prepender(final_name,"# Antonio Boffa   Davide Montagno Bozzone\n# Group BOFMON\n# ML-CUP19\n# 30/11/2019")