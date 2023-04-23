import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np
from sklearn.model_selection import train_test_split
import json



# training set and normalizing
train_data = torchvision.datasets.FashionMNIST("./data", train = True, download = True,
  transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)]))
# test set and normalizing
test_data = torchvision.datasets.FashionMNIST("./data", train = False,
  transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)]))

#split between training set and validation set
train_set, valid_set = train_test_split(train_data, test_size=0.1, stratify = train_data.targets)

#added from HW manual
train_generator = torch.utils.data.DataLoader(train_set, batch_size = 50, shuffle = True) #shuffle flag is True as desired in the manual
test_generator = torch.utils.data.DataLoader(test_data, batch_size = 50, shuffle = False)
valid_generator = torch.utils.data.DataLoader(valid_set, batch_size = 50, shuffle = True)

#Class definitions
# example mlp classifier, taken from homework manual
class model_mlp1(torch.nn.Module):
  def __init__(self):
    super(model_mlp1, self).__init__()
    self.fc1 = torch.nn.Linear(784, 64)
    self.relu = torch.nn.ReLU()
    self.fc10 = torch.nn.Linear(64, 10)    
  def forward(self, x):
    x = x.view(-1, 784)
    hidden = self.fc1(x)
    relu = self.relu(hidden)
    output = self.fc10(relu)
    return output 
#https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
#Sigmoid is taken from here
class model_mlp1_sigmoid(torch.nn.Module):
  def __init__(self):
    super(model_mlp1_sigmoid, self).__init__()
    self.fc1 = torch.nn.Linear(784, 64)
    self.sigmoid = torch.nn.Sigmoid()
    self.fc10 = torch.nn.Linear(64, 10)    
  def forward(self, x):
    x = x.view(-1, 784)
    hidden = self.fc1(x)
    sigmoid = self.sigmoid(hidden)
    output = self.fc10(sigmoid)
    return output       

class model_mlp2(torch.nn.Module):
  def __init__(self):
    super(model_mlp2, self).__init__()
    self.fc1 = torch.nn.Linear(784, 16)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(16, 64)
    self.fc10 = torch.nn.Linear(64, 10)
  def forward(self, x):
    x = x.view(-1, 784)
    hidden = self.fc1(x)
    relu = self.relu(hidden)
    hidden2 = self.fc2(relu)  
    output = self.fc10(hidden2)
    return output  

class model_mlp2_sigmoid(torch.nn.Module):
  def __init__(self):
    super(model_mlp2_sigmoid, self).__init__()
    self.fc1 = torch.nn.Linear(784, 16)
    self.sigmoid = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(16, 64)
    self.fc10 = torch.nn.Linear(64, 10)
  def forward(self, x):
    x = x.view(-1, 784)
    hidden = self.fc1(x)
    sigmoid = self.sigmoid(hidden)
    hidden2 = self.fc2(sigmoid)  
    output = self.fc10(hidden2)
    return output  

class model_cnn3_sigmoid(torch.nn.Module):   
    def __init__(self):
        super(model_cnn3_sigmoid, self).__init__()
        #Convolution layer definition is taken from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 7, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1, padding = 'valid')
        self.sigmoid = torch.nn.Sigmoid()
        #Pooling layer definition is taken from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        sigmoid1_output = self.sigmoid(fc1_output)
        fc2_output = self.fc2(sigmoid1_output)
        sigmoid2_output = self.sigmoid(fc2_output)
        mp1_output = self.mp2(sigmoid2_output)
        fc3_output = self.fc3(mp1_output)
        mp2_output = self.mp2(fc3_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output

class model_cnn3(torch.nn.Module):   
    def __init__(self):
        super(model_cnn3, self).__init__()
        #Convolution layer definition is taken from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 7, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1, padding = 'valid')
        self.relu = torch.nn.ReLU()
        #Pooling layer definition is taken from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        relu1_output = self.relu(fc1_output)
        fc2_output = self.fc2(relu1_output)
        relu2_output = self.relu(fc2_output)
        mp1_output = self.mp2(relu2_output)
        fc3_output = self.fc3(mp1_output)
        mp2_output = self.mp2(fc3_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output

class model_cnn4(torch.nn.Module):   
    def __init__(self):
        super(model_cnn4, self).__init__()
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 5, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc4 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1, padding = 'valid')
        self.relu = torch.nn.ReLU()
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        relu1_output = self.relu(fc1_output)
        fc2_output = self.fc2(relu1_output)
        relu2_output = self.relu(fc2_output)        
        fc3_output = self.fc3(relu2_output)
        relu3_output = self.relu(fc3_output)
        mp1_output = self.mp2(relu3_output)
        fc4_output = self.fc4(mp1_output)
        relu4_output = self.relu(fc4_output)
        mp2_output = self.mp2(relu4_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output  

class model_cnn4_sigmoid(torch.nn.Module):   
    def __init__(self):
        super(model_cnn4_sigmoid, self).__init__()
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 5, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc4 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1, padding = 'valid')
        self.sigmoid = torch.nn.Sigmoid()
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        sigmoid1_output = self.sigmoid(fc1_output)
        fc2_output = self.fc2(sigmoid1_output)
        sigmoid2_output = self.sigmoid(fc2_output)        
        fc3_output = self.fc3(sigmoid2_output)
        sigmoid3_output = self.sigmoid(fc3_output)
        mp1_output = self.mp2(sigmoid3_output)
        fc4_output = self.fc4(mp1_output)
        sigmoid4_output = self.sigmoid(fc4_output)
        mp2_output = self.mp2(sigmoid4_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output  


class model_cnn5(torch.nn.Module):
    def __init__(self):
        super(model_cnn5, self).__init__()
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc4 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc5 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc6 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.relu = torch.nn.ReLU()
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        relu1_output = self.relu(fc1_output)
        fc2_output = self.fc2(relu1_output)
        relu2_output = self.relu(fc2_output)
        fc3_output = self.fc3(relu2_output)
        relu3_output = self.relu(fc3_output)
        fc4_output = self.fc4(relu3_output)
        relu4_output = self.relu(fc4_output)
        mp1_output = self.mp2(relu4_output)
        fc5_output = self.fc5(mp1_output)
        relu5_output = self.relu(fc5_output)
        fc6_output = self.fc6(relu5_output)
        relu6_output = self.relu(fc6_output)
        mp2_output = self.mp2(relu6_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output     

class model_cnn5_sigmoid(torch.nn.Module):
    def __init__(self):
        super(model_cnn5_sigmoid, self).__init__()
        self.fc1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc2 = torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc3 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc4 = torch.nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc5 = torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.fc6 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 'valid')
        self.sigmoid = torch.nn.Sigmoid()
        self.mp2 = torch.nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc10 = torch.nn.Linear(144,10)
    def forward(self, x):
        fc1_output = self.fc1(x)
        sigmoid1_output = self.sigmoid(fc1_output)
        fc2_output = self.fc2(sigmoid1_output)
        sigmoid2_output = self.sigmoid(fc2_output)
        fc3_output = self.fc3(sigmoid2_output)
        sigmoid3_output = self.sigmoid(fc3_output)
        fc4_output = self.fc4(sigmoid3_output)
        sigmoid4_output = self.sigmoid(fc4_output)
        mp1_output = self.mp2(sigmoid4_output)
        fc5_output = self.fc5(mp1_output)
        sigmoid5_output = self.sigmoid(fc5_output)
        fc6_output = self.fc6(sigmoid5_output)
        sigmoid6_output = self.sigmoid(fc6_output)
        mp2_output = self.mp2(sigmoid6_output).view(50,144)
        fc10_output = self.fc10(mp2_output)
        return fc10_output        
#Class definitions

#https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Change of device is taken from here to compute trainings faster

#initializing the models
mlp1 = model_mlp1().to(device)
mlp2 = model_mlp2().to(device)
cnn3 = model_cnn3().to(device)
cnn4 = model_cnn4().to(device)
cnn4_lr01 = model_cnn4().to(device)
cnn4_lr001 = model_cnn4().to(device)
cnn4_lr0001 = model_cnn4().to(device)
cnn5 = model_cnn5().to(device)

mlp1_sigmoid = model_mlp1_sigmoid().to(device)
mlp2_sigmoid = model_mlp2_sigmoid().to(device)
cnn3_sigmoid = model_cnn3_sigmoid().to(device)
cnn4_sigmoid = model_cnn4_sigmoid().to(device)
cnn5_sigmoid = model_cnn5_sigmoid().to(device)
#initializing the models

#Training function for the q2
def train(model,model_name):   
    criterion = torch.nn.CrossEntropyLoss()
    training_loss_overall = []
    training_accuracy_overall = []
    validation_accuracy_overall = []
    max_weight = 0
    max_accuracy = 0
    print('TrainingStarts')
    for steps in range (0,1):
      print('step++')
      optimizer = torch.optim.Adam(model.parameters())
      model_training_loss = []
      model_training_accuracy = []
      validation_accuracy_list = []
      epochs = 15
#https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4 
#The website above is used for the training, validation and test steps
#Some parts of it are directly taken.
      for epoch in range (0,epochs):
        print('epoch++')          
        model.train()
        for train_batch_num, (images, labels) in enumerate(train_generator):
          x = Variable(images).to(device)
          y = Variable(labels).to(device)
          optimizer.zero_grad()
          outputs = model(x) 
          loss = criterion(outputs, y)
            
          loss.backward()
          optimizer.step()
          
          #Evaluating the model in each 10 steps
          if train_batch_num % 10 == 0:
            model.eval()
            prediction = outputs.data.max(dim=1)[1]
            model_training_accuracy.append((((prediction.data == y.data).float().mean()).item())*100)
            model_training_loss.append(loss.item())                                     
            correct = 0
            sample = 0
        
            for validation_batch_num, (images, labels) in enumerate(valid_generator):
              x=Variable(images).to(device)
              y=Variable(labels).to(device)
              score = model(x)

              _,prediction = score.max(1)
              correct += (prediction.data == y.data).sum()
              sample += prediction.size(0)

            validation_accuracy = float(correct) / float(sample) * 100            
            validation_accuracy_list.append(validation_accuracy)
            model.train()

      #Appending the data with relative lists. 
      training_loss_overall.append(model_training_loss)
      training_accuracy_overall.append(model_training_accuracy)
      validation_accuracy_overall.append(validation_accuracy_list)

      model.eval()

      true = 0
      sample = 0

      for test_batch_num, (images, labels) in enumerate(test_generator):
        x=Variable(images).to(device)
        y=Variable(labels).to(device)
        score = model(x)

        _,prediction = score.max(1)
        true += (prediction.data == y.data).sum()
        sample += prediction.size(0)
              
        accuracy = float(true) / float(sample) * 100

        if accuracy >= max_accuracy:
          max_accuracy = accuracy
          #Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
          #I took the error above, hence, I switched between cpu and device when I get the error in the following of the code.
          model.to('cpu')
          #https://discuss.pytorch.org/t/access-weights-of-a-specific-module-in-nn-sequential/3627 taken from here as the first layer of the models are fc1
          max_weight = model.fc1.weight.data.numpy()
          model.to(device)

      #Create dictionary (Created by observing utils.py)
      dictionary = {"name": model_name,
                    "loss_curve":  np.mean(training_loss_overall, axis=0).tolist(),
                    "train_acc_curve": np.mean(training_accuracy_overall, axis=0).tolist(),
                  "val_acc_curve": np.mean(validation_accuracy_overall, axis=0).tolist(),
                  "test_acc": max_accuracy,
                  "weights":max_weight.tolist()}

    #Write on json file
    #https://pythonexamples.org/python-write-json-to-file/          
      with open('C:/Users/Can/Desktop/EE449_HW1'+'part2_'+  model_name +'.json', 'w') as json_file:
          json.dump(dictionary, json_file)
          json_file.close()
        
             
    print('TrainingEnds')
    return     
#Training function for the q2

train(mlp1,'mlp1')
train(mlp2,'mlp2')
train(cnn3,'cnn3')
train(cnn4,'cnn4')
train(cnn5,'cnn5')

#Training function for the q3
def train_q3(model,model_name, model_type):
    criterion = torch.nn.CrossEntropyLoss()
    #Definitions are taken from homework manual
    relu_loss = []
    sigmoid_loss = []
    relu_grad = []
    sigmoid_grad = []     

    print('TrainingStarts')
    for steps in range (0,1):
      print('step++')
      lr = 0.01
      optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)
      epochs = 15

      for epoch in range (0,epochs):
        print('epoch++')          
        model.train()
        for train_batch_num, (images, labels) in enumerate(train_generator):
          x = Variable(images).to(device)
          y = Variable(labels).to(device)
          outputs = model(x) 
          loss = criterion(outputs, y)
          model.to('cpu')
          first_layer_first_weight = model.fc1.weight.data.numpy()
          model.to(device)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
                  
          if train_batch_num % 10 == 0:
            model.eval()
            model.to('cpu')            
            first_layer_last_weight = model.fc1.weight.data.numpy()
            model.to(device)
            #https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
            #grad to append calculation is taken from this site.
            weight_difference = first_layer_last_weight - first_layer_first_weight
            grad = weight_difference / lr
            grad_to_append = np.linalg.norm(grad).tolist()
            
            #According to the parameter of the function
            #Data is added to the related list 
            if model_type == 'sigmoid':
                sigmoid_loss.append(loss.item())
                sigmoid_grad.append(grad_to_append)
            if model_type == 'relu':
                relu_loss.append(loss.item())
                relu_grad.append(grad_to_append)
            model.train()

      #Create dictionary (Created by observing utils.py)
      dictionary = {"name": model_name,
                    "relu_loss_curve":  relu_loss,
                    "sigmoid_loss_curve": sigmoid_loss,
                    "relu_grad_curve": relu_grad,
                    "sigmoid_grad_curve": sigmoid_grad}

    #Write on json file
    #https://pythonexamples.org/python-write-json-to-file/          
      with open('C:/Users/Can/Desktop/EE449_HW1'+'part3_'+  model_name +'.json', 'w') as json_file:
          json.dump(dictionary, json_file)
          json_file.close()
        
             
    print('TrainingEnds')
    return

#json files are combined manually

train_q3(mlp1,'mlp1','relu')
train_q3(mlp1_sigmoid,'mlp1_sigmoid','sigmoid')
train_q3(mlp2,'mlp2','relu')
train_q3(mlp2_sigmoid,'mlp2_sigmoid','sigmoid')
train_q3(cnn3,'cnn3','relu')
train_q3(cnn3_sigmoid,'cnn3_sigmoid','sigmoid')
train_q3(cnn4,'cnn4','relu')
train_q3(cnn4_sigmoid,'cnn4_sigmoid','sigmoid')
train_q3(cnn5,'cnn5','relu')
train_q3(cnn5_sigmoid,'cnn5_sigmoid','sigmoid')
#json files are combined manually
#Training function for the q3    

#Training function for the q4_part1
def train_q4_part1(model,model_name,inp_lr): 
    loss_curve_1 = []
    loss_curve_01 = []
    loss_curve_001 = []
    val_acc_curve_1 = []
    val_acc_curve_01 = []
    val_acc_curve_001 = []
    #I tried without inp_lr parameter. However, I could not reset the model. 
    #Hence, I decided to took it as a parameter
    learning_rates = [inp_lr]

    #For loop is because of the first implementation attempt
    for lr in learning_rates: 
        criterion = torch.nn.CrossEntropyLoss()
        #Data observation
        print('TrainingStarts\n')        
        print(lr)
        #Data observation

        model_training_loss = []
        model_training_accuracy = []
        validation_accuracy_list = []
        #Defined epochs in order to test the function first.
        #e.g I gave epochs = 3 and observed the behavior.
        epochs = 20  
        for epoch in range (0,epochs):
          print('epoch++')          
          model.train()
          #Adjusting the optimizer with respect to lr
          optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)
          print(lr)

          for train_batch_num, (images, labels) in enumerate(train_generator):
            x = Variable(images).to(device)
            y = Variable(labels).to(device)
                
            optimizer.zero_grad()
            outputs = model(x) 
            loss = criterion(outputs, y)            
            loss.backward()
            optimizer.step()


            if train_batch_num % 10 == 0:
              model.eval()
              prediction = outputs.data.max(dim=1)[1]
              model_training_accuracy.append((((prediction.data == y.data).float().mean()).item())*100)
              model_training_loss.append(loss.item())                                           
              correct = 0
              sample = 0
          
              for validation_batch_num, (images, labels) in enumerate(valid_generator):
                x=Variable(images).to(device)
                y=Variable(labels).to(device)
                score = model(x)
                _,prediction = score.max(1)
                correct += (prediction.data == y.data).sum()
                sample += prediction.size(0)
  
              validation_accuracy = float(correct) / float(sample) * 100            
              validation_accuracy_list.append(validation_accuracy)
              model.train()

          #Appending data with respect to learning rate of the optimizer
          if lr == 0.1:
                loss_curve_1 = model_training_loss
                val_acc_curve_1 = validation_accuracy_list
          if lr == 0.01:
                loss_curve_01 = model_training_loss
                val_acc_curve_01 = validation_accuracy_list              
          if lr == 0.001:
                loss_curve_001 = model_training_loss
                val_acc_curve_001 = validation_accuracy_list                    
  
  

    #Create dictionary (Created by observing utils.py)
    dictionary = {"name": model_name,
                  "loss_curve_1":  loss_curve_1,
                  "loss_curve_01":  loss_curve_01,
                "loss_curve_001":  loss_curve_001,
                "val_acc_curve_1": val_acc_curve_1,
                "val_acc_curve_01": val_acc_curve_01,
                "val_acc_curve_001": val_acc_curve_001}
  
    #Write on json file
    #https://pythonexamples.org/python-write-json-to-file/     
    with open('C:/Users/Can/Desktop/EE449_HW1'+'part4_2LR_'+  model_name + 'lr = {}'.format(lr) + '.json', 'w') as json_file:
        json.dump(dictionary, json_file)
        json_file.close()
             
    print('TrainingEnds')
    return    
 
train_q4_part1(cnn4_lr01,'cnn4_lr01',0.1)
train_q4_part1(cnn4_lr001,'cnn4_lr001',0.01)
train_q4_part1(cnn4_lr0001,'cnn4_lr0001',0.001)

#json files are combined manually
#Training function for the q4_part1

#Training function for the q4_part2
def train_q4_part2(model,model_name,inp_lr):
    #Necessary list  
    loss_curve_1 = []
    loss_curve_01 = []
    loss_curve_001 = []
    val_acc_curve_1 = []
    val_acc_curve_01 = []
    val_acc_curve_001 = []
    learning_rates = [inp_lr]
    
    for lr in learning_rates: 
        criterion = torch.nn.CrossEntropyLoss()
        print('TrainingStarts\n')        
        print(lr)
        
        model_training_loss = []
        model_training_accuracy = []
        validation_accuracy_list = []
        epochs = 30
  
        for epoch in range (0,epochs):
          #Updating the learning rate with the epoch that is observed in the previous stages
          if epoch == 7:
            lr = 0.01
          #Updating the learning rate with the epoch that is observed in the previous stages  
          if epoch == 14:
            lr == 0.001  
          print('epoch++')          
          model.train()
          #Updating optimizer with parameters
          optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)
          print(lr)

          for train_batch_num, (images, labels) in enumerate(train_generator):
            x = Variable(images).to(device)
            y = Variable(labels).to(device)
                
            optimizer.zero_grad()
            outputs = model(x) 
            loss = criterion(outputs, y)            
            loss.backward()
            optimizer.step()
                    
            if train_batch_num % 10 == 0:
              model.eval()
              prediction = outputs.data.max(dim=1)[1]
              model_training_accuracy.append((((prediction.data == y.data).float().mean()).item())*100)
              model_training_loss.append(loss.item())                                           
              correct = 0
              sample = 0
          
              for validation_batch_num, (images, labels) in enumerate(valid_generator):
                x=Variable(images).to(device)
                y=Variable(labels).to(device)
                score = model(x)
                _,prediction = score.max(1)
                correct += (prediction.data == y.data).sum()
                sample += prediction.size(0)
  
              validation_accuracy = float(correct) / float(sample) * 100            
              validation_accuracy_list.append(validation_accuracy)
              model.train()
  
          if lr == 0.1:
                loss_curve_1 = model_training_loss
                val_acc_curve_1 = validation_accuracy_list
          if lr == 0.01:
                loss_curve_01 = model_training_loss
                val_acc_curve_01 = validation_accuracy_list              
          if lr == 0.001:
                loss_curve_001 = model_training_loss
                val_acc_curve_001 = validation_accuracy_list                    
  
  
  
    #Create dictionary (Created by observing utils.py)
    
    dictionary = {"name": model_name,
                  "loss_curve_1":  loss_curve_1,
                  "loss_curve_01":  loss_curve_01,
                "loss_curve_001":  loss_curve_001,
                "val_acc_curve_1": val_acc_curve_1,
                "val_acc_curve_01": val_acc_curve_01,
                "val_acc_curve_001": val_acc_curve_001}
  
    #Write on json file
    #https://pythonexamples.org/python-write-json-to-file/
    with open('C:/Users/Can/Desktop/EE449_HW1'+'part4_2LR_'+  model_name + 'lr = {}'.format(lr) + '.json', 'w') as json_file:
        json.dump(dictionary, json_file)
        json_file.close()
             
    print('TrainingEnds')
    return    
 
train_q4_part2(cnn4_lr01,'cnn4',0.1)
#Training function for the q4_part2