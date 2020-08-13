import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def toTensor(x, grad=False):
  t = torch.from_numpy(x)
  return t.float().requires_grad_(grad)

def toNumpy(t):
  x = t.detach().numpy()
  return x

def vectorize(weights):
  #weights with original model parameter shape
  shape_record = {}
  flatted_param = []

  for i, w in enumerate(weights):
    #print(w)
    if len(w) == 0:
      shape_record[i] = []
    else:
      shape_record[i] = []
      
      shape_record[i].append(w.shape)
      flatted = w.flatten()
      flatted_param = flatted_param + list(flatted)
      #print(len(flatted_param))
    
  return flatted_param, shape_record


def flattenParam(model_info):
  #flatten the model's parameter 

  attribute = model_info[0]
  weights = model_info[1]
  #print(len(weights), weights)
  x = model_info[2]
  y = model_info[3]

  shape_record = {}
  flatted_param = []

  flatted_param, shape_record = vectorize(weights)
  return flatted_param, shape_record

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset):
        'Initialization'
        self.dataset = dataset

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        item = self.dataset[index]

        # Load data and get label
        a = np.array(item[0])
        a = np.expand_dims(a, axis=0)
      
        w = item[1]
        w = self.vectorize(w)
        w = np.expand_dims(w, axis=0)

        x = np.array(item[2])
        y = np.array(item[3])

        return a, w, x, y

  def vectorize(self, weights):
        #weights with original model parameter shape
        flatted_param = []

        for i, w in enumerate(weights):
            #print(w)
            if len(w) == 0:
                pass
            else:
                flatted = w.flatten()
                flatted_param = flatted_param + list(flatted)
        
        return flatted_param

def reshape_w(params, shape_record):
    #print(shape_record, len(params))
    cur_indx = 0
    param_list = []
    for key in shape_record.keys():
        offset = np.prod(np.array(shape_record[key]))
        #print('cur_indx ', cur_indx,' offset ', offset)
        size_param = shape_record[key][0]
        #print('size param', len(size_param),size_param)
        #print('slice size ', params[0][cur_indx:cur_indx+offset].size())
        param_list.append(params[0, cur_indx:cur_indx+offset].view(size_param))
        cur_indx += offset
    return param_list

def taskVisualize(item, model, model_shape):
  attribute = item[0].squeeze()
  print(attribute)
  weights = item[1]
  x = item[2]
  y = item[3]

  reshaped_w = reshape_w(weights, model_shape)       
  pred_y = model(reshaped_w, x)
  print('pred_y shape', pred_y.size())
  loss = F.mse_loss(pred_y, y)

  print('Amplitude A={}, Frequency f={}, phase={}pi, pred loss={}'.format(attribute[0], attribute[1], attribute[2]/np.pi, loss))

  plt.plot(x, pred_y, '.', c='r', label='pred')
  plt.plot(x, y, '.', c='b', label='gt')
  plt.xlabel('Radian')
  plt.ylabel('Magitude')
  plt.legend()
  plt.show()

def sigmoid(theta):
  theta[theta < -100] = -100
  return 1/(1+np.exp(-theta))