import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from ZSTL_GPU import ZSTL
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

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

  def __getitem__(self, index, sampling=False, sampling_size=10):
        'Generates one sample of data'
        # Select sample
        item = self.dataset[index]

        # Load data and get label
        a = np.array(item[0])
        a = np.expand_dims(a, axis=0)
      
        w = item[1]
        w = self.vectorize(w)
        w = np.expand_dims(w, axis=0)
        if not sampling:
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


def genSplits_hectrec(compressed_data, detailed_data, train_size, test_size, support_size, train_batch_size=100):

    task_id = list(compressed_data.keys())
    tot_len = len(task_id)

    support_indx = list(np.random.choice(task_id, size=support_size, replace=False))
    print(len(support_indx))
    temp = [x for x in task_id if x not in support_indx]
    train_indx = list(np.random.choice(temp, size=train_size, replace=False))
    temp = [x for x in temp if x not in train_indx]
    print(len(train_indx))
    test_indx = temp
    print(len(test_indx))

    support_data = Dataset_hetrec([compressed_data[d] for d in support_indx], detailed_data)
    train_data = Dataset_hetrec([compressed_data[d] for d in train_indx], detailed_data)
    test_data = Dataset_hetrec([compressed_data[d] for d in test_indx], detailed_data)

    support_loader = DataLoader(support_data, batch_size=support_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=True)

    return support_loader, train_loader, test_loader

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

def hp_select_binClass(train_loader, val_loader, support_loader, d, dm,  model, model_shape, device, val_step=1500):
    train_a, train_w, train_x, train_y = next(iter(train_loader))
    train_a, train_w, train_x, train_y = train_a.float(), train_w.float(), train_x.float(), train_y.float()
    print(train_a.size()[0])
    val_a, val_w, val_x, val_y = next(iter(val_loader))
    val_a, val_w, val_x, val_y = val_a.float(), val_w.float(), val_x.float(), val_y.float()
    print(val_a.size()[0])

    support_a, support_w, support_x, support_y = next(iter(support_loader))
    support_a, support_w, support_x, support_y = support_a.float(), support_w.float(), support_x.float(), support_y.float()
    support_a = support_a.squeeze().t()
    support_w = support_w.squeeze().t()

    best_hp = {}
    best_metric = 0.0
    regu_param_rho = [ 10**(-1), 10**(-2), 10**(-3), 10**(-4),  10**(-5)]
    regu_param_mu = [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)]
    #dict_k_model = [5 , 6, 7, 8, 9, 10]
    param_dict = {}
    param_dict['rho'] = 0.0001
    param_dict['mu'] = 0.001
    param_dict['loss'] = 'binary class'
    param_dict['outer lr'] = 1e-3
    param_dict['align lr'] = 1e-4
    param_dict['dm'] = dm
    param_dict['d'] = d
    param_dict['model_shape'] = model_shape
    param_dict['atten_activation'] = 'Sparsemax'

    hp_lst = list(itertools.product(regu_param_rho, regu_param_mu))
    print('num of hp ', len(hp_lst))
    for hp in hp_lst:
        param_dict['rho'] = hp[0]
        param_dict['mu'] = hp[1]
        print('rho for w_kb {}; mu for a_kb {};'.format(param_dict['rho'], param_dict['mu']))
        
        ZSTL_model = ZSTL(support_w, support_a, model, param_dict, device)
        ZSTL_model.train(train_loader, val_loader, max_iter=val_step)

        mean_metric = ZSTL_model.zero_shot_transfer(val_loader)
        print('mean metric {}'.format(mean_metric))
        if mean_metric >= best_metric:
            print('New best acc {}'.format(mean_metric))
            best_metric = mean_metric
            best_hp['mu'] = float(param_dict['mu'])
            best_hp['rho'] = float(param_dict['rho'])
            #best_hp = param_dict

    return best_hp

def hp_select_regression(train_loader, val_loader, support_loader, d, dm,  model, model_shape, device, val_step=1500):
    train_a, train_w, train_x, train_y = next(iter(train_loader))
    train_a, train_w, train_x, train_y = train_a.float(), train_w.float(), train_x.float(), train_y.float()
    print(train_a.size()[0])
    val_a, val_w, val_x, val_y = next(iter(val_loader))
    val_a, val_w, val_x, val_y = val_a.float(), val_w.float(), val_x.float(), val_y.float()
    print(val_a.size()[0])

    support_a, support_w, support_x, support_y = next(iter(support_loader))
    support_a, support_w, support_x, support_y = support_a.float(), support_w.float(), support_x.float(), support_y.float()
    support_a = support_a.squeeze().t()
    support_w = support_w.squeeze().t()

    best_hp = {}
    best_metric = float('inf')
    regu_param_rho = [ 10**(-1), 10**(-2), 10**(-3), 10**(-4),  10**(-5)]
    regu_param_mu = [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)]
    #dict_k_model = [5 , 6, 7, 8, 9, 10]
    param_dict = {}
    # param_dict['rho'] = 0.0001
    # param_dict['mu'] = 0.001
    param_dict['loss'] = 'mse'
    param_dict['outer lr'] = 1e-3
    param_dict['align lr'] = 1e-4
    param_dict['dm'] = dm
    param_dict['d'] = d
    param_dict['model_shape'] = model_shape
    param_dict['atten_activation'] = 'Sparsemax'

    hp_lst = list(itertools.product(regu_param_rho, regu_param_mu))
    print('num of hp ', len(hp_lst))
    for hp in hp_lst:
        param_dict['rho'] = hp[0]
        param_dict['mu'] = hp[1]
        print('rho for w_kb {}; mu for a_kb {};'.format(param_dict['rho'], param_dict['mu']))
        
        ZSTL_model = ZSTL(support_w, support_a, model, param_dict, device)
        ZSTL_model.train(train_loader, val_loader, max_iter=val_step)

        mean_metric = ZSTL_model.zero_shot_transfer(val_loader)
        print('mean metric {}'.format(mean_metric))
        if mean_metric <= best_metric:
            print('New best acc {}'.format(mean_metric))
            best_metric = mean_metric
            best_hp['mu'] = float(param_dict['mu'])
            best_hp['rho'] = float(param_dict['rho'])
            #best_hp = param_dict

    return best_hp


def hp_select_mAP(train_loader, val_loader, support_loader, d, dm,  model, model_shape, device):
    train_a, train_w, train_x, train_y = next(iter(train_loader))
    train_a, train_w, train_x, train_y = train_a.float(), train_w.float(), train_x.float(), train_y.float()
    print(train_a.size()[0])
    val_a, val_w, val_x, val_y = next(iter(val_loader))
    val_a, val_w, val_x, val_y = val_a.float(), val_w.float(), val_x.float(), val_y.float()
    print(val_a.size()[0])

    support_a, support_w, support_x, support_y = next(iter(support_loader))
    support_a, support_w, support_x, support_y = support_a.float(), support_w.float(), support_x.float(), support_y.float()
    support_a = support_a.squeeze().t()
    support_w = support_w.squeeze().t()

    best_hp = {}
    best_metric = 0.0
    regu_param_rho = [ 10**(-1), 10**(-2), 10**(-3), 10**(-4),  10**(-5)]
    regu_param_mu = [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)]
    #dict_k_model = [5 , 6, 7, 8, 9, 10]
    param_dict = {}
    param_dict['rho'] = 0.0001
    param_dict['mu'] = 0.001
    param_dict['loss'] = 'binary class'
    param_dict['outer lr'] = 1e-3
    param_dict['align lr'] = 1e-4
    param_dict['dm'] = dm
    param_dict['d'] = d
    param_dict['model_shape'] = model_shape
    param_dict['atten_activation'] = 'Sparsemax'

    hp_lst = list(itertools.product(regu_param_rho, regu_param_mu))
    print('num of hp ', len(hp_lst))
    for hp in hp_lst:
        param_dict['rho'] = hp[0]
        param_dict['mu'] = hp[1]
        print('rho for w_kb {}; mu for a_kb {};'.format(param_dict['rho'], param_dict['mu']))
        
        ZSTL_model = ZSTL(support_w, support_a, support_x, model, param_dict, device)
        ZSTL_model.train(train_loader, val_loader, max_iter=1500)

        mean_metric = ZSTL_model.zero_shot_transfer(val_loader)
        print('mean metric {}'.format(mean_metric))
        if mean_metric >= best_metric:
            print('New best acc {}'.format(mean_metric))
            best_metric = mean_metric
            best_hp['mu'] = float(param_dict['mu'])
            best_hp['rho'] = float(param_dict['rho'])
            #best_hp = param_dict

    return best_hp

def cal_AvgPrecision_k(pred_y, y, k=5):
    relavance = torch.tensor(0.0, requires_grad=False, dtype=float) #new var
    count = torch.tensor(0.0, requires_grad=False, dtype=float) #new var
    num_data = torch.tensor(y[0,:].size()[0], dtype=float)
    precision = torch.tensor(0.0, requires_grad=False, dtype=float)
    #pos_label = torch.tensor(1.0, )

    #print('pred_y ', pred_y.shape)
    #print('y ', y.shape)
    
    pred_y_sorted_indx = torch.squeeze(torch.argsort(pred_y, dim=0, descending=True))
    #print('pred_y_sorted_indx ', pred_y_sorted_indx)
    pred_y_sorted = pred_y[pred_y_sorted_indx]
    #print('pred_y_sorted ', pred_y_sorted.shape, pred_y_sorted)
    y_sorted = y[pred_y_sorted_indx]
    #print('y_sorted ', y_sorted)

    for i in range(k):
        # if  pred_y_sorted[i] >= 0.5:
          count += 1
          if y_sorted[i] == 1 :
              relavance += 1
              precision += relavance/count
    
    if relavance == 0:
        return precision
    else:
        precision_atK = precision/relavance
        #print('precision_atK ', precision_atK)
        return precision_atK


def ZSTL_AvgPrecision(attr_test, x, y, ZSTL_model):
    attr_test = attr_test.to(ZSTL_model.device)
    #attr_test = attr_test.to(ZSTL_model.device)
    w_pred = ZSTL_model.task_transfer(attr_test)
    #print(w_pred.size(), x.size())
    #pred = getPred(x, w_pred, self.model, self.model_shape)
    precision_atK = torch.tensor(0.0, requires_grad=False, dtype=float)
    x = x.to(ZSTL_model.device)
    num_task = torch.tensor(y.size()[0], dtype=float)
    for t in range(y.size()[0]):
        cur_w = w_pred[:, t].unsqueeze(0).float()
        pred = ZSTL_model.getPred(x.float(), cur_w, ZSTL_model.model, ZSTL_model.model_shape).cpu()
        #print(y[t,:].shape)
        precision_atK += cal_AvgPrecision_k(pred, y[t,:], k=100)
        #a = ppp
    mAP = precision_atK/num_task
    print('mAP at 100 ', mAP, 'num task ', num_task)
    
    del x, w_pred, attr_test, cur_w, ZSTL_model
    torch.cuda.empty_cache()
    return mAP

class Dataset_hetrec(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, compressed_data, detailed_data, preset_x = None, phase='Normal'):
        'Initialization'
        self.compressed_data = compressed_data

        self.num_artist = detailed_data['num_artist']
        #print('num_artist ', self.num_artist, len(list(detailed_data['artistID_to_X_row'].keys())))
        self.num_tag = detailed_data['num_tag']
        #print('num_tag', self.num_tag)
        self.userID_to_Y_row = detailed_data['userID_to_Y_row']
        self.artistID_to_X_row = detailed_data['artistID_to_X_row']
        self.tag_to_x_col = detailed_data['tag_to_x_col']
        #print('tag_to_x_col ',len(list(self.tag_to_x_col.keys())))
        #print

        
        self.y = detailed_data['y']
        self.a = detailed_data['a']
        #print('a ', self.a)
        self.pahse = phase
        self.param = {}

        if preset_x is None:
            print('gen x')
            self.x = np.zeros((self.num_artist, self.num_tag))
            self.genX()
        else:
            self.x = preset_x

        print('gen w, wait ')
        self.genW()
        print('gen w done')

  def genX(self,):
        tag_len = 0
        not_inLst = set()
        for data in self.compressed_data:
            k = data[0]
            cur_tag_record = data[1][-1]
            tag_len += len(cur_tag_record)
            #print('cur_tag_record ', cur_tag_record)
            for item in cur_tag_record:
                #print(item)
                artist = item[0]
                tag = item[1]
                if artist in self.artistID_to_X_row.keys():
                    self.x[self.artistID_to_X_row[artist], self.tag_to_x_col[tag]] += 1
                else:
                    not_inLst.add(artist)
            
        #print('x shape ', self.x.shape)
        ones = np.ones((self.x.shape[0], 1))
        self.x = np.concatenate([ones, self.x], axis=1)
        print('x shape ', self.x.shape)

        
        # print('avg listen ', np.sum(self.x))
        # print('avg tag ', tag_len/len(self.compressed_data), tag_len)
        # print('num not in ', len(not_inLst))

  def genW(self,):
        
        for data in self.compressed_data:
            k = data[0]
            cur_compressed_data = data[1]
            cur_y = self.y[cur_compressed_data[0],:]

            task_y = cur_y[cur_compressed_data[1]]
            task_x = [np.expand_dims(self.x[i,:], axis=0) for i in cur_compressed_data[1]]
            task_x = np.concatenate(task_x, axis=0)

            clf = LogisticRegression(fit_intercept = False, max_iter=1000,random_state=0).fit(task_x, task_y)
            pred_y = clf.predict(task_x)

            #print('acc ', np.sum(pred_y==task_y)/task_y.shape[0])
            self.param[k] = clf.coef_
        #print(self.param[k].shape)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.compressed_data)

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

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        k = self.compressed_data[index][0]
        item = self.compressed_data[index][1]
        data_indx = item[0]
        # Load data and get label
        #print('data_indx ', data_indx)
        a = np.array(self.a[data_indx, :])
        a = np.expand_dims(a, axis=0)
      
        w = self.param[k]
        w = np.expand_dims(self.vectorize(w), axis=0)

        if self.pahse == 'Normal':
            selected_data = item[1]
            x = [np.expand_dims(self.x[i,:], axis=0) for i in selected_data]
            x = np.concatenate(x, axis=0)
            cur_y = self.y[data_indx, :]
            y = np.expand_dims(cur_y[selected_data], axis=0).T
            return a, w, x, y
        elif self.pahse == 'mAP':
            y = np.expand_dims(self.y[data_indx, :], axis=1)
            return a, w, y

class CSR_model(nn.Module):
    def __init__(self,num_in, num_out, num_hidden1=500, num_hidden2=200 ):
        super(CSR_model, self).__init__()
        
        self.fc1 =  nn.Linear(num_in, num_hidden1)
        #self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden1, num_out)
        

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x

def CSR_train(task_x, task_y, num_epochs, learning_rate, model, device):

    task_x = toTensor(task_x).float().to(device)
    task_y = toTensor(task_y).float().to(device)
    # model = model.to(device)
    
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

    
    for epoch in range(num_epochs):       
        
        #print('output ', output.shape)
        optimizer.zero_grad()
        output = model(task_x)
        output = torch.sigmoid(output)
        loss = F.binary_cross_entropy(output, task_y)
        #print('loss ', loss)
        loss.backward()
        optimizer.step()
        
        #aa =bb
        # if (epoch+1) % 100 == 0:    
        #   with torch.no_grad():
        #     train_loss = loss.item() 
        #     pred_y = model(task_x)
        #     pred_y = torch.sigmoid(pred_y)
        #     pred_y[pred_y>=0.5] = torch.ones_like(pred_y[pred_y>=0.5])
        #     pred_y[pred_y<0.5] = torch.zeros_like(pred_y[pred_y<0.5])
        #     #print('pred_y shape ', pred_y.shape)
        #     acc = torch.sum(pred_y==task_y)/torch.tensor(task_y.size()[0], dtype=torch.float32)

        #   print('====> Epoch: {} Average loss: {:.4f}; ACC {}'.format(
        #         epoch+1, train_loss, acc))
        
    # print('pred_y ', pred_y)
    # print('y ', task_y)
    #print('compare ', pred_y==task_y)
    
    weights = []
    for w in model.parameters():
      weights.append(w.cpu().detach().numpy())
      #print(w.shape)
    del model, task_x, task_y, output, optimizer
    torch.cuda.empty_cache()
    return weights