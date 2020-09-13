import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import itertools
import tqdm
from mlmodel import *
import utils
import numpy as np
from sparsemax import Sparsemax
from argparse import ArgumentParser
import sklearn

def getPred_regress(x_loss, w_pred, model, model_shape):
  reshaped_w = utils.reshape_w(w_pred, model_shape)    
  pred_y = model(reshaped_w, x_loss)
  del x_loss, w_pred
  return pred_y

def getPred_binClass(x_loss, w_pred, model, model_shape):
  reshaped_w = utils.reshape_w(w_pred, model_shape)    
  pred = model(reshaped_w, x_loss)
  pred = torch.sigmoid(pred)
  del x_loss, w_pred
  return pred

class ZSTL:
    def __init__(self, w_kb, a_kb, base_model, param_dict, device):
        self.param_dict = param_dict

        #pre-set to gpu
        self.device = device
        self.w_kb = w_kb.to(self.device) 
        self.a_kb = a_kb.to(self.device)
        self.model = base_model
        self.param_dict = param_dict

        self.model_shape = param_dict['model_shape']
        self.d = param_dict['d']
        self.dm = param_dict['dm']
        '''
        rho - regu coef for w_kb
        mu - regu coef for a_kb
        '''
        self.rho = torch.tensor(param_dict['rho']).to(self.device)
        self.mu = torch.tensor(param_dict['mu']).to(self.device)

        self.init()
        #for alignment loss
        self.tolerance = torch.tensor(1e-4, dtype=torch.float32, requires_grad=False).to(self.device)

        if param_dict['atten_activation'] == 'Sparsemax':
            self.atten_activation = Sparsemax(dim=1)
        elif param_dict['atten_activation'] == 'Softmax':
            print('softmax selected')
            self.atten_activation = nn.Softmax(dim=1)

        if param_dict['loss'] == 'mse':
            self.loss = nn.MSELoss()
            self.metric = self.task_transfer_loss
            self.getPred = getPred_regress
            self.getPred_batch = self.getPred_batch_regress
        elif param_dict['loss'] == 'binary class':
            self.loss = self.sigmoid_loss
            self.metric = self.task_transfer_bi_acc
            self.getPred = getPred_binClass
            self.getPred_batch = self.getPred_batch_class

        elif param_dict['loss'] == 'mAP':
            self.loss = self.sigmoid_loss
            self.metric = self.task_transfer_precision
            self.getPred = getPred_binClass
            self.getPred_batch = self.getPred_batch_class
        

    def init(self,):
        # indx 1: w_r; indx 2: w_kb (need mannual set to gpu )
        self.w_r = 0.001*torch.eye(self.dm, requires_grad=True,device=self.device)
        self.hp = [self.w_r.clone().detach().requires_grad_(True).to(self.device), \
            self.w_kb.clone().detach().requires_grad_(True).to(self.device)]
        self.a_kb_opt = [self.a_kb.clone().detach().requires_grad_(True).to(self.device)]
        self.outer_opt = torch.optim.Adam(self.hp, lr=self.param_dict['outer lr'])
        self.align_opt = torch.optim.Adam(self.a_kb_opt, lr=self.param_dict['align lr'])
        torch.cuda.empty_cache()


    def sigmoid_loss(self, pred, target):
        #op only
        try:
            loss = F.binary_cross_entropy(pred, target)
        except:
            print('a_kb ', self.a_kb_opt)
            print('w_kb ', self.hp[1])
            print('logit ', pred)

        del pred, target
        torch.cuda.empty_cache()
        return loss

    def train(self, train_loader, test_loader, max_iter = 1000):
        test_batch = next(iter(test_loader))
        test_a, test_w, test_x, test_y = test_batch[0].float().to(self.device), test_batch[1].float().to(self.device), \
            test_batch[2].float().to(self.device), test_batch[3].float().to(self.device)
        test_a = test_a.squeeze().t()
        test_w = test_w.squeeze().t()
        print('test ',test_a.shape, test_w.shape, test_x.shape, test_y.shape)
        print('weight_kb ', self.hp[1].shape)
        test_loss_batch = self.metric(test_a,  test_x, test_y)
        align_loss = self.align_loss(test_w, self.hp[1].clone().detach().requires_grad_(False), \
                test_a)
        print('init mean test metric {}; align loss {}'.format(test_loss_batch, align_loss))

        train_l_lst = []
        test_l_lst = []
        for i in range(max_iter):
            train_batch = next(iter(train_loader))
            train_a, train_w, train_x, train_y = train_batch[0].float().to(self.device), train_batch[1].float().to(self.device),\
                train_batch[2].float().to(self.device), train_batch[3].float().to(self.device)
            train_a = train_a.squeeze().t()
            train_w = train_w.squeeze().t()
            # print('train ',train_a.shape, train_w.shape, train_x.shape, train_y.shape)
            
            self.outer_opt.zero_grad()
            train_loss_batch = self.task_transfer_loss(train_a, train_x, train_y)
            o_loss = train_loss_batch + self.rho*torch.pow(torch.norm(self.hp[1]), 2)
            train_l_lst.append(utils.toNumpy(o_loss.cpu().clone().detach().requires_grad_(False)))
            o_loss.backward()
            self.outer_opt.step()

            self.align_opt.zero_grad()
            align_loss = self.align_loss(train_w, self.hp[1].clone().detach().requires_grad_(False), \
                train_a)
            align_loss.backward()
            self.align_opt.step()

            if (i+1) % 200 == 0 or i == 0:
                test_loss_batch = self.metric(test_a, test_x, test_y)
                test_l_lst.append(utils.toNumpy(test_loss_batch.cpu().clone().detach().requires_grad_(False)))
                train_metric_batch = self.metric(train_a,  train_x, train_y)
                print('{}/{} o_loss {}; m train metric {}; m test metric {}; align loss  {}'.\
                    format(i+1, max_iter, o_loss, train_metric_batch, test_loss_batch, align_loss\
                        -self.mu*torch.pow(torch.norm(self.a_kb_opt[0]), 2)))

                torch.cuda.empty_cache()

        print('lr ', self.param_dict['outer lr'])
        plt.plot(train_l_lst, label='Traning: Outer Objectives')
        
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

        
        plt.plot(test_l_lst, label='Testing: ZSTL Metric')
        plt.xlabel('Iteration (x10)')
        plt.legend()
        plt.show()

        del test_a, test_w, test_x, test_y
        del train_a, train_w, train_x, train_y
        del align_loss
        torch.cuda.empty_cache()
        return 0

    def align_loss(self,weight_train, weight_kb, attr_train):
        cal_affinity = lambda a, b: torch.exp(torch.matmul(a, b)/torch.sqrt(torch.tensor(b.size()[0], dtype=float).to(self.device)))
        cal_atten = lambda a : a/torch.sum(a, dim=1, keepdim=True)

        #print('weight_kb ', weight_kb.shape)
        affinity_y_kb = cal_affinity(weight_kb.t(), weight_kb)
        y_kb_atten = cal_atten(affinity_y_kb)

        w_pred = self.task_transfer(attr_train).clone().detach().requires_grad_(False)
        affinity_y_train_kb = cal_affinity(w_pred.t(), weight_kb)
        y_train_kb_atten = cal_atten(affinity_y_train_kb)

        affinity_attr_kb = cal_affinity(self.a_kb_opt[0].t(), self.a_kb_opt[0])
        attr_kb_atten = cal_atten(affinity_attr_kb)
        affinity_attr_train_kb = cal_affinity(attr_train.t(), self.a_kb_opt[0])
        attr_train_kb_atten = cal_atten(affinity_attr_train_kb)
        mse_loss_kb =  F.mse_loss(attr_kb_atten, y_kb_atten)
        mse_loss_train_kb = F.mse_loss(attr_train_kb_atten, y_train_kb_atten)
        mse_loss = mse_loss_kb.item() + mse_loss_train_kb.item() + self.mu*torch.pow(torch.norm(self.a_kb_opt[0]), 2)

        del weight_train, attr_train, weight_kb
        torch.cuda.empty_cache()
        return mse_loss

    def zero_shot_transfer(self, test_loader):
        test_batch = next(iter(test_loader))
        test_a, test_w, test_x, test_y = test_batch[0].float().to(self.device), test_batch[1].float().to(self.device), \
            test_batch[2].float().to(self.device), test_batch[3].float().to(self.device)
        test_a = test_a.squeeze().t()
        test_w = test_w.squeeze().t()

        test_metric_batch = self.metric(test_a, test_x, test_y)
        del test_a, test_w, test_x, test_y
        torch.cuda.empty_cache()
        return test_metric_batch

    def task_transfer_loss(self, attr_test,  x, y):
        w_pred = self.task_transfer(attr_test)
        o_loss = torch.tensor(0.0, requires_grad=True, dtype=float).to(self.device)
        batch_size = w_pred.size()[1]
        #print(w_pred.size(), x.size())
        for t in range(batch_size):
            cur_x = x[t,:].float()
            cur_w = w_pred[:,t].unsqueeze(0).float()
            cur_y = y[t,:].float()
            o_loss = o_loss + self.outer_loss(cur_w, cur_x, cur_y)

        del attr_test,  x, y
        torch.cuda.empty_cache()
        return o_loss/batch_size

    def outer_loss(self, w, x_loss, y_loss):
        #print('params shape', params[0].shape)
        pred_y = self.getPred(x_loss, w, self.model, self.model_shape)
        
        o_loss = self.loss(pred_y, y_loss)

        del w, x_loss, y_loss
        torch.cuda.empty_cache()
        return o_loss

    def task_transfer_bi_acc(self, attr_test, x, y):
        w_pred = self.task_transfer(attr_test)
        #print(w_pred.size(), x.size())
        acc = torch.tensor(0.0, requires_grad=False, dtype=float).to(self.device) #new var
        num_data = torch.tensor(y[0,:].size()[0], dtype=float).to(self.device)
        num_task = torch.tensor(y.size()[0], dtype=float).to(self.device)
        for t in range(x.size()[0]):
            cur_w = w_pred[:, t].unsqueeze(0).float()
            pred = self.getPred(x[t,:].float(), cur_w, self.model, self.model_shape)
            pred[pred>=0.5] = torch.ones_like(pred[pred>=0.5], device=self.device)
            pred[pred<0.5] = torch.zeros_like(pred[pred<0.5], device=self.device)
            acc += torch.sum(pred == y[t,:])/num_data
            
        mean_acc = acc/num_task

        del num_data, num_task, attr_test, x, y
        torch.cuda.empty_cache()
        return mean_acc

    def task_transfer_precision(self, attr_test, x, y):
        w_pred = self.task_transfer(attr_test)
        #print(w_pred.size(), x.size())
        precision = torch.tensor(0.0, requires_grad=False, dtype=float).to(self.device) #new var
        num_data = torch.tensor(y[0,:].size()[0], dtype=float).to(self.device)
        num_task = torch.tensor(y.size()[0], dtype=float).to(self.device)
        for t in range(x.size()[0]):
            cur_w = w_pred[:, t].unsqueeze(0).float()
            pred = self.getPred(x[t,:].float(), cur_w, self.model, self.model_shape)
            pred[pred>=0.5] = torch.ones_like(pred[pred>=0.5], device=self.device)
            pred[pred<0.5] = torch.zeros_like(pred[pred<0.5], device=self.device)
            p = sklearn.metrics.precision_score(pred.clone().detach().cpu(), \
            y[t,:].clone().detach().cpu(), average='micro')
            precision += torch.tensor(p, requires_grad=False, dtype=float).to(self.device)
            #print('precision cal ', p)
            
        mean_precision = precision/num_task
        #print('mean_precision ', mean_precision, precision, num_task)
        del num_data, num_task, attr_test, x, y,
        torch.cuda.empty_cache()
        return mean_precision

    def task_transfer(self, attr_test):
        '''
        get pred task parameter for noval task
        '''
        w_pred = self.analytical_soln_atten(self.hp[1], attr_test, self.a_kb_opt[0])

        del attr_test
        torch.cuda.empty_cache()
        return w_pred

    def analytical_soln_atten(self, w_kbb, e_item, e_kbb):
        #find analytical son for one specific novel task
        '''
        c - row vector 
        w_kbb 
        '''
        affinity = self.Dot_Attention(e_item, e_kbb)
        #print('affinity ',affinity.size())
        
        c_newnew = self.atten_activation(affinity) 
        w = torch.matmul(c_newnew, w_kbb.t())

        del w_kbb, e_item, e_kbb
        torch.cuda.empty_cache()
        return w.t()

    def Dot_Attention(self, e_item, e_kbb):
        affinity = torch.matmul(e_item.t(), self.hp[0])
        affinity = torch.matmul(affinity, e_kbb)

        del e_item, e_kbb
        torch.cuda.empty_cache()
        return affinity

    def getPred_batch_regress( self,x,  weight, model, model_shape):
        pred_y_batch = []
        batch_size = weight.size()[1]
        for t in range(batch_size):
            cur_x = x[t,:].float()
            cur_w = weight[:,t].unsqueeze(0).float()
            pred_y = self.getPred(cur_x, cur_w, model, model_shape)
            pred_y_batch.append(pred_y.t())

        pred_y_batch = torch.cat(pred_y_batch, dim=0)

        del weight, x
        torch.cuda.empty_cache()
        return pred_y_batch

    def getPred_batch_class( self,x,  weight, model, model_shape):
        pred_y_batch = []
        batch_size = weight.size()[1]
        for t in range(batch_size):
            cur_x = x[t,:].float()
            cur_w = weight[:,t].unsqueeze(0).float()
            pred_y = self.getPred(cur_x, cur_w, model, model_shape)
            pred_y[pred_y>=0.5] = torch.ones_like(pred_y[pred_y>=0.5], device=self.device)
            pred_y[pred_y<0.5] = torch.zeros_like(pred_y[pred_y<0.5], device=self.device)
            pred_y_batch.append(pred_y.t())

        pred_y_batch = torch.cat(pred_y_batch, dim=0)

        del x,  weight
        torch.cuda.empty_cache()
        return pred_y_batch