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
# import hypergrad as hg
from mlmodel import *
import utils
import numpy as np
from sparsemax import Sparsemax
from argparse import ArgumentParser

def getPred_regress(x_loss, w_pred, model, model_shape):
  reshaped_w = utils.reshape_w(w_pred, model_shape)    
  pred_y = model(reshaped_w, x_loss)
  return pred_y

def getPred_binClass(x_loss, w_pred, model, model_shape):
  reshaped_w = utils.reshape_w(w_pred, model_shape)    
  pred = model(reshaped_w, x_loss)
  pred = torch.sigmoid(pred)
  
  return pred



class ZSTL:
    def __init__(self, w_kb, a_kb, x_kb, base_model, param_dict):
        self.param_dict = param_dict
        self.w_kb = w_kb
        self.a_kb = a_kb
        self.x_kb = x_kb
        self.model = base_model
        self.model_shape = param_dict['model_shape']
        self.d = param_dict['d']
        self.dm = param_dict['dm']
        '''
        lam - regu coef for w_kb
        mu - regu coef for a_kb
        '''
        self.rho = param_dict['rho']
        self.mu = param_dict['mu']

        # indx 1: w_r; indx 2: w_kb
        self.hp = [torch.eye(self.dm, requires_grad=True), self.w_kb.clone().detach().requires_grad_(True)]
        #self.hp = [torch.eye(self.dm, requires_grad=True)]
        self.a_kb_opt = a_kb.clone().detach().requires_grad_(False)
        self.outer_opt = torch.optim.Adam(self.hp, lr=param_dict['outer lr'])

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
            #self.loss = nn.BCEWithLogitsLoss()
            self.loss = self.sigmoid_loss
            self.metric = self.task_transfer_bi_acc
            self.getPred = getPred_binClass
            self.getPred_batch = self.getPred_batch_class

    def sigmoid_loss(self, pred, target):
        
        try:
            loss = F.binary_cross_entropy(pred, target)
        except:
            print('a_kb ', self.a_kb_opt)
            print('w_kb ', self.hp[1])
            print('logit ', pred)
        return loss

    def train(self, train_loader, test_loader, max_iter = 1000):
        test_batch = next(iter(test_loader))
        test_a, test_w, test_x, test_y = test_batch[0].float(), test_batch[1].float(), test_batch[2].float(), test_batch[3].float()
        test_a = test_a.squeeze().t()
        test_w = test_w.squeeze().t()

        test_loss_batch = self.metric(test_a, self.a_kb_opt, test_x, test_y)
        print('init mean test metric {};'.format(test_loss_batch))

        train_l_lst = []
        test_l_lst = []
        for i in range(max_iter):
            train_batch = next(iter(train_loader))
            train_a, train_w, train_x, train_y = train_batch[0].float(), train_batch[1].float(), train_batch[2].float(), train_batch[3].float()
            train_a = train_a.squeeze().t()
            train_w = train_w.squeeze().t()

            train_loss_batch = self.task_transfer_loss(train_a, self.a_kb_opt, train_x, train_y)
            o_loss = train_loss_batch + self.rho*torch.pow(torch.norm(self.hp[1]), 2)
            train_l_lst.append(utils.toNumpy(o_loss.clone().detach().requires_grad_(False)))
            self.outer_opt.zero_grad()
            o_loss.backward()
            self.outer_opt.step()

            self.a_kb_opt, mse_loss = self.attention_alignment(train_w, self.hp[1].clone().detach().requires_grad_(False), \
                train_a, self.a_kb_opt, train_x)

            test_loss_batch = self.metric(test_a, self.a_kb_opt, test_x, test_y)
            test_l_lst.append(utils.toNumpy(test_loss_batch.clone().detach().requires_grad_(False)))

            if (i+1) % 10 == 0 or i == 0:
                train_metric_batch = self.metric(train_a, self.a_kb_opt, train_x, train_y)
                print('{}/{} o_loss {}; m train metric {}; m test metric {}; align loss  {}'.\
                    format(i+1, max_iter, o_loss, train_metric_batch, test_loss_batch, mse_loss))

        print('lr ', self.param_dict['outer lr'])
        plt.plot(train_l_lst, label='Traning: Outer Objectives')
        
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

        
        plt.plot(test_l_lst, label='Testing: ZSTL Metric')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

        return 0

    def attention_alignment(self, weight_train, weight_kb, attr_train, attr_kb, x_loss):
        cal_affinity = lambda a, b: torch.exp(torch.matmul(a, b)/torch.sqrt(torch.tensor(b.size()[0], dtype=float)))
        cal_atten = lambda a : a/torch.sum(a, dim=1, keepdim=True)
        
        attr_kb_opt = [attr_kb.clone().detach().requires_grad_(True)]
        opt = torch.optim.Adam(attr_kb_opt, lr=self.param_dict['align lr'])

        totIter = 200

        #print('weight_train ', weight_train.shape, 'weight_kb ', weight_kb.shape)
        affinity_y_kb = cal_affinity(weight_kb.t(), weight_kb)
        y_kb_atten = cal_atten(affinity_y_kb)
        #print('y_kb_atten ', y_kb_atten.shape, torch.sum(y_kb_atten, dim=1))

        affinity_y_train_kb = cal_affinity(weight_train.t(), weight_kb)
        y_train_kb_atten = cal_atten(affinity_y_train_kb)
        #print('y_train_kb_atten ', y_train_kb_atten.shape, torch.sum(y_train_kb_atten, dim=1))
        #a = ppp

        # y_kb_pred = self.getPred_batch(x_loss,  weight_kb, self.model, self.model_shape)
        # affinity_y_kb = cal_affinity(y_kb_pred, y_kb_pred.t())
        # y_kb_atten = cal_atten(affinity_y_kb)

        # y_train_pred = self.getPred_batch(x_loss,  weight_train, self.model, self.model_shape)
        # affinity_y_train_kb = cal_affinity(y_train_pred, y_kb_pred.t())
        # y_train_kb_atten = cal_atten(affinity_y_train_kb)
        
        for t in range(totIter):
            affinity_attr_kb = cal_affinity(attr_kb_opt[0].t(), attr_kb_opt[0])
            attr_kb_atten = cal_atten(affinity_attr_kb)

            affinity_attr_train_kb = cal_affinity(attr_train.t(), attr_kb_opt[0])
            attr_train_kb_atten = cal_atten(affinity_attr_train_kb)

            
            mse_loss_kb =  F.mse_loss(attr_kb_atten, y_kb_atten)
            mse_loss_train_kb = F.mse_loss(attr_train_kb_atten, y_train_kb_atten)
            mse_loss = mse_loss_kb + mse_loss_train_kb + self.mu*torch.pow(torch.norm(attr_kb_opt[0]), 2)
            
            opt.zero_grad()
            mse_loss.backward()
            opt.step()
        #print('{}/{}: mse loss is {}'.format(t, totIter, mse_loss))
        return attr_kb_opt[0].clone().detach().requires_grad_(False), mse_loss

    def task_transfer_loss(self, attr_test, attr_kb, x, y):
        w_pred = self.task_transfer(attr_test, attr_kb)
        o_loss = torch.tensor(0.0, requires_grad=True, dtype=float)
        batch_size = w_pred.size()[1]
        #print(w_pred.size(), x.size())
        for t in range(batch_size):
            cur_x = x[t,:].float()
            cur_w = w_pred[:,t].unsqueeze(0).float()
            cur_y = y[t,:].float()
            o_loss = o_loss + self.outer_loss(cur_w, cur_x, cur_y)
        return o_loss/batch_size

    def outer_loss(self, w, x_loss, y_loss):
        #print('params shape', params[0].shape)
        pred_y = self.getPred(x_loss, w, self.model, self.model_shape)
        
        o_loss = self.loss(pred_y, y_loss)
        return o_loss

    def task_transfer_bi_acc(self, attr_test, attr_kb, x, y):
        w_pred = self.task_transfer(attr_test, attr_kb)
        #print(w_pred.size(), x.size())
        #pred = getPred(x, w_pred, self.model, self.model_shape)
        acc = torch.tensor(0.0, requires_grad=False, dtype=float)
        num_data = torch.tensor(y[0,:].size()[0], dtype=float)
        num_task = torch.tensor(y.size()[0], dtype=float)
        for t in range(x.size()[0]):
            cur_w = w_pred[:, t].unsqueeze(0).float()
            pred = self.getPred(x[t,:].float(), cur_w, self.model, self.model_shape)
            pred[pred>=0.5] = torch.ones_like(pred[pred>=0.5])
            pred[pred<0.5] = torch.zeros_like(pred[pred<0.5])
            acc += torch.sum(pred == y[t,:])/num_data
            
        mean_acc = acc/num_task

        # pred = torch.sigmoid(pred)
        # pred[pred>=0.5] = torch.ones_like(pred[pred>=0.5])
        # pred[pred<0.5] = torch.zeros_like(pred[pred<0.5])
        # print('pred ', pred.shape, 'y ', y.shape)
        # compare = (pred == y)
        # #print('compare ', compare.size(), compare)
        # compare = torch.sum(compare, dim=1)
        # #print('compare ', compare.size(), compare)
        # #print(y_loss.size()[1])
        # acc = torch.mean(compare.float())
        # #print('acc ', acc)
        # mean_acc = acc/y.size()[1]
        # #print(mean_acc)
        
        return mean_acc

    def task_transfer(self, attr_test, attr_kb):
        w_pred = self.analytical_soln_atten(self.hp[1], attr_test, attr_kb, self.hp[0])
        return w_pred

    

    def analytical_soln_atten(self, w_kbb, e_item, e_kbb, w_atten):
        #find analytical son for one specific novel task
        '''
        c - row vector 
        w_kbb 
        '''
        affinity = self.build_c_byAtten(e_item, e_kbb, w_atten)
        #print('affinity ',affinity.size())
        softmax = Sparsemax(dim=1)
        c_newnew = softmax(affinity) 
        #print('check softmax sum ', torch.sum(c_newnew, dim=1, keepdim=True))
        w = torch.matmul(c_newnew, w_kbb.t())
        return w.t()

    def build_c_byAtten(self, e_item, e_kbb, w_atten):
        dim = e_item.size()[0]
        #print('dim ', dim)
        affinity = torch.matmul(e_item.t(), w_atten)
        affinity = torch.matmul(affinity, e_kbb)

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
        return pred_y_batch

    def getPred_batch_class( self,x,  weight, model, model_shape):
        pred_y_batch = []
        batch_size = weight.size()[1]
        for t in range(batch_size):
            cur_x = x[t,:].float()
            cur_w = weight[:,t].unsqueeze(0).float()
            pred_y = self.getPred(cur_x, cur_w, model, model_shape)
            pred_y[pred_y>=0.5] = torch.ones_like(pred_y[pred_y>=0.5])
            pred_y[pred_y<0.5] = torch.zeros_like(pred_y[pred_y<0.5])
            pred_y_batch.append(pred_y.t())

        pred_y_batch = torch.cat(pred_y_batch, dim=0)
        return pred_y_batch




            