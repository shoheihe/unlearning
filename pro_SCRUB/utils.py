#!/usr/bin/env python3
import argparse
import json
import os
import copy
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed

from io import BytesIO
import os
import errno

import models
import datasets_multiclass as datasets
import time


feat_dim=192

def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)
        
def mkdir(directory):
    '''Make directory and all parents, if needed.
    Does not raise and error if directory already exists.
    '''

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])	


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from matplotlib.backends.backend_pdf import PdfPages
def feature_vector(args, model_net, retain_loader, forget_loader=None, num_classes=5, epoch=None, save_flg=True, data='train', modes='double', centers=None, cluster_labels=None):
    #data:表示するdata_loader, 'forget'はretainとforget
    #mode:可視化するものをencoderかoutoutか両方か
    if 'resnet' in args.model:
        feat_dim=512
    elif 'allcnn' in args.model: feat_dim=192
    print('-'*100)
    copy_data_loader=copy.deepcopy(retain_loader)
    if forget_loader is not None:
        copy_forget_loader=copy.deepcopy(forget_loader)
    #dataloader=copy_forget_loader
    '''for (input, targets, clean) in dataloader:
        print(f'targetts:{targets}\nclean:{clean}')
    print('&'*100)
    dataloader=copy_forget_loader
    for (input, targets, clean) in dataloader:
        print(f'targetts:{targets}\nclean:{clean}')
'''
    os.environ['PYTHONHASEED']=str(args.seed)
    torch.backends.cudnn.deterministic = True
    # Faleseにしないと再現性が落ちる
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #t0=time.time()
    mode_ls=[]
    mode_ls.append(modes)
    
    if 'small' in args.dataset:
        size=25
    elif 'cifar' in args.dataset:
        size=5
    else:
        raise ValueError("args.dataset None")
    print(f'size:{size}')
    
    if modes=='double':
        mode_ls=['encoder', 'output']
    
    num_classes=args.num_classes
    color = ['blue', 'gray', 'lime', 'red', 'orange']
    if 'cifar10' in args.dataset:
        color = ['blue', 'orange', 'lime', 'red', 'purple', 'yellow', 'pink', 'gray', 'green', 'cyan']

    if args.noise_mode == 'SDN':
        color = ["blue","red","green","orange","purple","brown","pink","gray","olive","cyan","magenta","yellow","black","violet","gold","silver","navy","maroon","teal","lime"]


    label = list(range(args.num_classes))
    
    clean_or_noise_label=['clean']
    # if data!='test':
    #     clean_or_noise_label=['clean', 'noise']

    for clean_or_noise in clean_or_noise_label:
        for mode in mode_ls:
             
            plt.figure(figsize=(7,7))
            label_color_dict={}
            for key, item in zip(label, color):
                label_color_dict[key]=item


            dicts={}
            retain_dict={}
            retain_dict['loader']=copy_data_loader
            retain_dict['marker']='o'
            dicts['retain']=retain_dict

            if forget_loader is not None:
                forget_dict={}
                forget_dict['loader']=copy_forget_loader
                forget_dict['marker']='*'
                dicts['forget']=forget_dict

            markers_type = ['o','*']
            #print(dicts.items())
            #計算と可視化
            feature_vector = []
            labels = []
            clean_forget = []
            for dict in dicts.values():
                t1=time.time()
                print(f'plot_{dict}_start')
                data_loader=dict['loader']
                marker=dict['marker']
                with torch.no_grad():
                
                    for batch_idx, (inputs, noisy_targets, clean_targets) in enumerate(data_loader):
                        targets=clean_targets if clean_or_noise=='clean' else noisy_targets
                        
                        inputs, targets = inputs.cuda(), targets.cuda()
                        #encoder = model_net(inputs)
                        output, encoder = model_net(inputs, mode='t-SNE')
                        encoder = encoder.cpu() 
                        if mode=='output':
                            encoder = output.cpu() 
                        feature_vector.append(F.normalize(encoder, dim=1)) 
                        targets = targets.cpu()
                        labels.extend(targets)
                        if dict['marker'] == 'o':
                            clean_forget.extend([0] * len(targets))
                            
                        else:
                            clean_forget.extend([1] * len(targets))
                print(f'散布図にplotするデータ数:{len(np.vstack(feature_vector))}')

            if centers is not None and mode == 'encoder':
                centers=centers.cpu()
                print('--------------------------------')
                for i, encoder in enumerate(centers):
                    labels.append(100+i)
                    clean_forget.append(2)
                feature_vector.extend(F.normalize(centers, dim=1))    
            feature_vector = np.vstack(feature_vector)
            print(len(feature_vector))
            labels = np.array(labels)
            t2=time.time()
            print(f'until_before_TSNE:{t2-t1}')
            if len(feature_vector)>1000:
                sfeature_vector_tsne = TSNE(perplexity=args.tsne_per, n_components=2, learning_rate=args.tsne_lr, random_state=args.seed).fit_transform(feature_vector)
            else:
                sfeature_vector_tsne = TSNE(n_components=2, random_state=args.seed).fit_transform(feature_vector)
            t3=time.time()
            print(f'TSNE_time:{t3-t2}')
            df_tsne = pd.DataFrame(sfeature_vector_tsne)
            clean_forget = torch.tensor(clean_forget)
            labels = torch.tensor(labels)
            labels[clean_forget==1] =  labels[clean_forget==1] + args.num_classes
            df_labels = pd.DataFrame(labels)
            df_vec = pd.concat([df_tsne, df_labels], axis=1)
            df_vec = df_vec.set_axis(['tsne_X', 'tsne_Y', 'label'], axis = 'columns')
            labels_unique = torch.unique(labels)
            t4=time.time()
            print(labels_unique.tolist())
            # exit('100')
            for instance in labels_unique.tolist():
                if instance >50:
                    plt.scatter(df_vec.loc[df_vec["label"] == instance, "tsne_X"], 
                                df_vec.loc[df_vec["label"] == instance, "tsne_Y"], 
                                s=250, c=color[instance-100], marker='+')
                elif instance < args.num_classes:
                    
                    plt.scatter(df_vec.loc[df_vec["label"] == instance, "tsne_X"], 
                                df_vec.loc[df_vec["label"] == instance, "tsne_Y"], 
                                s=size, c=color[instance], marker=markers_type[0])
                else:
                    plt.scatter(df_vec.loc[df_vec["label"] == instance, "tsne_X"], 
                                df_vec.loc[df_vec["label"] == instance, "tsne_Y"], 
                                s=size, c=color[instance-args.num_classes], marker=markers_type[1])#, label=label_name[name])
            
            t5=time.time()
            print(f'scatter_time:{t5-t4}')
            font_size=0
            if font_size != 0:
                plt.legend(loc="upper left", fontsize=font_size)   
            plt.xticks([])
            plt.yticks([])
            title_name='dataloader_{}_mode_{}_label_{}_epoch_{}'.format(data, mode, clean_or_noise, epoch)
            plt.title(title_name)
            print('plot end')
            print(f'label_and_color:{label_color_dict}')
            print(title_name, epoch)
            if save_flg:
                plot_path=args.dir_name

                os.makedirs(plot_path,exist_ok=True)
                plot_name = os.path.join(plot_path,
                    'T-SNE_'+title_name)
                fig = plt.gcf()
                fig.savefig(f"{plot_name}.png", bbox_inches='tight', pad_inches=0.1)
                pp = PdfPages(f"{plot_name}.pdf")
                pp.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                pp.close()
                plt.clf()

def mean(args, local_cluster_labels, local_encoder, dataloader): # 重心求める, # GMMのasym0.9まだ
    if 'resnet' in args.model:
        feat_dim=512*args.filters
    else: feat_dim=192
    feat_dim=int(feat_dim)
    # それぞれの大きさの配列作る
    # print(args.filters, feat_dim)
    encoders = torch.zeros(len(dataloader.dataset), feat_dim).cuda()
    cluster_labels = torch.zeros(len(dataloader.dataset), args.num_classes).cuda()
    # print(cluster_labels.shape, encoders.shape, args.num_classes)
    # 配列の形変更
    local_encoder = torch.cat(local_encoder, dim=0)
    local_cluster_labels = torch.cat(local_cluster_labels, dim=0)
    # TCL参考にサンプルの情報を取得
    
    
    indices = torch.Tensor(list(iter(dataloader.sampler))).long().cuda()
    
    # サンプルの情報を追加
    encoders.index_add_(0, indices, local_encoder)
    cluster_labels.index_add_(0, indices, local_cluster_labels.float())
    # 重心を見つけている
    centers = F.normalize(cluster_labels.T.mm(encoders), dim=1)
    # centers = cluster_labels.T.mm(encoders)
    return centers
