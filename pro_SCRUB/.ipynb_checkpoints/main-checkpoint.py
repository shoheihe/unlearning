#!/usr/bin/env python3
import argparse
import json
import os
import time
import copy
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed

import models
import datasets_multiclass as datasets
from utils import *
from logger import Logger
import wandb

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill
from thirdparty.repdistiller.helper.pretrain import init
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate

def adjust_learning_rate(optimizer, epoch):
    if args.step_size is not None:lr = args.lr * 0.1 ** (epoch//args.step_size)
    else:lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss +=  (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from matplotlib.backends.backend_pdf import PdfPages
def feature_vector(args, model_net, retain_loader, forget_loader=None, num_classes=5, epoch=None, save_flg=True, data='train', modes='double'):
    #data:表示するdata_loader, 'forget'はretainとforget
    #mode:可視化するものをencoderかoutoutか両方か
    t0=time.time()
    mode_ls=[]
    mode_ls.append(modes)
    if modes=='double':
        mode_ls=['encoder', 'output']
    
    num_classes=args.num_classes
    label = ['0', '1', '2', '3', '4']
    color = ['blue', 'gray', 'lime', 'red', 'yellow']
    if args.dataset=='cifar10':
        label=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        color = ['blue', 'orange', 'lime', 'red', 'purple', 'yellow', 'pink', 'gray', 'green', 'cyan']
    label=[int(i) for i in label]
    #label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',  'frog', 'horse', 'ship', 'truck']
    #color = ['blue', 'orange', 'lime', 'red', 'purple']
    '''noise_label=label
    temp=noise_label.pop()
    noise_label.insert(0, temp)'''
    clean_or_noise_label=['clean', 'noise']
    #fig_count=0
    for clean_or_noise in clean_or_noise_label:
        for mode in mode_ls:
             
            plt.figure(figsize=(7,7))
            label_color_dict={}
            for key, item in zip(label, color):
                label_color_dict[key]=item


            dicts={}
            retain_dict={}
            retain_dict['loader']=retain_loader
            retain_dict['marker']='o'
            dicts['retain']=retain_dict

            if forget_loader is not None:
                forget_dict={}
                forget_dict['loader']=forget_loader
                forget_dict['marker']='*'
                dicts['forget']=forget_dict


            #print(dicts.items())
            #計算と可視化

            for dict in dicts.values():
                t1=time.time()
                print(f'plot_{dict}_start')
                data_loader=dict['loader']
                marker=dict['marker']
            

                print('*'*100)
                #print(f'represent_label:{represent_label}')
                feature_vector = []
                labels = []
                with torch.no_grad():
                    if dict.keys==['forget']:
                        for batch_idx, (inputs, clean_targets, noisy_targets) in enumerate(data_loader):
                            targets=clean_targets if clean_or_noise=='clean' else noisy_targets
                            print(targets)
                            inputs, targets = inputs.cuda(), targets.cuda()
                            #encoder = model_net(inputs)
                            output, encoder = model(inputs, mode='t-SNE')
                            encoder = encoder.cpu() 
                            if mode=='output':
                                encoder = output.cpu() 
                            feature_vector.append(encoder)   
                            targets = targets.cpu()
                            labels.extend(targets) 

                    else:
                        for batch_idx, (inputs, targets) in enumerate(data_loader):
                            inputs, targets = inputs.cuda(), targets.cuda()
                            #encoder = model_net(inputs)
                            output, encoder = model(inputs, mode='t-SNE')
                            encoder = encoder.cpu() 
                            if mode=='output':
                                encoder = output.cpu() 
                            feature_vector.append(encoder)   
                            targets = targets.cpu()
                            labels.extend(targets) 

                feature_vector = np.vstack(feature_vector)
                labels = np.array(labels)
                print(f'args.seed:{args.seed}')
                t2=time.time()
                print(f'until_before_TSNE:{t2-t1}')
                sfeature_vector_tsne = TSNE(n_components=2, random_state=args.seed).fit_transform(feature_vector)
                t3=time.time()
                print(f'TSNE_time:{t3-t2}')
                df_tsne = pd.DataFrame(sfeature_vector_tsne)
                df_labels = pd.DataFrame(labels)
                df_vec = pd.concat([df_tsne, df_labels], axis=1)
                df_vec = df_vec.set_axis(['tsne_X', 'tsne_Y', 'label'], axis = 'columns')

                #print('plot strt')
                t4=time.time()
                for instance in label: 
                    plt.scatter(df_vec.loc[df_vec["label"] == instance, "tsne_X"], 
                                df_vec.loc[df_vec["label"] == instance, "tsne_Y"], 
                                s=2, c=color[instance], marker=marker)#, label=label_name[name])
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

            if save_flg:
                plot_path='result/t-SNE/{}/class_{}_num_to_forget_{}/noise_mode_{}_msteps_{}_sgda_epochs_{}'.format(args.dataset, args.forget_class, args.num_to_forget, args.noise_mode, args.msteps, args.sgda_epochs)
                os.makedirs(plot_path,exist_ok=True)
                plot_name = os.path.join(plot_path,
                    'T-SNE_'+title_name)
                fig = plt.gcf()
                fig.savefig(f"{plot_name}.png", bbox_inches='tight', pad_inches=0.1)
                pp = PdfPages(f"{plot_name}.pdf")
                pp.savefig(fig, bbox_inches='tight', pad_inches=0.1)
                pp.close()
                plt.clf()


def run_epoch(args, model, model_init, train_loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, scheduler=None, epoch=0, weight_decay=0.0, mode='train', quiet=False, t_sne=False):
    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()
    elif mode == 'dry_run':
        model.eval()
        set_batchnorm_mode(model, train=True)
    else:
        raise ValueError("Invalid mode.")
    
    if args.disable_bn:
        set_batchnorm_mode(model, train=False)
    
    mult=0.5 if args.lossfn=='mse' else 1
    metrics = AverageMeter()

    with torch.set_grad_enabled(mode != 'test'):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            if args.lossfn=='mse':
                target=(2*target-1)
                target = target.type(torch.cuda.FloatTensor).unsqueeze(1)
                
            if 'mnist' in args.dataset:
                data=data.view(data.shape[0],-1)
                
            output = model(data)
            loss = mult*criterion(output, target) + l2_penalty(model,model_init,weight_decay)
            
            if args.l1:
                l1_loss = sum([p.norm(1) for p in model.parameters()])
                loss += args.weight_decay * l1_loss

            if ~quiet:
                metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
            
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    log_metrics(mode, metrics, epoch)
    logger.append('train' if mode=='train' else 'test', epoch=epoch, loss=metrics.avg['loss'], error=metrics.avg['error'], 
                  lr=optimizer.param_groups[0]['lr'])
    print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    # 可視化テスと
    '''if t_sne:
        #output, features = model(data, mode='t-SNE')
        feature_vector(args, model, train_loader, epoch=args.epochs)'''
    return metrics

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'forget'])
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--disable-bn', action='store_true', default=False,
                        help='Put batchnorm in eval mode and don\'t update the running averages')
    parser.add_argument('--epochs', type=int, default=31, metavar='N',
                        help='number of epochs to train (default: 31)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--forget-class', type=str, default=None,
                        help='Class to forget')
    parser.add_argument('--l1', action='store_true', default=False,
                        help='uses L1 regularizer instead of L2')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--num-to-forget', type=int, default=None,
                        help='Number of samples of class to forget')
    parser.add_argument('--confuse-mode', action='store_true', default=False,
                        help="enables the interclass confusion test")
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step-size', default=None, type=int, help='learning rate scheduler')
    parser.add_argument('--unfreeze-start', default=None, type=str, help='All layers are freezed except the final layers starting from unfreeze-start')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,30,30', help='lr decay epochs')
    parser.add_argument('--sgda-learning-rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
    parser.add_argument('--msteps', type=int, default=2, help='learning epoch of forget_dataset')
    parser.add_argument('--sgda-epochs', type=int, default=10, help='scrub_epochs')
    parser.add_argument('--noise-mode', type=str, default='sym', help='asym or sym')
    
    

    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    
    if args.forget_class is not None:
        clss = args.forget_class.split(',')
        args.forget_class = list([])
        for c in clss:
            args.forget_class.append(int(c))

    
    manual_seed(args.seed)
    seed=args.seed
    args.change_classes=None
    noise_mode=args.noise_mode
    
    if args.step_size==None:args.step_size=args.epochs+1
    
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.','_')}"
        if args.confuse_mode:
            args.name += f"_forget_{args.forget_class}"
            if args.num_to_forget is not None:
                args.name += f"_num_{args.num_to_forget}"
        elif args.split == 'train':
            args.name += f"_forget_{None}"
        else:
            args.name += f"_forget_{args.forget_class}"
            if args.num_to_forget is not None:
                args.name += f"_num_{args.num_to_forget}"
        if args.unfreeze_start is not None:
            args.name += f"_unfreeze_from_{args.unfreeze_start.replace('.','_')}"
        if args.augment:
            args.name += f"_augment"
        args.name+=f"_lr_{str(args.lr).replace('.','_')}"
        args.name+=f"_bs_{str(args.batch_size)}"
        args.name+=f"_ls_{args.lossfn}"
        args.name+=f"_wd_{str(args.weight_decay).replace('.','_')}"
        args.name+=f"_seed_{str(args.seed)}"
        args.name+=f"_msteps_{str(args.msteps)}"
        args.name+=f"_sgda_epochs_{str(args.sgda_epochs)}"
        args.name+=f"_noise_mode_{args.noise_mode}"
        
    print(f'Checkpoint name: {args.name}')
    
    mkdir('logs')

    logger = Logger(index=args.name+'_training')
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index+'.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index+'_{}.pth')

    print("[Logging in {}]".format(logger.index))
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    print('GPU or cpu : use {}'.format(args.device))
    os.makedirs('checkpoints', exist_ok=True)

    train_loader, valid_loader, test_loader = datasets.get_loaders(args, args.dataset, class_to_replace=args.forget_class,
                                                     num_indexes_to_replace=args.num_to_forget, confuse_mode=args.confuse_mode,
                                                     batch_size=args.batch_size, split=args.split, seed=args.seed,
                                                    root=args.dataroot, augment=args.augment)
    #t-SNEでforgetを表示するためだけにforegt_loaderとretain_loader作ってる
    args.retain_bs = 32
    args.forget_bs = 32
    #train_loader_full, valid_loader_full, test_loader_full = datasets.get_loaders(dataset, split="train",confuse_mode=True,class_to_replace=class_to_forget, num_indexes_to_replace=num_to_forget, batch_size=args.batch_size, seed=seed, root=args.dataroot, augment=False, shuffle=True)
    marked_loader, _, _ = datasets.get_loaders(args, args.dataset, split="forget", confuse_mode=True,class_to_replace=args.forget_class, num_indexes_to_replace=args.num_to_forget, only_mark=True, batch_size=1, seed=args.seed, root=args.dataroot, augment=False, shuffle=True)

    def replace_loader_dataset(data_loader, dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        manual_seed(seed)
        loader_args = {'num_workers': 0, 'pin_memory': False}
        def _init_fn(worker_id):
            np.random.seed(int(seed))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    print(forget_dataset.targets)
    print(len(forget_dataset.targets))
    print(type(forget_dataset.targets))
    
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(train_loader, forget_dataset, batch_size=args.forget_bs, seed=seed, shuffle=True)

    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(train_loader, retain_dataset, batch_size=args.retain_bs, seed=seed, shuffle=True)

    assert(len(forget_dataset) + len(retain_dataset) == len(train_loader.dataset))

    
    
    
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)
    
    if args.model=='allcnn':classifier_name='classifier.'
    elif 'resnet' in args.model:classifier_name='linear.'
    
    if args.resume is not None:
        state = torch.load(args.resume)
        state = {k: v for k, v in state.items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        assert all([k.startswith(classifier_name) for k in incompatible_keys.missing_keys])
    model_init = copy.deepcopy(model)
    

    torch.save(model.state_dict(), f"checkpoints/{args.name}_init.pt")
    
    parameters = model.parameters()
    if args.unfreeze_start is not None:
        parameters = []
        layer_index = 1e8
        for i, (n,p) in enumerate(model.named_parameters()):
            if (args.unfreeze_start in n) or (i > layer_index):
                layer_index = i
                parameters.append(p)
        
    weight_decay = args.weight_decay if not args.l1 else 0.
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=0.0)
    #optimizer = optim.Adam(parameters,lr=args.lr,weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn=='ce' else torch.nn.MSELoss().to(args.device)
    #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=0.1, last_epoch=-1)

    train_time = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer,epoch)
        #sgda_adjust_learning_rate(epoch, args, optimizer)
        t1 = time.time()
        t_sne_flg=False 
        run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='train', quiet=args.quiet, t_sne=t_sne_flg)
        if epoch>args.epochs-2:
            feature_vector(args, model, retain_loader, forget_loader, epoch=epoch, data='forget')
            feature_vector(args, model, test_loader, epoch=epoch, data='test')
            
        #train_acc, train_loss = train_vanilla(epoch, train_loader, model, criterion, optimizer, args)
        t2 = time.time()
        train_time += np.round(t2-t1,2)
        if epoch % 500000 == 0:
            if not args.disable_bn:
                run_epoch(args, model, model_init, train_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='dry_run')
            run_epoch(args, model, model_init, test_loader, criterion, optimizer, scheduler, epoch, weight_decay, mode='test')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}.pt")
            print(f'save:{args.name}_{epoch}')
        print(f'Epoch Time: {np.round(time.time()-t1,2)} sec')
    print (f'Pure training time: {train_time} sec')

# if __name__ == '__main__':
#     main()



