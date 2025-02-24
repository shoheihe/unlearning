
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import variational
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from itertools import cycle
import os
import time
import math
import pandas as pd
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
    
import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
from models import *
import models
from logger import *
import wandb
import sys
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss

from thirdparty.repdistiller.helper.loops import train_forget, train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate
from thirdparty.repdistiller.helper.pretrain import init
import warnings

# 警告を非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

cls_non_avearge = nn.CrossEntropyLoss(reduction='none')

def interclass_confusion(model, dataloader, class_to_forget, name):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=128, shuffle=False)
    model.eval()
    reals=[]
    predicts=[]
    for batch_idx, (data, target, _) in enumerate(dataloader):
        data, target = data.to(args.device), target.to(args.device) 
        if 'mnist' in args.dataset:
            data=data.view(data.shape[0],-1)
        output = model(data)
        probs = torch.nn.functional.softmax(output, dim=1)
        predict = np.argmax(probs.cpu().detach().numpy(),axis=1)
        reals = reals + list(target.cpu().detach().numpy())
        predicts = predicts + list(predict)
    
    classes = class_to_forget
    cm = confusion_matrix(reals, predicts, labels=classes)
    counts = 0
    '''for i in range(len(cm)):
        if i != class_to_forget[0]:
            counts += cm[class_to_forget[0]][i]
        if i != class_to_forget[1]:
            counts += cm[class_to_forget[1]][i]
    
    ic_err = counts / (np.sum(cm[class_to_forget[0]]) + np.sum(cm[class_to_forget[1]]))
    fgt = cm[class_to_forget[0]][class_to_forget[1]] + cm[class_to_forget[1]][class_to_forget[0]]
    #print (cm)
    return ic_err, fgt'''
    for i in range(len(cm)):
        for j in range(len(class_to_forget)):
            if i != class_to_forget[j]:
                counts += cm[class_to_forget[j]][i]
    
    sum_err=0
    for i in range(len(class_to_forget)):
        sum_err+=np.sum(cm[class_to_forget[i]])
    ic_err = counts / sum_err
    sum_fgt = 0
    for i in range(len(class_to_forget)):
        for j in range(len(class_to_forget)):
            if i!=j:
                sum_fgt+=cm[class_to_forget[i]][class_to_forget[j]]
    #print (cm)
    return ic_err, sum_fgt

from utils import *

def l2_penalty(model,model_init,weight_decay):
    l2_loss = 0
    for (k,p),(k_init,p_init) in zip(model.named_parameters(),model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p-p_init).pow(2).sum()
    l2_loss *= (weight_decay/2.)
    return l2_loss


def parameter_count(model):
    count=0
    for p in model.parameters():
        count+=np.prod(np.array(list(p.shape)))
    print(f'Total Number of Parameters: {count}')

def re_create_loader(args, loader):
    eval_train_loader = loader.run('noise_pred', shuffle=False)
    #各サンプルの損失保存リスト
    losses=[]
    c_or_n_s = []   #真にクリーンかノイズかを保存,Trueの場合はclean_sample, Falseの場合はnoisy_sample
    with torch.no_grad():
        for batch_idx, (data, target, _, c_or_n) in enumerate(eval_train_loader):
            data, target = data.cuda(), target.cuda()
            out = model_s(data)
            loss =cls_non_avearge(out, target)
            losses.extend(loss)
            c_or_n_s.extend(c_or_n)
    losses = np.array([loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses])
    print(sum(c_or_n_s))
    if args.pred == 'cls':
        #閾値より小さいものがclean sample
        preb = losses < args.cls_threshhold
        
    elif args.pred == 'gmm':
        losses = losses.reshape(-1, 1)
        gmm=GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        print(prob)
        #各サンプルがcleanである確率が格納されてる
        prob = prob[:, gmm.means_.argmin()]
        print(prob)
        #クリーンサンプルのindexはFalse, ノイジーサンプルはTrue
        # pred = (prob < args.gmm_threshhold)
        #クリーンサンプルのindexはTrue, ノイジーサンプルはFalse こっちが正解やと思うけど上より精度悪い
        pred = (prob > args.gmm_threshhold)
        
    
    # # クラスタリング結果
    # data=losses
    # labels = gmm.predict(data)
    # clean_cluster = np.argmin(gmm.means_)  # 平均値が小さいクラスタをクリーンと仮定
    # clean_data = data[labels == clean_cluster]
    # noisy_data = data[labels != clean_cluster]

    # # 各クラスタの確率密度
    # x = np.linspace(data.min() - 1, data.max() + 1, 500).reshape(-1, 1)
    # density = np.exp(gmm.score_samples(x))  # 全体の確率密度関数

    # # ヒストグラム: クリーンとノイジーを分ける
    # bin_width = (data.max() - data.min()) / 30  # ビン幅を手動計算
    # plt.hist(clean_data, bins=30, density=False, alpha=0.6, color='orange', label="Clean Data")
    # plt.hist(noisy_data, bins=30, density=False, alpha=0.6, color='blue', label="Noisy Data")

    # # 各コンポーネントの密度をスケール調整
    # for k, w in enumerate(gmm.weights_):
    #     component_density = w * np.exp(gmm._estimate_weighted_log_prob(x)[:, k]) * len(data) * bin_width
    #     plt.plot(x, component_density, linestyle="--", label=f"Component {k+1} Density")


    # # グラフ設定
    # plt.title("1D Gaussian Mixture Model")
    # plt.xlabel("loss")
    # plt.ylabel("num samples")
    # plt.legend()
    # plt.show()

    # fig = plt.gcf()
    # fig.savefig(f"test_{args.noise_mode}{args.num_to_forget}.png", bbox_inches='tight', pad_inches=0.1)

    print(f'clean sample num:{len(pred.nonzero()[0])}')
    cm = confusion_matrix(c_or_n_s, pred)
    report = classification_report(c_or_n_s, pred)
    print(cm)
    print(report)
    re_retain_loader = loader.run('re-retain', pred=pred)
    re_forget_loader = loader.run('re-forget', pred=pred)
    print(f'retain:{len(re_retain_loader.dataset)}\tforget:{len(re_forget_loader.dataset)}')
    # exit('pred end')
    return re_retain_loader, re_forget_loader
    
def scrub(args, mstep, sgda, centers=None):
    #pred noise
    
    if args.pred != '':
        retain_loader, forget_loader = re_create_loader(args, loader)
    else:
        retain_loader, forget_loader = correct_retain_loader, correct_forget_loader
    # print(correct_forget_loader, forget_loader, retain_loader)

    #事前学習時の重心の計算
    with torch.no_grad():
        local_encoder = []
        local_cluster_labels= [] 
        for id, (inputs, targets, _) in enumerate(retain_loader):
            inputs, targets=inputs.cuda(), targets.cuda()
            output, encoder=model(inputs, mode='t-SNE')
            local_cluster_labels.append(torch.softmax(output, dim=1))
            local_encoder.append(F.normalize(encoder, dim=1)) 
    centers = mean(args, local_cluster_labels, local_encoder, retain_loader)



    for epoch in range(1, sgda + 1):
        
        #moemntumで重心を使用
        if args.method=='pro-momentum':
            with torch.no_grad():
                local_encoder = []
                local_cluster_labels= [] 
                for id, (inputs, targets, _) in enumerate(retain_loader):
                    inputs, targets=inputs.cuda(), targets.cuda()
                    output, encoder=model(inputs, mode='t-SNE')
                    local_cluster_labels.append(torch.softmax(output, dim=1))
                    local_encoder.append(F.normalize(encoder, dim=1)) 

                next_centers = mean(args, local_cluster_labels, local_encoder, retain_loader)
                centers=args.pro_momentum * centers + (1 - args.pro_momentum) * next_centers

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        print("==> SCRUB unlearning ...")
        acc_r, acc5_r, loss_r = validate(correct_retain_loader, model_s, criterion_cls, args, True)
        acc_f, acc5_f, loss_f = validate(correct_forget_loader, model_s, criterion_cls, args, True)
        acc_test, acc5_test, loss_test = validate(test_loader, model_s, criterion_cls, args, True)
        acc_f_c, acc5_f_c, loss_f_c = validate(correct_forget_loader, model_s, criterion_cls, args, True, clean_label_val=True)
        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())
        acc_tests.append(100-acc_test.item())
        acc_f_cs.append(100-acc_f_c.item())

        maximize_loss = 0
        if epoch <= mstep:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer, args, "maximize", centers=centers)
        if args.run =='!' and epoch > 15:
            train_forget(args, epoch, forget_loader, module_list[0], optimizer, centers)
        train_acc, train_loss = train_distill(epoch, retain_loader, module_list, None, criterion_list, optimizer, args, "minimize", centers=centers)

        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
        f.write("epoch: {}\nmaximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}\n".format(epoch, maximize_loss, train_loss, train_acc))
        
        if epoch==mstep or epoch ==sgda:
            torch.save(model_s.state_dict(), f"../../scrub_model_{epoch}.pt")
            if args.tsne!=0:
                feature_vector(args, model_s, eval_retain_loader, eval_forget_loader, epoch=epoch, data='forget', modes=mode, centers=centers)
        


parser=argparse.ArgumentParser()
parser.add_argument('--model', default='preactresnet')
parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--forget-class', type=str, default='all',
                        help='Class to forget')
parser.add_argument('--num-to-forget', type=int, default=None,
                        help='Number of samples of class to forget')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--print_freq', type=int, default=500, help='print frequency')

parser.add_argument('--gamma', type=float, default=1, help='cross entropy in minimize')
parser.add_argument('--beta', type=float, default=0, help='cross entropy in minimize')
parser.add_argument('--alpha', type=float, default=0.01, help='cross entropy in minimize')
#自分で追加した分
parser.add_argument('--delta', type=float, default=0, help='cos_sim in minimize')
parser.add_argument('--zeta', type=float, default=0, help='cos_sim in maximize')
parser.add_argument('--eta', type=float, default=1, help='loss_div in maximize')
parser.add_argument('--pro-momentum', type=float, default=0.9, help='importance of pre_centers')
parser.add_argument('--pre_train_epoch', type=int, default=200)
parser.add_argument('--noise_mode', type=str, default='asym', choices=['sym', 'asym'], help='asym or sym')
parser.add_argument('--method', type=str, default='scrub')
parser.add_argument('--msteps', nargs='+', type=int, default=[2])
parser.add_argument('--sgda_epochs', nargs='*', type=int, default=[10])
parser.add_argument('--run', type=str, default=None
parser.add_argument('--par', type=float, default=0.5)
parser.add_argument('--theta', type=float, default=0.1)
parser.add_argument('--std', type=bool, default=False)
parser.add_argument('--tsne', type=int, default=0)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--forget_bs', type=int, default=128)
parser.add_argument('--retain_bs', type=int, default=128)
parser.add_argument('--kd_T', type=float, default=2)
parser.add_argument('--pred', type=str, default='')
parser.add_argument('--cls_threshhold', type=float, default=10.)
parser.add_argument('--gmm_threshhold', type=float, default=0.5)

parser.add_argument('--tsne_lr', type=int, default=350)
parser.add_argument('--tsne_per', type=int, default=50)
parser.add_argument('--file_name', type=str, default=None)
args = parser.parse_args()


#seedの固定
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
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else "cpu")

if 'cifar5' in args.dataset:
	args.num_classes=5
elif args.dataset=='cifar100':
    args.num_classes=100
elif 'cifar10' in args.dataset:
	args.num_classes=10
args.forget_class=list(range(args.num_classes))
args.dataroot='../../data/cifar100/cifar-100-python' if args.dataset=='cifar100' else '../../data/cifar10/cifar-10-batches-py'


training_epochs=args.pre_train_epoch-1
arch = args.model 
dataset = args.dataset
class_to_forget = args.forget_class
forget_class_name=f'{np.min(class_to_forget)}_{np.max(class_to_forget)}'
num_classes=args.num_classes
num_to_forget = args.num_to_forget
seed = args.seed
learningrate=f"lr_{str(args.lr).replace('.','_')}"
batch_size=f"_bs_{str(args.batch_size)}"
lossfn=f"_ls_{args.lossfn}"
wd=f"_wd_{str(args.weight_decay).replace('.','_')}"
seed_name=f"_seed_{args.seed}_"
noise_mode_name=f"noise_mode_{args.noise_mode}_"
args.noise_rate=float(args.num_to_forget/50000)
    

#モデルのLoad
# if args.model=='allcnn':
#     # m_name = f'../../checkpoints/{dataset}_{arch_filters}_forget_{class_to_forget}_num_{num_to_forget}{unfreeze_tag}{augment_tag}{learningrate}{batch_size}{lossfn}{wd}{seed_name}{noise_mode_name}{training_epochs}.pt'
#     m_name=f'../../check/{args.dataset}_{args.model}_forget_{forget_class_name}_num_{num_to_forget}_{learningrate}{batch_size}{lossfn}{wd}{seed_name}{noise_mode_name}{args.pre_train_epoch}.pt'

#     model=models.get_model(args.model, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)

#     print(f'---{m_name}---')
#     model.load_state_dict(torch.load(m_name))
file_name=f'{args.model}_{args.dataset}_{args.noise_rate}_{args.noise_mode}'

def create_model():
    model = models.get_model(args.model, num_classes=args.num_classes)
    model = model.cuda()
    return model
checkpoint = torch.load('../../weight/{}/net_seed{}/net/weight_save_{:04d}.tar'.format(file_name, args.seed, args.pre_train_epoch))
model = create_model()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
model.load_state_dict(checkpoint['state_dict'])
model_forget = create_model()
model_last=create_model()
model_forget.load_state_dict(torch.load('../../scrub_model_5.pt'))
model_last.load_state_dict(torch.load('../../scrub_model_15.pt'))

#modelの設定
torch.cuda.set_device(args.gpu)
model.cuda()
model_forget.cuda()
model_last.cuda()

#loaderの生成
noise_file = f'../../weight/{file_name}/net_seed{args.seed}/noise_file.json'
loader = datasets.cifar_dataloader(args.dataset,r=args.noise_rate,noise_mode=args.noise_mode,
                                     batch_size=args.batch_size,num_workers=12,\
    root_dir=args.dataroot,noise_file=noise_file,\
        retain_bs=args.retain_bs, forget_bs=args.forget_bs)
from collections import Counter
train_loader = loader.run('eval_train')
test_loader = loader.run('test')
correct_retain_loader = loader.run('retain')
eval_retain_loader = loader.run('retain', shuffle=False)
correct_forget_loader = loader.run('forget')
eval_forget_loader = loader.run('forget', shuffle=False)
train_labels=[]
for _,l,_ in train_loader:
    train_labels.extend(l.numpy())
print(Counter(train_labels))
print(f'forget sample num:{len(correct_forget_loader.dataset)}')
print(f'retain sample num:{len(correct_retain_loader.dataset)}')
print(f'test sanple num:{len(test_loader.dataset)}')
retain_loader = correct_retain_loader
forget_loader = correct_forget_loader

#ハイパーパラメータ
args.optim = 'sgd'
args.smoothing = 0.5
args.clip = 0.5
args.sstart = 10
args.distill = 'kd'
args.sgda_learning_rate = 0.0005
args.lr_decay_epochs = [7,10,10]
args.lr_decay_rate = 0.1
args.sgda_weight_decay = 5e-4
args.sgda_momentum = 0.9
#結果格納用リスト
acc_rs = []
acc_fs = []
acc_f_cs=[]
acc_vs = []
acc_tests=[]
ic_rs = []
ic_vs = []
ic_ts=[]
fgt_rs = []
fgt_vs = []
fgt_ts=[]
mode='encoder'
if args.method=="scrub":
    args.gamma=1
    args.alpha=0.01
    args.beta=0
    args.delta=0
    # args.eta=1
    args.zeta=0
    # args.kd_T=2
    # args.msteps=4 if args.dataset=='cifar100' else 3
    # args.sgda_epochs=4 if args.dataset=='ifar100' else 4
    print('--in==scrub')

#モデル・損失のリストの作成
criterion_cls = nn.CrossEntropyLoss()
criterion_div = DistillKL(args.kd_T)
criterion_kd = DistillKL(args.kd_T)
criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)    # classification loss
criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
criterion_list.append(criterion_kd)     # other knowledge distillation loss

#ファイルとディレクトリの作成
from datetime import datetime
import pytz
args.dir_name=f'../../lr_{args.tsne_lr}_per_{args.tsne_per}/'
os.makedirs(args.dir_name, exist_ok=True)

#事前学習時の重心の計算
with torch.no_grad():
    local_encoder = []
    local_cluster_labels= [] 
    for id, (inputs, targets, _) in enumerate(retain_loader):
        inputs, targets=inputs.cuda(), targets.cuda()
        output, encoder=model(inputs, mode='t-SNE')
        local_cluster_labels.append(torch.softmax(output, dim=1))
        local_encoder.append(F.normalize(encoder, dim=1)) 
centers = mean(args, local_cluster_labels, local_encoder, retain_loader)

feature_vector(args, model_forget, eval_retain_loader, eval_forget_loader, epoch=5, data='forget', modes=mode, centers=centers)
feature_vector(args, model_last, eval_retain_loader, eval_forget_loader, epoch=15, data='forget', modes=mode, centers=centers)
