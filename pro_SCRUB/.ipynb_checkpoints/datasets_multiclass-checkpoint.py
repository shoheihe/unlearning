from typing import List, Union

import os
import copy
import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms
from lacuna import Lacuna10, Lacuna100, Small_Lacuna10, Small_Binary_Lacuna10, Small_Lacuna5,Small_Lacuna6
from Small_CIFAR10 import Small_CIFAR10, Small_Binary_CIFAR10, Small_CIFAR5, Small_CIFAR6
from Small_MNIST import Small_MNIST, Small_Binary_MNIST
from TinyImageNet import TinyImageNet_pretrain, TinyImageNet_finetune, TinyImageNet_finetune5
from IPython import embed

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_DATASETS = {}

def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_mnist_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor()
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_lacuna_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_cifar_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Pad(padding=4, fill=(125,123,113)),
        transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_imagenet_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

def _get_mix_transforms(augment=True):
    transform_augment = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transform_augment if augment else transform_test

    return transform_train, transform_test

@_add_dataset   
def cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar5(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_CIFAR5(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar6(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_CIFAR6(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def small_binary_cifar10(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = Small_Binary_CIFAR10(root=root, train=True, transform=transform_train)
    test_set  = Small_Binary_CIFAR10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset   
def cifar100(root, augment=False):
    transform_train, transform_test = _get_cifar_transforms(augment=augment)
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_mnist(root, augment=False):
    transform_train, transform_test = _get_mnist_transforms(augment=augment)
    train_set = Small_Binary_MNIST(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_MNIST(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna100(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna100(root=root, train=True, transform=transform_train)
    test_set = Lacuna100(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna5(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna5(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna6(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna6(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna6(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def small_binary_lacuna10(root, augment=False):
    transform_train, transform_test = _get_lacuna_transforms(augment=augment)
    train_set = Small_Binary_Lacuna10(root=root, train=True, transform=transform_train)
    test_set = Small_Binary_Lacuna10(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_pretrain(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_pretrain(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_pretrain(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def tinyimagenet_finetune5(root, augment=False):
    transform_train, transform_test = _get_imagenet_transforms(augment=augment)
    train_set = TinyImageNet_finetune5(root=root, train=True, transform=transform_train)
    test_set = TinyImageNet_finetune5(root=root, train=False, transform=transform_test)
    return train_set, test_set

@_add_dataset
def mix10(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna10(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna10(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(5)
    for c in classes:
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]
    return lacuna_train_set, lacuna_test_set

@_add_dataset
def mix100(root, augment=False):
    transform_train, transform_test = _get_mix_transforms(augment=augment)
    lacuna_train_set = Lacuna100(root=root, train=True, transform=transform_train)
    lacuna_test_set = Lacuna100(root=root, train=False, transform=transform_test)
    cifar_train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)
    cifar_test_set  = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform_test)
    
    lacuna_train_set.targets = np.array(lacuna_train_set.targets)
    lacuna_test_set.targets = np.array(lacuna_test_set.targets)
    cifar_train_set.targets = np.array(cifar_train_set.targets)
    cifar_test_set.targets = np.array(cifar_test_set.targets)
        
    lacuna_train_set.data = lacuna_train_set.data[:,::2,::2,:]
    lacuna_test_set.data = lacuna_test_set.data[:,::2,::2,:]
    
    classes = np.arange(50)    
    for c in classes:
        lacuna_train_class_len = np.sum(lacuna_train_set.targets==c)
        lacuna_train_set.data[lacuna_train_set.targets==c]=cifar_train_set.data[cifar_train_set.targets==c]\
                                                            [:lacuna_train_class_len,:,:,:]
        lacuna_test_class_len = np.sum(lacuna_test_set.targets==c)
        lacuna_test_set.data[lacuna_test_set.targets==c]=cifar_test_set.data[cifar_test_set.targets==c]\
                                                            [:lacuna_test_class_len,:,:,:]
    return lacuna_train_set, lacuna_test_set



#def replace_indexes(dataset: torch.utils.data.Dataset, indexes: Union[List[int], np.ndarray], seed=0, only_mark: bool = False):
def replace_indexes(dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False):
    if type(indexes)==type({}):
        temp=[]
        for _,item in indexes.items():
            temp=temp+item
        indexes=temp
    
    #print(len(indexes))
    indexes=np.array(indexes)
    indexes=indexes.astype(int)
    if not only_mark:
        print(f'indexes_in_replace_indexes{indexes}')
        rng = np.random.RandomState(seed)
        
        new_indexes = rng.choice(list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        dataset.targets[indexes] = dataset.targets[new_indexes]

    else:
        #print(f'indexes_in_replace_indexes{indexes}')
        # Notice the -1 to make class 0 work
        print(f'indexes_type{type(indexes)}')
        print(f'any_indexes_value:{type(indexes[0])}')
        print(indexes)
        for i in indexes:
            dataset.targets[i] = - dataset.targets[i] - 1


def replace_class(dataset: torch.utils.data.Dataset, class_to_replace: List[int], num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):

    indexes = np.array([])
    for itm in class_to_replace:
        indexes = np.concatenate((indexes, np.flatnonzero(np.array(dataset.targets) == itm)))
    indexes = indexes.astype(int)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)

def confuse_class(dataset: torch.utils.data.Dataset, class_to_replace: List[int], shuffle_class_to_replace: List[int], num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):

    '''
    indexes0 = np.flatnonzero(np.array(dataset.targets) == class_to_replace[0])
    indexes0 = indexes0.astype(int)

    indexes1 = np.flatnonzero(np.array(dataset.targets) == class_to_replace[1])
    indexes1 = indexes1.astype(int)

    np.random.seed(seed)
    np.random.shuffle(indexes0)
    np.random.seed(seed)
    np.random.shuffle(indexes1)

    sub_indexes0 = indexes0[:int(len(indexes0)/2)]
    sub_indexes1 = indexes1[:int(len(indexes1)/2)]

    dataset.targets[sub_indexes0] = class_to_replace[1]
    dataset.targets[sub_indexes1] = class_to_replace[0]

    indexes = np.concatenate((sub_indexes0, sub_indexes1))
    '''

    index={}
    sub_index={}
    len_class_to_replace=len(class_to_replace)
    for id in class_to_replace:
        index[id]=np.flatnonzero(np.array(dataset.targets) == class_to_replace[id])
        index[id]=index[id].astype(int)
        np.random.seed(seed)
        np.random.shuffle(index[id])
        sub_index[id]=index[id][:int(len(index[id]/len_class_to_replace))]
      
    indexes=[]
    for id,j in zip(class_to_replace,shuffle_class_to_replace):
        dataset.targets[sub_index[j]]=class_to_replace[id]
        inedxes=np.concatenate((indexes, sub_index[id]))
    print(f'indexes_in_confuse_class_method:{indexes}')
        
        
    replace_indexes(dataset, indexes, seed, only_mark)

def get_loaders(args, dataset_name, class_to_replace: List[int] = None, num_indexes_to_replace: int = None,
                indexes_to_replace: List[int] = None, confuse_mode: bool = False, seed: int = 1, only_mark: bool = False, 
                root: str = None, batch_size=128, shuffle=True, split: str = 'train', noise_mode:str='sym', 
                **dataset_kwargs):
    def unpickle(file):
        import _pickle as cPickle
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo, encoding='latin1')
        return dict
    print(f'in_get_loaders_args.forget_class:{class_to_replace}')
    '''
    :param dataset_name: Name of dataset to use
    :param class_to_replace: If not None, specifies which class to replace completely or partially
    :param num_indexes_to_replace: If None, all samples from `class_to_replace` are replaced. Else, only replace
                                   `num_indexes_to_replace` samples
    :param indexes_to_replace: If not None, denotes the indexes of samples to replace. Only one of class_to_replace and
                               indexes_to_replace can be specidied.
    :param seed: Random seed to sample the samples to replace and to initialize the data loaders so that they sample
                 always in the same order
    :param root: Root directory to initialize the dataset
    :param batch_size: Batch size of data loader
    :param shuffle: Whether train data should be randomly shuffled when loading (test data are never shuffled)
    :param dataset_kwargs: Extra arguments to pass to the dataset init.
    :return: The train_loader and test_loader
    
    only_mark: trueならforget_sampleのラベルを負の数にしてわかるようにする．
               falseならforget_sampleをclean_sampleで上書き．ノイジーサンプルをなしにする．retrain用
    '''
    import numpy as np
    import random
    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')
    train_set, test_set = _DATASETS[dataset_name](root, **dataset_kwargs)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    
    valid_set = copy.deepcopy(train_set)
    rng = np.random.RandomState(seed)
    
    valid_idx=[]
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets==i)[0]
        valid_idx.append(rng.choice(class_idx,int(0.2*len(class_idx)),replace=False))
    valid_idx = np.hstack(valid_idx)
    
    train_idx = list(set(range(len(train_set)))-set(valid_idx))
    
    train_set_copy = copy.deepcopy(train_set)

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    print ("confuse mode:",confuse_mode)
    print ("split mode:", split)
    print(f'pre_confuse_mode:args.forget_class:{class_to_replace}')
    
    #noisy labelの生成
    if confuse_mode:
        '''train_data=[]
        train_label=[]
        if args.dataset=='cifar10': 
            for n in range(1,6):
                dpath = '%scifar-10-batches-py/data_batch_%d'%(args.dataroot,n)
                data_dic = unpickle(dpath)
                train_data.append(data_dic['data'])
                train_label = train_label+data_dic['labels']
            train_data = np.concatenate(train_data)
        elif args.dataset=='cifar100':    
            train_dic = unpickle('%scifar-100-python/train'%args.dataroot)
            train_data = train_dic['data']
            train_label = train_dic['fine_labels']
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))'''
        train_data=train_set.data
        train_label=train_set.targets
        print(train_data.shape, train_label.shape)

        
        if class_to_replace is not None:
            #全部のクラスじゃないときサンプル数を調整しなあかん
            #for class_number in class_to_replace:
                
            #入れ替えるクラスの数
            dataset_sample_num=40000
            num_classes=9
            if args.dataset=='small_cifar5':
                dataset_sample_num=500
                num_classes=4
            elif args.dataset=='cifar100':
                num_classes=99
            replace_class_num=len(class_to_replace)
            rng=np.random.RandomState(seed-1)
            idx=list(range(dataset_sample_num))
            random.shuffle(idx)
            num_noise=args.num_to_forget
            noise_idx=idx[:num_noise]
            #indexをkeyとしてcleanとnoisyのラベルを辞書をvaluesとして持つ辞書
            #ex.{'i':{'clean':2, 'noise':1}}, i∈(index of noisy_sample)
            change_classes={}
            #変更したlabelを格納するリスト，clean_sampleのlabelも含まれる
            noise_labels=[]
            for i in range(dataset_sample_num):
                if i in noise_idx:
                    #change_classesにappendするための辞書
                    clean_and_noise_dict={}
                    #noise_mode次第で変更するlabelの値
                    changed_label=0
                    if noise_mode=='sym':
                        changed_label=random.randint(0, num_classes)
                    #とりあえず写しただけ，多分いじる必要ある
                    elif noise_mode=='asym':
                        changed_label=transition[train_label[i]]
                    noise_labels.append(changed_label)
                    clean_and_noise_dict['clean']=train_label[i]
                    clean_and_noise_dict['noise']=changed_label
                    change_classes[i]=clean_and_noise_dict
                else:
                    noise_labels.append(train_label[i])
            
            
            if args.change_classes is None:
                args.change_classes=change_classes
            else:
                assert args.change_classes==change_classes, '前回作成したchange_classesと異なる'
            train_set.data= train_data
            
            print(f'clean_label{train_set.targets}')
            print(f'noise_label{np.array(noise_labels)}')
            train_set.clean_labels = train_set.targets
            train_set.targets = np.array(noise_labels)
            print(train_set.clean_labels)
            
            indexes=noise_idx
            #print(f'sample_indexes_after_choice_sample_indexes_to_change_noisy_label:{noise_idx}')
            #print(f'noise_sanple_indexes_and_from_clean_to_noise:{change_classes}')



        if split == "train":
            class_to_replace = None
            indexes_to_replace = None
        elif split == "forget":
            class_to_replace = None
            indexes_to_replace = indexes
        print(f'split:{split}')

    #上でclass_to_replace=Noneにするから実行されへん    
    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError("Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        if confuse_mode:
            '''
            if len(class_to_replace) != 2:
                raise ValueError("In the confusion mode, the number of classes should be 2")
            '''
            confuse_class(train_set, class_to_replace, shuffle_class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,\
                      only_mark=only_mark)
            

        else:
            replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,\
                          only_mark=only_mark)
            if num_indexes_to_replace is None:
                test_indexes = np.array([])
                for c in class_to_replace:
                    test_indexes = np.concatenate((test_indexes, np.where(test_set.targets == c)[0]))
                test_indexes = test_indexes.astype(int)
                all_indexes = np.indices(test_set.targets.shape)
                not_indices = np.setxor1d(all_indexes, test_indexes)
                test_set.data = test_set.data[not_indices]
                test_set.targets = test_set.targets[not_indices]
    elif indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace, seed=seed-1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))
    valid_set.clean_labels=valid_set.targets
    test_set.clean_labels=test_set.targets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)


    return train_loader, valid_loader, test_loader




