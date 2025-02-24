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
