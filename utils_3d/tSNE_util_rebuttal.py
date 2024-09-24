import numpy as np
import torch


def t_SNE(feat_train, feat_val, label_train, label_val, perplexity=[10,20,30,50], fig_name=''):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TKAgg')
    # class_colors = np.random.random([40, 3])
    class_colors = ['black', 'tomato', 'yellow', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru',
                    'olivedrab', 'hotpink', 'crimson', 'burlywood', 'aquamarine', 'blueviolet', 'brown', 'darkviolet', 'deeppink', 'deepskyblue',
                    'dimgray', 'indigo', 'dodgerblue', 'chocolate',  'slategrey', 'silver', 'springgreen', 'steelblue', 'tan', 'teal',
                    'thistle', 'maroon', 'turquoise', 'cadetblue', 'wheat', 'khaki', 'whitesmoke', 'orchid', 'yellowgreen', 'navy']
    class_names =['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                  'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                  'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                  'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
                  'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    # filter_list = [1,2,8,12,14,22,23,30,33,35]
    filter_list = list(np.arange(0, np.max(label_train)+1))
    train_len = feat_train.shape[0]
    feat_all=np.concatenate([feat_train, feat_val], axis=0)
    label_all=np.concatenate([label_train, label_val], axis=0)
    import random
    # t_sne_randint = random.randint(0, 1000)
    t_sne_randint = 0
    print('rand int value:', t_sne_randint)
    for pp in perplexity:
        print('pp:', pp)
        X_tsne = TSNE(n_components=2, random_state=t_sne_randint, perplexity=pp).fit_transform(feat_all)
        X_tsne_dataset_i = X_tsne[:train_len, :]
        y_dataset_i = label_all[:train_len]
        plt.figure()
        for class_i, color_i, name_i in zip(range(40), class_colors, class_names):
            if len(filter_list)>0 and class_i in filter_list:
                plt.scatter(X_tsne_dataset_i[y_dataset_i == class_i, 0],
                            X_tsne_dataset_i[y_dataset_i == class_i, 1],
                            c=color_i, label=name_i,s=5)
        plt.title(f'train 40 {str(pp)} t-SNE')
        plt.legend()
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        # plt.show()

        # plt.figure()
        # for class_i, color_i, name_i in zip(range(10), class_colors, class_names):
        #     if len(filter_list) > 0 and class_i in filter_list:
        #         plt.scatter(X_tsne_dataset_i[y_dataset_i == class_i, 0],
        #                     X_tsne_dataset_i[y_dataset_i == class_i, 1],
        #                     c=color_i, label=name_i,s=50)
        # plt.title(f'train 10 {str(pp)} t-SNE')
        # plt.legend()
        # plt.show()

        X_tsne_dataset_i = X_tsne[train_len:, :]
        y_dataset_i = label_all[train_len:]
        plt.figure()
        for class_i, color_i, name_i in zip(range(40), class_colors, class_names):
            if len(filter_list) > 0 and class_i in filter_list:
                plt.scatter(X_tsne_dataset_i[y_dataset_i == class_i, 0],
                            X_tsne_dataset_i[y_dataset_i == class_i, 1],
                            c=color_i, label=name_i,s=5)
        # plt.title(f'val 40 {str(pp)} t-SNE')
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        num1 = 1.05
        num2 = 0
        num3 = 3
        num4 = 0
        plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4, ncol=2)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        plt.savefig('fig/'+fig_name+str(pp)+'.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)

        # plt.figure()
        # for class_i, color_i, name_i in zip(range(10), class_colors, class_names):
        #     if len(filter_list) > 0 and class_i in filter_list:
        #         plt.scatter(X_tsne_dataset_i[y_dataset_i == class_i, 0],
        #                     X_tsne_dataset_i[y_dataset_i == class_i, 1],
        #                     c=color_i, label=name_i,s=50)
        # plt.title(f'val 10 {str(pp)} t-SNE')
        # plt.legend()
        # plt.show()
    return 0



