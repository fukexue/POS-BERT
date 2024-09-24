import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from visdom import Visdom

def get_first_para(model):
    j = 0
    for i in model.parameters():
        j = j + 1
        if j == 1:
            return(i.abs().sum())

def show_pointcloud(pc, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, alpha=0.5)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

def save_pointcloud(pc, figname = None, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.axis("off")
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, alpha=0.5)
    ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    fig.savefig(figname+'.png', transparent=True)

def show_pointcloud_batch(pc, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if pc.shape[1]==3:
        pc = pc.transpose(0,2,1)
    B,N,C = pc.shape
    fig = plt.figure()
    for i in range(B):
        ax = fig.add_subplot(2, int(B/2), i+1, projection='3d')
        ax.scatter(pc[i, :, 0], pc[i, :, 1], pc[i, :, 2], s=size, alpha=0.5)
        ax.auto_scale_xyz([-1, 1],[-1, 1],[-1, 1])


def show_pointcloud2d(pc ,size=10):
    if type(pc) == torch.Tensor:
        pc = pc.numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pc[:,0], pc[:,1],s=size)


def show_distribution(features):
    # features should be B*C*N
    if type(features) == torch.Tensor:
        features = features.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(features)
    # B,C,N = features.shape
    # fig = plt.figure()
    # idx_B = 0 # only show the first batch
    # num_bin = 10
    # for idx_C in range(C):
    #     ax = fig.add_subplot(2, int(C/2), idx_C+1)
    #     bin = np.linspace(features[idx_B, idx_C, :].min(),features[idx_B, idx_C, :].min(), num_bin)
    #     ax.hist(features[idx_B,idx_C,:])#, bins=bin)


def show_pointcloud_2pc(pc_1, pc_2, ax=None, c1='r', c2='b',s1=1, s2=1):
    if type(pc_1) == torch.Tensor:
        pc_1 = pc_1.cpu().detach().numpy()
        pc_2 = pc_2.cpu().detach().numpy()
    if pc_1.shape[0]==3:
        pc_1 = pc_1.transpose()
        pc_2 = pc_2.transpose()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc_1[:, 0], pc_1[:, 1], pc_1[:, 2], s=s1, c=c1, alpha=0.5)
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2], s=s2, c=c2, alpha=0.5)


def show_pointcloudVSvoxel(pc, voxels, ax=None):
    if type(pc) == torch.Tensor:
        pc = pc.cpu().detach().numpy()
    if type(voxels) == torch.Tensor:
        voxels = voxels.cpu().detach().numpy()
    on_voxels = np.where(voxels>=1)
    x_min = on_voxels[0].min()-1
    x_max = on_voxels[0].max()+1
    y_min = on_voxels[1].min()-1
    y_max = on_voxels[1].max()+1
    z_min = on_voxels[2].min()-1
    z_max = on_voxels[2].max()+1
    pc_local = pc-np.array([x_min, y_min, z_min])
    print(pc_local.min(0))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc_local[:,0], pc_local[:,1], pc_local[:,2],s=1)
    ax.voxels(voxels[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], alpha=0.5)


def show_voxels(voxels, facecolors=None):
    if type(voxels) == torch.Tensor:
        voxels = voxels.cpu().detach().numpy()
    on_voxels = np.where(voxels>=1)
    x_min = on_voxels[0].min()
    x_max = on_voxels[0].max()
    y_min = on_voxels[1].min()
    y_max = on_voxels[1].max()
    z_min = on_voxels[2].min()
    z_max = on_voxels[2].max()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if facecolors is None:
        ax.voxels(voxels[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1],alpha=0.5)
    else:
        ax.voxels(voxels[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], facecolors=facecolors,alpha=0.5)


def show_voxels_2vox(vox1, vox2):
    if type(vox1) == torch.Tensor:
        vox1 = vox1.cpu().detach().numpy()
    if type(vox2) == torch.Tensor:
        vox2 = vox2.cpu().detach().numpy()
    print("Dice between 2 voxels",get_dice(vox1, vox2))
    on_voxels = np.where((vox1+vox2) >= 1)
    x_min = on_voxels[0].min()-1
    x_max = on_voxels[0].max()+1
    y_min = on_voxels[1].min()-1
    y_max = on_voxels[1].max()+1
    z_min = on_voxels[2].min()-1
    z_max = on_voxels[2].max()+1

    vox1_local = vox1[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    vox2_local = vox2[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    voxel_common12 = (vox1==1)|(vox2==1)
    voxel_common12_local = voxel_common12[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    colors12 = np.empty(voxel_common12_local.shape, dtype=object)
    colors12[vox1_local == 1] = 'green'
    colors12[vox2_local == 1] = 'red'
    colors12[(vox1_local == 1) & (vox2_local == 1)] = 'yellow'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.voxels(voxel_common12_local.transpose(1,2,0), facecolors=colors12.transpose(1,2,0), alpha=0.5)


def show_pointcloud_perpointcolor(pc, size=10,c='r'):
    # pc.shape = Nx3, c.shape = N
    if type(pc) == torch.Tensor:
        pc = pc.cpu().detach().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    if type(c) == torch.Tensor:
        c = c.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax0 = ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, alpha=0.5,c=c)
    plt.colorbar(ax0, ax=ax)


def show_pointcloud_addquiver(pc, q, ax=None, size=10, alpha_p=0.5, alpha_q=0.3, c='royalblue'):
    # pc.shape = Nx3(or 3xN), c.shape = Nx3(or 3xN)
    if type(pc) == torch.Tensor:
        pc = pc.cpu().detach().numpy()
        if pc.shape[0]==3:
            pc = pc.transpose()
    if type(q) == torch.Tensor:
        q = q.cpu().detach().numpy()
        if q.shape[0]==3:
            q = q.transpose()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size,c=c, alpha=alpha_p)
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    u = q[:,0]
    v = q[:,1]
    w = q[:,2]
    ax.quiver(x,y,z,u,v,w, normalize=False,linewidths=1, alpha=alpha_q, colors=c)


def EuclideanDistances(A, B):
    if (type(A) == torch.Tensor) & (type(B) == torch.Tensor):
        BT = B.transpose(1,0)
        verProd = torch.matmul(A, BT)
        SqA = A ** 2
        sumSqA = SqA.sum(1).unsqueeze(0)
        sumSqAEx = sumSqA.transpose(1,0).repeat(1,verProd.shape[1])

        SqB = B ** 2
        sumSqB = SqB.sum(1)
        sumSqBEx = sumSqB.repeat(verProd.shape[0], 1)
        SqED = sumSqBEx + sumSqAEx - 2*verProd
        # SqED[SqED<0] = 0.0
        torch.clamp(SqED, min=1e-4)
        ED = torch.sqrt(SqED)
        return ED
        # return SqED

    if (type(A) != torch.Tensor) & (type(B) != torch.Tensor):
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A, BT)
        # print(vecProd)
        SqA = A ** 2
        # print(SqA)
        sumSqA = np.expand_dims(np.sum(SqA, axis=1), 0)
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B ** 2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
        SqED = sumSqBEx + sumSqAEx - 2 * vecProd
        SqED[SqED < 0] = 0.0
        ED = np.sqrt(SqED)
        return ED


def get_dice(vox1, vox2):
    if type(vox1) == np.ndarray:
        vox1 = torch.from_numpy(vox1)
    if type(vox2) == np.ndarray:
        vox2 = torch.from_numpy(vox2)
    return 2*torch.sum(vox1*vox2)/(vox1.sum() + vox2.sum().type(torch.float32))


class Check_layer_parameters(object):
    def __init__(self, model):
        self.layer_names = []
        self.layer_paras = []
        for i in model.named_parameters():
            self.layer_names.append( i[0] )
            self.layer_paras.append( i[-1].cpu().detach().numpy() )
    def print_idx_name(self):
        for i in range(len(self.layer_names)):
            print(i, self.layer_names[i])
    def plot_hist(self, idx_layer):
        fig = plt.figure()
        for i in range(len(idx_layer)):
            ax = fig.add_subplot(len(idx_layer),1,i+1)
            ax.hist(self.layer_paras[idx_layer[i]])
            ax.set_title(self.layer_names[idx_layer[i]])
        plt.plot()



def check_layer_parameters(model, layer_idx = None):
    for layer in model.named_parameters():
        if layer in layer_idx:
            print(layer[layer][0])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.scatters = {}
        self.scatters2D = {}
        self.images_dict = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def scatter(self, var_name, split_name, title_name, x, size=10, color=0, symbol='dot'):
        if var_name not in self.scatters:
            self.scatters[var_name] = self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                markersize=size,
                markercolor=color,
                markerborderwidth=0,
                # opacity=0.5
                # markersymbol=symbol,
                # linecolor='white',
            ))
        else:
            self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, win=self.scatters[var_name], name=split_name, update='replace')

    def scatter2D(self, var_name, split_name, title_name, x, size=1, color=[255,0,0]):
        if var_name not in self.scatters2D:
            self.scatters2D[var_name] = self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                markersize=size,
                markercolor=np.array([color])
            ))
        else:
            self.viz.scatter(X=x.cpu().detach().numpy(), env=self.env, win=self.scatters2D[var_name], name=split_name, update='update', opts=dict(
                markersize=size,
                markercolor=np.array([color])
            ))

    def images(self, var_name, split_name, title_name, x, nrow=2):
        if var_name not in self.images_dict:
            self.images_dict[var_name] = self.viz.images(tensor=x, env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                nrow=nrow,
            ))
        else:
            self.viz.images(tensor=x, env=self.env, win=self.images_dict[var_name], nrow=nrow)

    def image_3D(self, x, win='image_3d',title='image_3d'):
        for i in range(x.shape[0]):
            self.viz.image(img=x[i,:,:], win=win, env=self.env, opts=dict(
                caption='Z: '+str(i),
                store_history=True,
                title=title
            ))


def get_color(L=32, H=32):
    '''
    get a LxHx3 rgb color matirx
    '''
    from colour import Color

    red = Color("red")
    blue = Color("blue")
    green = Color("lime")
    yellow = Color("yellow")
    r2b = list(red.range_to(blue, L))
    r2g = list(red.range_to(green, L))

    r2b2g_rgb = []
    for i in r2b:
        r2b_i = list(i.range_to(green, H))
        r2b_i_rgb = [i.rgb for i in r2b_i]
        r2b2g_rgb.append(r2b_i_rgb)
    grid_color_1 = np.array(r2b2g_rgb)

    r2b2y_rgb = []
    for i in r2b:
        r2b_i = list(i.range_to(yellow, H))
        r2b_i_rgb = [i.rgb for i in r2b_i]
        r2b2y_rgb.append(r2b_i_rgb)
    grid_color_2 = np.array(r2b2y_rgb)

    hslmodel_rbg = np.zeros((L,H,3))
    for i in range(L):
        for j in range(H):
            hslmodel_rbg[i,j,:] = np.asarray(Color(hsl=(1, i/L, j/H)).get_rgb())
    grid_color_3 = hslmodel_rbg

    return grid_color_2


if __name__ == '__main__':
    A = torch.rand(1024,3)
    B = torch.rand(2048,3)
    dist_torch = EuclideanDistances(A, B)
    A_np = A.numpy()
    B_np = B.numpy()
    dist_numpy = EuclideanDistances(A_np, B_np)
    print(np.sum(dist_torch.numpy()-dist_numpy))
