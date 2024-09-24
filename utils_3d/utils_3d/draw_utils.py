"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os,torch,sys,copy
import numpy as np
import open3d as o3d

cwd = os.getcwd()
sys.path.append(cwd)

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def to_tensor(array):
    """
    Convert array to tensor
    """
    if (not isinstance(array, torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if (not isinstance(tensor, np.ndarray)):
        if (tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def get_inlier(src_pcd, tgt_pcd, gt, inlier_distance_threshold = 0.1):
    """
    Compute inlier with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    rot, trans = to_tensor(gt[:3,:3]), to_tensor(gt[:3,3][:,None])

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)

    dist = torch.norm(src_pcd- tgt_pcd,dim=1)

    c_inlier_where = np.where(to_array(dist < inlier_distance_threshold))

    return tuple(c_inlier_where[0])

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent

def draw_point(pc, weight, title):
    pcd = to_o3d_pcd(pc)
    pc_weight = to_array(weight[:, None].repeat(1, 3))
    pc_color = lighter(get_blue(), 1 - pc_weight)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Point Cloud '+title, width=640, height=520, left=0, top=0)
    vis1.add_geometry(pcd)
    while True:
        vis1.update_geometry(pcd)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
    vis1.destroy_window()


def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, est_tsfm, gt_tsfm, title='', cp=None, inlier_distance_threshold=0.1):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(est_tsfm[:3 , :3]), to_tensor(est_tsfm[:3 , 3][: , None])
    src_overlap = src_overlap[: ,None].repeat(1 ,3).numpy()
    tgt_overlap = tgt_overlap[: ,None].repeat(1 ,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_est_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_est_overlap.transform(est_tsfm)
    tgt_pcd_est_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_est_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_est_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    src_pcd_gt_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_gt_overlap.transform(gt_tsfm)
    tgt_pcd_gt_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_gt_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_gt_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. compute cp line
    if cp is not None:
        translate = [-1.3, -1.5, 0]
        gt_inliers = get_inlier(src_raw[cp[:, 0], :], tgt_raw[cp[:, 1], :], gt_tsfm, inlier_distance_threshold)
        cp_temp = cp + np.array([0, src_raw.shape[0]])
        colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        # cp_temp = cp[gt_inliers, :] + np.array([0, src_raw.shape[0]])
        line_set = o3d.geometry.LineSet()
        src_pcd_cp = to_o3d_pcd(src_raw)
        tgt_pcd_cp = to_o3d_pcd(tgt_raw)
        src_pcd_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_cp.points,tgt_pcd_cp.points]))
        line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    ########################################
    # 4. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)

    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=f'Inferred {title} region for est', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_overlap)
    vis2.add_geometry(tgt_pcd_est_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=640, height=520, left=1280, top=0)
    vis3.add_geometry(src_pcd_est_after)
    vis3.add_geometry(tgt_pcd_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name ='Gt registration', width=640, height=520, left=1280, top=570)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    vis5 = o3d.visualization.Visualizer()
    vis5.create_window(window_name=f'Inferred {title} region for gt', width=640, height=520, left=640, top=570)
    vis5.add_geometry(src_pcd_gt_overlap)
    vis5.add_geometry(tgt_pcd_gt_overlap)

    if cp is not None:
        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=f'cp {title}', width=640, height=520, left=0, top=570)
        vis6.add_geometry(src_pcd_cp)
        vis6.add_geometry(tgt_pcd_cp)
        vis6.add_geometry(line_set)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_overlap)
        vis2.update_geometry(tgt_pcd_est_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_est_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

        vis5.update_geometry(src_pcd_gt_overlap)
        vis5.update_geometry(tgt_pcd_gt_overlap)
        if not vis5.poll_events():
            break
        vis5.update_renderer()

        if cp is not None:
            vis6.update_geometry(src_pcd_cp)
            vis6.update_geometry(tgt_pcd_cp)
            vis6.update_geometry(line_set)
            if not vis6.poll_events():
                break
            vis6.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()
    vis5.destroy_window()
    if cp is not None:
        vis6.destroy_window()


def draw_registration_pcpair(src_raw, tgt_raw, est_tsfm=np.eye(4), title='', cp=None):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_est_after)
    vis1.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_registration_2pcpair(src_raw, tgt_raw, gt_tsfm, est_tsfm, init_tsfm=None, title=''):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    gt_tsfm = to_array(gt_tsfm)
    est_tsfm = to_array(est_tsfm)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)
    if init_tsfm is not None:
        init_tsfm = to_array(init_tsfm)
        src_pcd_est_after_init = copy.deepcopy(src_pcd_before)
        src_pcd_est_after_init.transform(init_tsfm)
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='est final registration', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_after)
    vis2.add_geometry(tgt_pcd_before)

    if init_tsfm is not None:
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name='est init registration', width=640, height=520, left=640, top=570)
        vis3.add_geometry(src_pcd_est_after_init)
        vis3.add_geometry(tgt_pcd_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='gt registration', width=640, height=520, left=1280, top=0)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        if init_tsfm is not None:
            vis3.update_geometry(src_pcd_est_after_init)
            vis3.update_geometry(tgt_pcd_before)
            if not vis3.poll_events():
                break
            vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    if init_tsfm is not None:
        vis3.destroy_window()
    vis4.destroy_window()


def draw_registration_2pcpairmuch(src_raw, tgt_raw, gt_tsfm=None, est1_tsfm=None, est2_tsfm=None, est3_tsfm=None, est4_tsfm=None,
                                  title1='est 1', title2='est 2', title3='est 3', title4='est 4'):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    gt_tsfm = to_array(gt_tsfm)
    if est1_tsfm is not None:
        est1_tsfm = to_array(est1_tsfm)
    if est2_tsfm is not None:
        est2_tsfm = to_array(est2_tsfm)
    if est3_tsfm is not None:
        est3_tsfm = to_array(est3_tsfm)
    if est4_tsfm is not None:
        est4_tsfm = to_array(est4_tsfm)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)
    if est1_tsfm is not None:
        src_pcd_est1_after = copy.deepcopy(src_pcd_before)
        src_pcd_est1_after.transform(est1_tsfm)
    if est2_tsfm is not None:
        src_pcd_est2_after = copy.deepcopy(src_pcd_before)
        src_pcd_est2_after.transform(est2_tsfm)
    if est3_tsfm is not None:
        src_pcd_est3_after = copy.deepcopy(src_pcd_before)
        src_pcd_est3_after.transform(est3_tsfm)
    if est4_tsfm is not None:
        src_pcd_est4_after = copy.deepcopy(src_pcd_before)
        src_pcd_est4_after.transform(est4_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)


    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='gt registration', width=640, height=520, left=0, top=570)
    vis2.add_geometry(src_pcd_gt_after)
    vis2.add_geometry(tgt_pcd_before)

    if est1_tsfm is not None:
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name=title1, width=640, height=520, left=640, top=0)
        vis3.add_geometry(src_pcd_est1_after)
        vis3.add_geometry(tgt_pcd_before)

    if est2_tsfm is not None:
        vis4 = o3d.visualization.Visualizer()
        vis4.create_window(window_name=title2, width=640, height=520, left=1280, top=0)
        vis4.add_geometry(src_pcd_est2_after)
        vis4.add_geometry(tgt_pcd_before)

    if est3_tsfm is not None:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name=title3, width=640, height=520, left=640, top=570)
        vis5.add_geometry(src_pcd_est3_after)
        vis5.add_geometry(tgt_pcd_before)

    if est4_tsfm is not None:
        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=title4, width=640, height=520, left=1280, top=570)
        vis6.add_geometry(src_pcd_est4_after)
        vis6.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_gt_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        if est1_tsfm is not None:
            vis3.update_geometry(src_pcd_est1_after)
            vis3.update_geometry(tgt_pcd_before)
            if not vis3.poll_events():
                break
            vis3.update_renderer()

        if est2_tsfm is not None:
            vis4.update_geometry(src_pcd_est2_after)
            vis4.update_geometry(tgt_pcd_before)
            if not vis4.poll_events():
                break
            vis4.update_renderer()

        if est3_tsfm is not None:
            vis5.update_geometry(src_pcd_est3_after)
            vis5.update_geometry(tgt_pcd_before)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

        if est4_tsfm is not None:
            vis6.update_geometry(src_pcd_est4_after)
            vis6.update_geometry(tgt_pcd_before)
            if not vis6.poll_events():
                break
            vis6.update_renderer()


    vis1.destroy_window()
    vis2.destroy_window()
    if est1_tsfm is not None:
        vis3.destroy_window()
    if est2_tsfm is not None:
        vis4.destroy_window()
    if est3_tsfm is not None:
        vis5.destroy_window()
    if est4_tsfm is not None:
        vis6.destroy_window()


def draw_registration_2pcpair_ds(src_raw, tgt_raw, src_sparse, tgt_sparse, gt_tsfm, est_tsfm=np.eye(4), title='', cp=None):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_sparse_before = to_o3d_pcd(src_sparse)
    tgt_pcd_sparse_before = to_o3d_pcd(tgt_sparse)
    src_pcd_sparse_before.paint_uniform_color(get_yellow())
    tgt_pcd_sparse_before.paint_uniform_color(get_blue())

    ########################################
    # 3. compute cp line
    if cp is not None:
        translate = [-1.3, -1.5, 0]
        gt_inliers = get_inlier(src_sparse[cp[:, 0], :], tgt_sparse[cp[:, 1], :], gt_tsfm)
        est_inliers = get_inlier(src_sparse[cp[:, 0], :], tgt_sparse[cp[:, 1], :], est_tsfm)
        cp_temp = cp + np.array([0, src_sparse.shape[0]])
        gt_colors = [[0, 1, 0] if i in gt_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        est_colors = [[0, 1, 0] if i in est_inliers else [1, 0, 0] for i in range(cp.shape[0])]
        # cp_temp = cp[gt_inliers, :] + np.array([0, src_raw.shape[0]])
        gt_line_set = o3d.geometry.LineSet()
        est_line_set = o3d.geometry.LineSet()
        src_pcd_gt_cp = to_o3d_pcd(src_sparse)
        src_pcd_est_cp = to_o3d_pcd(src_sparse)
        tgt_pcd_cp = to_o3d_pcd(tgt_sparse)
        src_pcd_gt_cp.paint_uniform_color(get_yellow())
        src_pcd_est_cp.paint_uniform_color(get_yellow())
        tgt_pcd_cp.paint_uniform_color(get_blue())
        src_pcd_gt_cp.transform(gt_tsfm)
        src_pcd_est_cp.transform(gt_tsfm)
        tgt_pcd_cp.translate(translate)

        gt_line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_gt_cp.points,tgt_pcd_cp.points]))
        gt_line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        gt_line_set.colors = o3d.utility.Vector3dVector(gt_colors)

        est_line_set.points = o3d.utility.Vector3dVector(np.vstack([src_pcd_est_cp.points,tgt_pcd_cp.points]))
        est_line_set.lines = o3d.utility.Vector2iVector(cp_temp)
        est_line_set.colors = o3d.utility.Vector3dVector(est_colors)

    ########################################
    # 2. draw registrations
    src_pcd_est_after = copy.deepcopy(src_pcd_before)
    src_pcd_est_after.transform(est_tsfm)
    src_pcd_sparse_est_after = copy.deepcopy(src_pcd_sparse_before)
    src_pcd_sparse_est_after.transform(est_tsfm)
    src_pcd_gt_after = copy.deepcopy(src_pcd_before)
    src_pcd_gt_after.transform(gt_tsfm)
    src_pcd_sparse_gt_after = copy.deepcopy(src_pcd_sparse_before)
    src_pcd_sparse_gt_after.transform(gt_tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Before registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='After registration', width=640, height=520, left=640, top=0)
    vis2.add_geometry(src_pcd_est_after)
    vis2.add_geometry(tgt_pcd_before)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='After registration', width=640, height=520, left=1280, top=0)
    vis3.add_geometry(src_pcd_sparse_est_after)
    vis3.add_geometry(tgt_pcd_sparse_before)

    vis4 = o3d.visualization.Visualizer()
    vis4.create_window(window_name='gt registration', width=640, height=520, left=640, top=570)
    vis4.add_geometry(src_pcd_gt_after)
    vis4.add_geometry(tgt_pcd_before)

    if cp is not None:
        vis5 = o3d.visualization.Visualizer()
        vis5.create_window(window_name=f'gt cp', width=640, height=520, left=1280, top=570)
        vis5.add_geometry(src_pcd_gt_cp)
        vis5.add_geometry(tgt_pcd_cp)
        vis5.add_geometry(gt_line_set)

        vis6 = o3d.visualization.Visualizer()
        vis6.create_window(window_name=f'est cp', width=640, height=520, left=0, top=570)
        vis6.add_geometry(src_pcd_est_cp)
        vis6.add_geometry(tgt_pcd_cp)
        vis6.add_geometry(est_line_set)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis1.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_est_after)
        vis2.update_geometry(tgt_pcd_before)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_sparse_est_after)
        vis3.update_geometry(tgt_pcd_sparse_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        vis4.update_geometry(src_pcd_gt_after)
        vis4.update_geometry(tgt_pcd_before)
        if not vis4.poll_events():
            break
        vis4.update_renderer()

        if cp is not None:
            vis5.update_geometry(src_pcd_gt_cp)
            vis5.update_geometry(tgt_pcd_cp)
            vis5.update_geometry(gt_line_set)
            if not vis5.poll_events():
                break
            vis5.update_renderer()

            vis6.update_geometry(src_pcd_est_cp)
            vis6.update_geometry(tgt_pcd_cp)
            vis6.update_geometry(est_line_set)
            if not vis6.poll_events():
                break
            vis6.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()
    vis4.destroy_window()
    if cp is not None:
        vis5.destroy_window()
        vis6.destroy_window()


def draw_registration_s2d_pcpair(src_raw, tgt_raw, src_down, tgt_down, est_tsfm=np.eye(4), title='', cp=None):
    # show those sparse and dense point clouds
    ########################################
    # 1. input point cloud
    src_pcd_raw_before = to_o3d_pcd(src_raw)
    tgt_pcd_raw_before = to_o3d_pcd(tgt_raw)
    src_pcd_raw_before.paint_uniform_color(lighter(get_yellow(), 0.5))
    tgt_pcd_raw_before.paint_uniform_color(lighter(get_blue(), 0.9))
    src_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_before = to_o3d_pcd(src_down)
    tgt_pcd_down_before = to_o3d_pcd(tgt_down)
    src_pcd_down_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_before.paint_uniform_color(get_blue_down())
    src_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_raw_est_after = copy.deepcopy(src_pcd_raw_before)
    src_pcd_raw_est_after.transform(est_tsfm)
    src_pcd_down_est_after = copy.deepcopy(src_pcd_down_before)
    src_pcd_down_est_after.transform(est_tsfm)


    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_raw_est_after)
    vis1.add_geometry(tgt_pcd_raw_before)
    vis1.add_geometry(src_pcd_down_est_after)
    vis1.add_geometry(tgt_pcd_down_before)

    while True:
        vis1.update_geometry(src_pcd_raw_est_after)
        vis1.update_geometry(tgt_pcd_raw_before)
        vis1.update_geometry(src_pcd_down_est_after)
        vis1.update_geometry(tgt_pcd_down_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def draw_registration_sphere_pcpair(src_raw, tgt_raw, src_down, tgt_down, src_down_sp, tgt_down_sp, est_tsfm=np.eye(4), title='', cp=None):
    # show those sparse and dense point clouds
    ########################################
    # 1. input point cloud
    src_pcd_raw_before = to_o3d_pcd(src_raw)
    tgt_pcd_raw_before = to_o3d_pcd(tgt_raw)
    src_pcd_raw_before.paint_uniform_color(get_yellow())
    tgt_pcd_raw_before.paint_uniform_color(get_blue())
    src_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_raw_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_before = to_o3d_pcd(src_down)
    tgt_pcd_down_before = to_o3d_pcd(tgt_down)
    src_pcd_down_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_before.paint_uniform_color(get_blue_down())
    src_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_pcd_down_sp_before = to_o3d_pcd(src_down_sp)
    tgt_pcd_down_sp_before = to_o3d_pcd(tgt_down_sp)
    src_pcd_down_sp_before.paint_uniform_color(get_yellow_down())
    tgt_pcd_down_sp_before.paint_uniform_color(get_blue_down())
    src_pcd_down_sp_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_down_sp_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. draw registrations
    src_pcd_raw_est_after = copy.deepcopy(src_pcd_raw_before)
    src_pcd_raw_est_after.transform(est_tsfm)
    src_pcd_down_est_after = copy.deepcopy(src_pcd_down_before)
    src_pcd_down_est_after.transform(est_tsfm)
    src_pcd_down_sp_est_after = copy.deepcopy(src_pcd_down_sp_before)
    src_pcd_down_sp_est_after.transform(est_tsfm)


    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='registration', width=640, height=520, left=0, top=0)
    vis1.add_geometry(src_pcd_raw_est_after)
    vis1.add_geometry(tgt_pcd_raw_before)
    vis1.add_geometry(src_pcd_down_est_after)
    vis1.add_geometry(tgt_pcd_down_before)
    vis1.add_geometry(src_pcd_down_sp_est_after)
    vis1.add_geometry(tgt_pcd_down_sp_before)

    while True:
        vis1.update_geometry(src_pcd_raw_est_after)
        vis1.update_geometry(tgt_pcd_raw_before)
        vis1.update_geometry(src_pcd_down_est_after)
        vis1.update_geometry(tgt_pcd_down_before)
        vis1.update_geometry(src_pcd_down_sp_est_after)
        vis1.update_geometry(tgt_pcd_down_sp_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()


def show_pointcloud(pc, size=10, theta_1=0, theta_2=0, c0=[1,0,0], weights=None, savename=None, close=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if type(weights) == torch.Tensor:
        weights = weights.detach().cpu().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    if weights is not None:
        weights_pd = pd.Series(weights)
        c0 = weights_pd.apply(lambda x: (c0[0], c0[1], c0[2], x)).tolist()
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, c=c0)
    plt.axis("off")
    ax.view_init(theta_1, theta_2)
    ax.auto_scale_xyz([-1,1],[-1,1],[-1,1])
    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
        plt.margins(0, 0, 0)
        plt.savefig(savename, format='svg', bbox_inches='tight', transparent=True, dpi=600)
        # plt.show()
        # if close:
        #     plt.close(fig)
    # legend = ax.legend()
    # frame = legend.get_frame()
    # frame.set_alpha(1)
    # frame.set_facecolor('none')