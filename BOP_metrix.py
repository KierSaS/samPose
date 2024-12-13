import cv2
import numpy as np
from transforms3d.axangles import mat2axangle
from transforms3d.euler import mat2euler
from transforms3d.quaternions import quat2mat
from tqdm import tqdm
from utils.base_utils import transformation_inverse_2d, project_points, transformation_apply_2d, hpts_to_pts, \
    pts_to_hpts, transformation_decompose_2d, angle_to_rotation_2d, look_at_rotation, transformation_offset_2d, \
    transformation_compose_2d, transformation_scale_2d, transformation_rotation_2d, pose_inverse, transform_points_pose, pose_apply
import torch

def estimate_pose_from_similarity_transform(ref_pose, ref_K, que_K, M_que_to_ref, object_center):
    # todo: here we assume the scale is approximately correct, even for the rectified version
    M_ref_to_que = transformation_inverse_2d(M_que_to_ref) # from reference to query
    ref_cam = (-ref_pose[:,:3].T @ ref_pose[:,3:])[...,0]
    ref_obj_center, _ = project_points(object_center[None,:],ref_pose,ref_K)
    # ref_obj_center = ref_obj_center[0]
    que_obj_center = transformation_apply_2d(M_ref_to_que, ref_obj_center)[0]
    que_obj_center_ = hpts_to_pts(pts_to_hpts(que_obj_center[None]) @ np.linalg.inv(que_K).T)[0]  # normalized
    scale, rotation, _ = transformation_decompose_2d(M_ref_to_que)

    # approximate depth
    que_f = (que_K[0,0]+que_K[1,1])/2
    ref_f = (ref_K[0,0]+ref_K[1,1])/2
    que_obj_center__ = que_obj_center_ * que_f
    que_f_ = np.sqrt(que_f ** 2 + np.linalg.norm(que_obj_center__,2)**2)
    ref_dist = np.linalg.norm(ref_cam - object_center)
    que_dist = ref_dist * que_f_ / ref_f / scale
    # que_cam = object_center + (ref_cam - object_center) / ref_dist * que_dist
    que_obj_center___ = pts_to_hpts(que_obj_center_[None])[0]
    que_cen3d = que_obj_center___ / np.linalg.norm(que_obj_center___)  * que_dist
    # que_cen3d = R @ object_center + t

    ref_rot = ref_pose[:,:3]
    R0 = np.eye(3)
    R0[:2,:2] = angle_to_rotation_2d(rotation)

    # x_, y_ = que_obj_center_
    # R1 = euler2mat(-np.arctan2(x_, 1),0,0,'syxz')
    # R2 = euler2mat(np.arctan2(y_, 1),0,0,'sxyz')
    R = look_at_rotation(que_obj_center_)
    # print(R2 @ R1 @ pts_to_hpts(que_obj_center_[None])[0])
    # que_rot = R1.T @ R2.T @ (R0 @ ref_rot)
    que_rot = R.T @ (R0 @ ref_rot)
    que_trans = que_cen3d - que_rot @ object_center
    return np.concatenate([que_rot, que_trans[:,None]], 1)

def let_me_look_at(pose, K, obj_center):
    image_center, _ = project_points(obj_center[None, :], pose, K)
    return let_me_look_at_2d(image_center[0], K)

def let_me_look_at_2d(image_center, K):
    f_raw = (K[0, 0] + K[1, 1]) / 2
    image_center = image_center - K[:2, 2]
    f_new = np.sqrt(np.linalg.norm(image_center, 2, 0) ** 2 + f_raw ** 2)
    image_center_ = image_center / f_raw
    R_new = look_at_rotation(image_center_)
    return R_new, f_new

def scale_rotation_difference_from_cameras(ref_poses, que_poses, ref_Ks, que_Ks, center):
    """
    relative scale and rotation from ref to que (apply M on ref to get que)
    @param ref_poses:
    @param que_poses:
    @param ref_Ks:
    @param que_Ks:
    @param center:
    @return:
    """
    que_rot, que_f = [], []
    for qi in range(que_poses.shape[0]):
        R, f = let_me_look_at(que_poses[qi],que_Ks[qi],center)
        que_rot.append(R @ que_poses[qi,:,:3])
        que_f.append(f)
    que_rot = np.stack(que_rot,0)
    que_f = np.asarray(que_f)

    ref_rot, ref_f = [], []
    for qi in range(ref_poses.shape[0]):
        R, f = let_me_look_at(ref_poses[qi],ref_Ks[qi],center)
        ref_rot.append(R @ ref_poses[qi,:,:3])
        ref_f.append(f)
    ref_rot = np.stack(ref_rot,0)
    ref_f = np.asarray(ref_f)

    ref_cam = (-ref_poses[:, :, :3].transpose([0, 2, 1]) @ ref_poses[:, :, 3:])[..., 0]  # rfn,3
    que_cam = (-que_poses[:, :, :3].transpose([0, 2, 1]) @ que_poses[:, :, 3:])[..., 0]  # qn,3
    ref_dist = np.linalg.norm(ref_cam - center[None, :], 2, 1)  # rfn
    que_dist = np.linalg.norm(que_cam - center[None, :], 2, 1)  # qn

    scale_diff = ref_dist / que_dist * que_f / ref_f

    # compute relative rotation
    # from ref to que
    rel_rot = que_rot @ ref_rot.transpose([0, 2, 1])  # qn, 3, 3
    angle_diff = []
    for qi in range(rel_rot.shape[0]):
        angle, _, _ = mat2euler(rel_rot[qi], 'szyx')
        angle_diff.append(angle)
    angle_diff = np.asarray(angle_diff)

    return scale_diff, angle_diff

def estimate_pose_from_similarity_transform_compose(position, scale_r2q, angle_r2q, ref_pose, ref_K, que_K, object_center):
    ref_cen = project_points(object_center[None],ref_pose,ref_K)[0][0]
    M_q2r = transformation_offset_2d(-position[0], -position[1])
    M_q2r = transformation_compose_2d(M_q2r, transformation_scale_2d(1 / scale_r2q))
    M_q2r = transformation_compose_2d(M_q2r, transformation_rotation_2d(-angle_r2q))
    M_q2r = transformation_compose_2d(M_q2r, transformation_offset_2d(ref_cen[0], ref_cen[1]))
    pose_pr = estimate_pose_from_similarity_transform(ref_pose, ref_K, que_K, M_q2r, object_center)
    return pose_pr

def estimate_pose_from_refinement(context_info, refine_info, ref_pose, ref_K, que_K, object_center):
    context_position, context_scale_r2q, context_angle_r2q, warp_M = \
        context_info['position'], context_info['scale_r2q'], context_info['angle_r2q'], context_info['warp_M']
    offset_r2c, scale_r2c, rot_r2c = refine_info['offset_r2q'], refine_info['scale_r2q'], refine_info['rot_r2q']
    ref_cen = project_points(object_center[None], ref_pose, ref_K)[0][0]

    # find the corrected center
    cen_pr = ref_cen + offset_r2c
    cen_pr = transformation_apply_2d(transformation_inverse_2d(warp_M), cen_pr[None, :])[0]  # coordinate on original image

    rect_R, rect_f = let_me_look_at_2d(cen_pr, que_K)
    scale_r2q = context_scale_r2q * scale_r2c

    # compute the camera from scale
    ref_f = (ref_K[0, 0] + ref_K[1, 1]) / 2
    que_f = rect_f
    ref_cam = pose_inverse(ref_pose)[:, 3]
    ref_dist = np.linalg.norm(ref_cam - object_center)
    que_dist = ref_dist * que_f / ref_f / scale_r2q
    obejct_dir = pts_to_hpts(cen_pr[None]) @ np.linalg.inv(que_K).T
    obejct_dir /= np.linalg.norm(obejct_dir,2,1,True)
    que_cam_ = (obejct_dir * que_dist)[0]

    # compute the rotation
    ref_R = ref_pose[:, :3]
    rot_r2c = quat2mat(rot_r2c)
    rot_sel = np.asarray([[np.cos(context_angle_r2q), -np.sin(context_angle_r2q), 0],
                          [np.sin(context_angle_r2q), np.cos(context_angle_r2q), 0], [0, 0, 1]], np.float32)
    que_R = rect_R.T @ rot_sel @ rot_r2c @ ref_R

    # compute the translation
    # que_t = -que_R @ que_cam
    que_t = que_cam_[:,None] - que_R @ object_center[:,None]
    pose_pr = np.concatenate([que_R, que_t], 1)
    return pose_pr

def compute_pose_errors(object_pts, pose_pr, pose_gt, K):
    # eval projection errors
    pts2d_pr, _ = project_points(object_pts, pose_pr, K)
    pts2d_gt, _ = project_points(object_pts, pose_gt, K)
    prj_err = np.mean(np.linalg.norm(pts2d_pr - pts2d_gt, 2, 1))

    # eval 3D pts errors
    pts3d_pr = transform_points_pose(object_pts, pose_pr)
    pts3d_gt = transform_points_pose(object_pts, pose_gt)
    obj_err = np.mean(np.linalg.norm(pts3d_pr - pts3d_gt, 2, 1))

    # eval pose errors
    dr = pose_pr[:3,:3] @ pose_gt[:3,:3].T
    try:
        _, dr = mat2axangle(dr)
    except ValueError:
        print(dr)
        dr = np.pi
    cam_pr = -pose_pr[:3,:3].T @ pose_pr[:3,3:]
    cam_gt = -pose_gt[:3,:3].T @ pose_gt[:3,3:]
    dt = np.linalg.norm(cam_pr - cam_gt)
    pose_err = np.asarray([np.abs(dr),dt])
    return prj_err, obj_err, pose_err

#############新的指标mssd#####
import torch


def compute_mssd(object_pts, pose_pr, pose_gt, symmetry_matrices=None):
    """
    GPU-accelerated Maximum Symmetry-Aware Surface Distance (MSSD).
    """
    # Convert data to PyTorch tensors
    object_pts = torch.tensor(object_pts, dtype=torch.float32).cuda()
    pose_pr = torch.tensor(pose_pr, dtype=torch.float32).cuda()
    pose_gt = torch.tensor(pose_gt, dtype=torch.float32).cuda()

    # Transform points using predicted pose
    pts_pr = torch.matmul(object_pts, pose_pr[:3, :3].T) + pose_pr[:3, 3]

    # Default: no symmetry
    if symmetry_matrices is None:
        symmetry_matrices = [torch.eye(4, dtype=torch.float32).cuda()]

    min_mssd = float('inf')
    for S in symmetry_matrices:
        S = torch.tensor(S, dtype=torch.float32).cuda()
        # Apply symmetry transformation
        pts_gt = torch.matmul(object_pts, (pose_gt[:3, :3] @ S[:3, :3]).T) + (pose_gt[:3, 3] + S[:3, 3])

        # Compute pairwise distances
        distances = torch.cdist(pts_pr, pts_gt)
        max_dist = torch.max(torch.min(distances, dim=1)[0])
        min_mssd = min(min_mssd, max_dist.item())
    return min_mssd

def compute_mspd(object_pts, pose_pr, pose_gt, K, symmetry_matrices=None):
    """
    GPU-accelerated Maximum Symmetry-Aware Projection Distance (MSPD).
    """
    # Convert data to PyTorch tensors
    object_pts = torch.tensor(object_pts, dtype=torch.float32).cuda()
    pose_pr = torch.tensor(pose_pr, dtype=torch.float32).cuda()
    pose_gt = torch.tensor(pose_gt, dtype=torch.float32).cuda()
    K = torch.tensor(K, dtype=torch.float32).cuda()

    # Project predicted points
    pts_pr = torch.matmul(object_pts, pose_pr[:3, :3].T) + pose_pr[:3, 3]
    pts2d_pr = torch.matmul(pts_pr, K.T)
    pts2d_pr = pts2d_pr[:, :2] / pts2d_pr[:, 2:3]

    # Default: no symmetry
    if symmetry_matrices is None:
        symmetry_matrices = [torch.eye(4, dtype=torch.float32).cuda()]

    min_mspd = float('inf')
    for S in symmetry_matrices:
        S = torch.tensor(S, dtype=torch.float32).cuda()
        # Apply symmetry transformation
        pts_gt = torch.matmul(object_pts, (pose_gt[:3, :3] @ S[:3, :3]).T) + (pose_gt[:3, 3] + S[:3, 3])
        pts2d_gt = torch.matmul(pts_gt, K.T)
        pts2d_gt = pts2d_gt[:, :2] / pts2d_gt[:, 2:3]

        # Compute pairwise 2D distances
        distances = torch.cdist(pts2d_pr, pts2d_gt)
        max_dist = torch.max(torch.min(distances, dim=1)[0])
        min_mspd = min(min_mspd, max_dist.item())
    return min_mspd

def compute_vsd(object_pts, pose_pr, pose_gt, K, tau=0.5, img_size=(480, 640)):

    # Convert data to PyTorch tensors (GPU)
    object_pts = torch.tensor(object_pts, dtype=torch.float32).cuda()
    pose_pr = torch.tensor(pose_pr, dtype=torch.float32).cuda()
    pose_gt = torch.tensor(pose_gt, dtype=torch.float32).cuda()
    K = torch.tensor(K, dtype=torch.float32).cuda()

    # Generate depth maps
    depths_pr = generate_depth_map_gpu(object_pts, pose_pr, K, img_size)
    depths_gt = generate_depth_map_gpu(object_pts, pose_gt, K, img_size)

    # Compute visibility mask (intersection of visible areas)
    visibility_mask = (depths_gt > 0) & (depths_pr > 0)

    # Compute depth difference
    depth_diff = torch.abs(depths_pr - depths_gt)

    # Apply VSD formula using tau
    vsd_values = torch.where(depth_diff[visibility_mask] < tau, 0, 1)  # Binary error
    vsd = torch.mean(vsd_values.float())  # Average over all visible pixels

    return vsd.item()

def generate_depth_map_gpu(object_pts, pose, K, img_size):
    # Transform points to camera coordinates
    pts_cam = torch.matmul(object_pts, pose[:3, :3].T) + pose[:3, 3]
    depths = pts_cam[:, 2]  # Z-coordinate represents depth

    # Project points to image plane
    pts_img = torch.matmul(pts_cam, K.T)
    pts2d = pts_img[:, :2] / pts_img[:, 2:3]

    # Initialize depth map
    depth_map = torch.full(img_size, float('inf'), dtype=torch.float32).cuda()

    # Round 2D points to nearest pixel
    pts2d = torch.round(pts2d).long()

    # Filter valid points (within image bounds)
    valid_mask = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < img_size[1]) & \
                 (pts2d[:, 1] >= 0) & (pts2d[:, 1] < img_size[0])
    pts2d, depths = pts2d[valid_mask], depths[valid_mask]

    # Populate depth map (keeping the minimum depth for each pixel)
    for pt, depth in zip(pts2d, depths):
        y, x = pt[1], pt[0]
        depth_map[y, x] = min(depth_map[y, x], depth)

    # Replace infinite values with 0 (background)
    depth_map[depth_map == float('inf')] = 0

    return depth_map



#############  新的指标mspd  ######

def compute_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def compute_metrics_impl(object_pts, diameter, pose_gt_list, pose_pr_list, Ks, img_size, scale=1.0, symmetric=False):
    """
    Computes metrics including VSD, MSSD, and MSPD, and returns recall values for AR calculation.
    """
    # Initialize storage for errors and AR-related lists
    vsd_recalls, mssd_recalls, mspd_recalls = [], [], []

    # Threshold ranges for AR computation
    vsd_thresholds = [0.05 * diameter * i for i in range(1, 11)]  # VSD thresholds: 0.05D to 0.5D
    mssd_thresholds = [0.05 * diameter * i for i in range(1, 11)]  # MSSD thresholds: 0.05D to 0.5D
    r = img_size[1] / 640  # Compute `r` based on image width
    mspd_thresholds = [5 * r * i for i in range(1, 11)]  # MSPD thresholds: 5r to 50r

    # Iterate over all poses
    for pose_gt, pose_pr, K in tqdm(zip(pose_gt_list, pose_pr_list, Ks), total=len(pose_gt_list)):
        # Compute VSD
        # vsd_errors = []
        # for tau in vsd_thresholds:
        #     vsd_errors.append(compute_vsd(object_pts, pose_pr, pose_gt, K, tau=tau))
        #
        # vsd_recalls.append([1 if err < tau else 0 for err, tau in zip(vsd_errors, vsd_thresholds)])

        # Compute MSSD
        mssd_error = compute_mssd(object_pts, pose_pr, pose_gt)
        # MSSD recall calculation using mssd_thresholds
        mssd_recalls.append([1 if mssd_error < theta else 0 for theta in mssd_thresholds])

        # Compute MSPD
        mspd_error = compute_mspd(object_pts, pose_pr, pose_gt, K)
        mspd_recalls.append([1 if mspd_error < theta else 0 for theta in mspd_thresholds])

    # Calculate average recalls for AR
    # ar_vsd = np.mean([np.mean(recall) for recall in vsd_recalls])
    ar_mssd = np.mean([np.mean(recall) for recall in mssd_recalls])
    ar_mspd = np.mean([np.mean(recall) for recall in mspd_recalls])

    # Dataset-level AR
    # ar_dataset = (ar_vsd + ar_mssd + ar_mspd) / 3

    # Return the computed AR values and ARMSSD
    results = {
        # 'ar_vsd': ar_vsd,
        'ar_mssd': ar_mssd,
        'ar_mspd': ar_mspd,
        # 'ar_dataset': ar_dataset
    }
    return results



def pose_sim_to_pose_rigid(pose_sim_in_to_que, pose_in, K_que, K_in, center):
    f_que = np.mean(np.diag(K_que)[:2])
    f_in = np.mean(np.diag(K_in)[:2])
    center_in = pose_apply(pose_in, center)
    depth_in = center_in[2]

    U, S, V = np.linalg.svd(pose_sim_in_to_que[:3,:3])
    R = U @ V
    scale = np.mean(np.abs(S))
    depth_que = depth_in / scale * f_que / f_in

    center_sim = pose_apply(pose_sim_in_to_que, center_in)
    center_que = center_sim / center_sim[2] * depth_que

    rotation = R @ pose_in[:3,:3]

    offset = center_que - rotation @ center
    pose_que = np.concatenate([rotation, offset[:,None]], 1)
    return pose_que

def compose_sim_pose(scale, quat, offset, in_pose, object_center):
    offset = np.concatenate([offset, np.zeros(1)])
    rotation = quat2mat(quat)
    center_in = pose_apply(in_pose, object_center)
    center_que = center_in + offset
    offset = center_que - (scale * rotation @ center_in)
    pose_sim_in_to_que = np.concatenate([scale * rotation, offset[:, None]], 1)
    return pose_sim_in_to_que

def pnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

def ransac_pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE, iter_num=100, rep_error=1.0):
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    state, R_exp, t, inliers = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, dist_coeffs, flags=method,
                                                  iterationsCount=iter_num, reprojectionError=rep_error, confidence=0.999)
    mask = np.zeros([points_3d.shape[0]], np.bool)
    if state:
        R, _ = cv2.Rodrigues(R_exp)
        mask[inliers[:,0]] = True
        return np.concatenate([R, t], axis=-1), mask
    else:
        return np.concatenate([np.eye(3),np.zeros([3,1])],1).astype(np.float32), mask
