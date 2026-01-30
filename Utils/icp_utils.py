# encoding: utf-8
# @File: icp_utils.py
# @Author: Spade-Atek
# @Date  : 2025/10/05/17:32
# ------------------------------------------------------------

import open3d as o3d
import numpy as np
import copy
import time

def load_and_preprocess_pcd(pcd, voxel_size=0.05):
    """
    读取点云并下采样 + 法线估计
    """
    print(f"[INFO] Loaded {pcd}, points={np.asarray(pcd.points).shape[0]}")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd, pcd_down

def draw_registration(source, target, transformation):
    """
    可视化对齐结果
    """
    src_temp = copy.deepcopy(source)
    tgt_temp = copy.deepcopy(target)
    src_temp.paint_uniform_color([1, 0, 0])
    tgt_temp.paint_uniform_color([0, 0, 1])
    src_temp.transform(transformation)

    o3d.visualization.draw_geometries(
        [src_temp, tgt_temp],window_name="Local region after ICP"
    )

def icp_multiscale(source_down, target_down, voxel_size, init_trans=np.eye(4)):
    """
    多尺度 ICP（模仿 CloudCompare 迭代收紧策略）
    """
    # 逐级阈值和迭代次数（可调）
    # thresholds = [voxel_size * 3.0, voxel_size * 1.5, voxel_size * 0.75]
    # # iterations = [40, 25, 15]
    # iterations = [60, 35, 20]

    thresholds = [voxel_size * 8, voxel_size * 2.0, voxel_size * 0.3]
    #thresholds = [voxel_size * 2, voxel_size * 2.0, voxel_size * 2]
    iterations = [80, 50, 30]
    current_trans = init_trans

    s = time.time()
    for i, (th, it) in enumerate(zip(thresholds, iterations)):
        print(f"[Level {i+1}] threshold={th:.3f}, iter={it}")

        reg_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, th, current_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
        )
        current_trans = reg_icp.transformation
        print("fitness",reg_icp.fitness)
        print("inlier_rmse",reg_icp.inlier_rmse)

    icp_time = time.time() - s
    return current_trans, reg_icp.fitness, reg_icp.inlier_rmse, icp_time

def icp_registration_pcd(source_pcd, target_pcd,
                              voxel_size=0.05,
                              init_trans=np.eye(4),
                              visualize=False,
                              save_aligned=False):
    """
    主流程：加载点云 -> 预处理 -> 多尺度ICP -> 可视化 & 保存 （此函数直接传入已读出的点云）
    """
    print("[INFO] Starting multi-scale ICP registration ...")
    src_raw, src_down = load_and_preprocess_pcd(source_pcd, voxel_size)
    tgt_raw, tgt_down = load_and_preprocess_pcd(target_pcd, voxel_size)

    final_trans, fitness, inlier_rmse, icp_time= icp_multiscale(src_down, tgt_down, voxel_size, init_trans)

    if visualize:
        draw_registration(src_raw, tgt_raw, final_trans)

    if save_aligned:
        aligned = copy.deepcopy(src_raw)
        aligned.transform(final_trans)
        o3d.io.write_point_cloud("aligned_source.pcd", aligned)
        print("[INFO] Saved aligned point cloud -> aligned_source.pcd")

    # return final_trans, fitness, inlier_rmse, icp_time
    return final_trans


# if __name__ == "__main__":
#     src_path = r"C:\Users\XIE Yutai\Desktop\11a.pcd"  # 改成你的源点云路径
#     tgt_path = r"C:\Users\XIE Yutai\Desktop\22b.pcd"  # 改成你的目标点云路径w
#
#     # 粗配准矩阵
#     init = np.eye(4)
#
#     # 执行ICP
#     transform = icp_registration(src_path, tgt_path,
#                                           voxel_size=0.05,
#                                           init_trans=init,
#                                           visualize=True)


# 不用版本：这个是直接读取路径版本
def load_and_preprocess(pcd_path, voxel_size=0.05):
    """
    读取点云并下采样 + 法线估计
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[INFO] Loaded {pcd_path}, points={np.asarray(pcd.points).shape[0]}")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd, pcd_down

# 不用
def icp_registration(source_path, target_path,
                              voxel_size=0.05,
                              init_trans=np.eye(4),
                              visualize=True,
                              save_aligned=False):
    """
    主流程：加载点云 -> 预处理 -> 多尺度ICP -> 可视化 & 保存
    """
    src_raw, src_down = load_and_preprocess(source_path, voxel_size)
    tgt_raw, tgt_down = load_and_preprocess(target_path, voxel_size)

    print("\n[INFO] Starting multi-scale ICP registration ...")
    final_trans = icp_multiscale(src_down, tgt_down, voxel_size, init_trans)

    print("\n========== ICP Registration Done ==========")
    print("T_icp = \n", final_trans)

    if visualize:
        draw_registration(src_raw, tgt_raw, final_trans)

    if save_aligned:
        aligned = copy.deepcopy(src_raw)
        aligned.transform(final_trans)
        o3d.io.write_point_cloud("aligned_source.ply", aligned)
        print("[INFO] Saved aligned point cloud -> aligned_source.ply")

    return final_trans