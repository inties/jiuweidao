"""
(1) 功能说明：计算模型朝向的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
from sklearn.decomposition import PCA
from utils import mesh_utils, draw_utils
import numpy as np


# 根据点计算朝向
def compute_long_axis(vertices):
    # 使用PCA计算主轴方向
    pca = PCA(n_components=3)
    pca.fit(vertices)

    # PCA的主成分就是包围盒的长轴、宽轴和高轴
    principal_axes = pca.components_

    # 返回主轴中最大的方向作为长轴
    long_axis = principal_axes[0]  # 最大的主成分方向即为长轴方向

    return long_axis


# 获取模型方向向量
def get_direction_vector_by_pca(path, show=False):
    mesh = mesh_utils.get_mesh(path)
    # 提取 STL 模型的顶点作为点云
    pcd = mesh.sample_points_uniformly(len(mesh.vertices))  # 生成均匀分布的点云
    # 获取点云的坐标数据
    points = np.asarray(pcd.points)

    principal_axis = compute_long_axis(points)
    if show:
        # 画图查看
        draw_utils.draw_geometries(
            mesh,
            draw_utils.draw_line(origin=np.mean(points, axis=0),
                                 direction=principal_axis, length=5))  # 绘制点云和朝向向量
    direction = principal_axis / np.linalg.norm(principal_axis)
    return direction
