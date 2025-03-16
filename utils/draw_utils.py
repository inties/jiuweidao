"""
(1) 功能说明：模型可视化的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import open3d as o3d
import numpy as np


# 画线
def draw_line(origin, direction, length=1, color=[1, 0, 0], show=False):
    line_set = o3d.geometry.LineSet()  # 绘制朝向向量
    line_set.points = o3d.utility.Vector3dVector([origin, origin + direction * length])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])  # 设置朝向向量的颜色
    if show:
        draw_geometries([line_set])
    return line_set


# 画点云
def draw_points(points, show=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 设置点云颜色
    pcd.paint_uniform_color([1, 0, 0])  # 设置为红色
    if show:
        # 可视化
        o3d.visualization.draw_geometries([pcd])
    return pcd


# 画图
def draw_geometries(*args):
    o3d.visualization.draw_geometries(np.asarray(args))

