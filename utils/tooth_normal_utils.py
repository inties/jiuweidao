"""
(1) 功能说明：牙齿朝向的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import open3d as o3d
import numpy as np
from utils import geometry_utils


# 通过一个点和法向量组成的平面，把顶点分为上平面点和下平面点，再根据不同的权值进行组合形成新的法向量，可以迭代
def find_normal_by_vertices_divide_bayes_iteration(vertices, normals, plane_normal, centroid, iteration_num, top_weight=1,
                        low_weight=0.1, start_num=1):
    if start_num <= iteration_num:
        plane_point = centroid  # 平面上的点
        plane_offset = -np.dot(plane_normal, plane_point)  # 偏移量 d
        distances = np.dot(vertices, plane_normal) + plane_offset
        points_normal_above_plane = normals[distances > 0]
        points_normal_below_plane = normals[distances < 0]
        average_normal = (top_weight * points_normal_above_plane.mean(axis=0) +
                          low_weight * points_normal_below_plane.mean(axis=0)) / (top_weight+low_weight)  # 计算平均法向量
        average_normal = average_normal / np.linalg.norm(average_normal)  # 归一化法向量
        return find_normal_by_vertices_divide_bayes_iteration(vertices, normals, average_normal, centroid, iteration_num, top_weight, low_weight,
                                   start_num + 1)
    else:
        return plane_normal


# 使用三角形面积作为权值，每个三角形的法向量乘以权值求平均得到模型朝向
def find_normal_by_triangles(mesh):
    # 2. 获取三角形法向量和顶点
    triangle_normals = np.asarray(mesh.triangle_normals)  # 获取所有面的法向量
    triangles = np.asarray(mesh.triangles)  # 获取三角形索引
    vertices = np.asarray(mesh.vertices)  # 获取顶点坐标

    areas = np.array([geometry_utils.triangle_area(vertices[t[0]], vertices[t[1]], vertices[t[2]]) for t in triangles])

    # 4. 使用加权平均法向量计算整体朝向
    weighted_normals = triangle_normals * areas[:, np.newaxis]  # 法向量按面积加权
    average_normal = np.mean(weighted_normals, axis=0)  # 计算平均法向量
    average_normal = average_normal / np.linalg.norm(average_normal)  # 单位化
    return average_normal


# 根据某个点和法向量组成的平面，分别找到模型中平面上面和下面的点和三角形面片
def find_above_plane_vertices_and_triangles(mesh, point, point_normal):
    # 获取网格顶点和面
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 计算顶点到平面的符号距离
    distances = np.dot(vertices - point, point_normal)

    # 筛选平面上方的顶点索引
    above_plane_indices = np.where(distances > 0)[0]
    above_plane_mask = distances > 0

    # 筛选平面上方的三角面片
    # 判断一个三角形的三个顶点是否都在平面上方
    above_plane_triangles = [
        tri for tri in triangles if all(above_plane_mask[tri])
    ]

    # 创建新的三角网格
    above_plane_mesh = o3d.geometry.TriangleMesh()
    above_plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    above_plane_mesh.triangles = o3d.utility.Vector3iVector(above_plane_triangles)

    # 筛选平面上方的顶点索引
    below_plane_indices = np.where(distances < 0)[0]
    below_plane_mask = distances < 0

    # 筛选平面上方的三角面片
    # 判断一个三角形的三个顶点是否都在平面上方
    below_plane_triangles = [
        tri for tri in triangles if all(below_plane_mask[tri])
    ]

    # 创建新的三角网格
    below_plane_mesh = o3d.geometry.TriangleMesh()
    below_plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    below_plane_mesh.triangles = o3d.utility.Vector3iVector(below_plane_triangles)

    return above_plane_mesh, below_plane_mesh


