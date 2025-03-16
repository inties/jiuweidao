"""
(1) 功能说明：获取模型属性的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import open3d as o3d
import numpy as np


# 获取模型数据
def get_mesh(path):
    # 读取扫描杆STL文件
    mesh = o3d.io.read_triangle_mesh(path)
    # 确保网格已加载并计算法线
    mesh.compute_vertex_normals()
    return mesh


# 获取模型顶点数据
def get_mesh_vertices(mesh):
    # 顶点坐标
    return np.asarray(mesh.vertices)


# 获取模型顶点法向量
def get_mesh_vertices_normals(mesh):
    return np.asarray(mesh.vertex_normals)


# 获取模型三角形面
def get_mesh_triangles(mesh):
    return np.asarray(mesh.triangles)


# 获取模型三角形面法向量
def get_mesh_triangle_normals(mesh):
    return np.asarray(mesh.triangle_normals)


# 获取模型顶点颜色
def get_mesh_vertex_colors(mesh):
    return np.asarray(mesh.vertex_colors)


# 获取模型纹理
def get_mesh_texture(mesh):
    return mesh.textures


# 获取质心
def get_mesh_centroid(mesh):
    vertices = get_mesh_vertices(mesh)
    return np.mean(vertices, axis=0)


# 保存模型
def save_triangle_mesh(mesh, path="./new_mesh.obj"):
    o3d.io.write_triangle_mesh(path, mesh)
