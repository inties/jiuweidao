import open3d as o3d
import numpy as np

# 假设 utils 模块已正确导入
from utils import tooth_normal_utils, draw_utils, mesh_utils

####通过计算顶点法向量均值方法，计算牙齿朝向####


file_path = "E:\\MylabProjects\\jiuweidao_deteting\\wzpdata\\teeth2\\repaired\\repaired_teeth8.ply"
mesh = o3d.io.read_triangle_mesh(file_path)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.1, 0.7, 0.3])

def find_tooth_normal_by_vertices_divide(mesh, iteration_num, top_weight=1, low_weight=0.1, show=False):
    vertices = mesh_utils.get_mesh_vertices(mesh)  # 获取顶点坐标
    normals = mesh_utils.get_mesh_vertices_normals(mesh)  # 获取顶点法向量
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms

    # 2. 计算质心和平均法向量
    centroid = vertices.mean(axis=0)  # 计算质心
    average_normal = normals.mean(axis=0)  # 计算平均法向量
    average_normal = average_normal / np.linalg.norm(average_normal)  # 归一化法向量

    normal = tooth_normal_utils.find_normal_by_vertices_divide_bayes_iteration(vertices, normals, average_normal, centroid, iteration_num,
                                                    top_weight, low_weight)
    if show:
        arrow = draw_utils.draw_line(origin=centroid, direction=normal)
        draw_utils.draw_geometries(arrow, mesh)
    return mesh, normal
# 计算朝向
_, normal = find_tooth_normal_by_vertices_divide(mesh, iteration_num=2, top_weight=1, low_weight=0.1, show=False)

# 创建箭头
centroid = np.mean(np.asarray(mesh.vertices), axis=0)
arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, 
                                               cylinder_height=8.0, cone_height=2.0)
arrow.translate(centroid)
arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross([0, 0, 1], normal)), center=centroid)
arrow.paint_uniform_color([1, 0, 0])

# 显示结果
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([mesh, arrow, axis_pcd])