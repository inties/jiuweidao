import open3d as o3d
import numpy as np
import argparse
from sklearn.decomposition import PCA  # 导入 PCA 依赖

####通过PCA方法计算牙齿朝向####

# 假设 utils 模块已正确导入，模拟其基本功能
# 如果 utils 模块未提供，以下为占位实现
# class UtilsPlaceholder:
#     @staticmethod
#     def get_mesh_vertices(mesh):
#         return np.asarray(mesh.vertices)

#     @staticmethod
#     def get_mesh_vertices_normals(mesh):
#         return np.asarray(mesh.vertex_normals)

#     @staticmethod
#     def draw_line(origin, direction):
#         # 占位实现，返回一个简单的箭头几何体
#         arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04,
#                                                        cylinder_height=1.0, cone_height=0.3)
#         arrow.translate(origin)
#         return arrow

#     @staticmethod
#     def draw_geometries(*args):
#         # 占位实现，调用 open3d 的可视化
#         o3d.visualization.draw_geometries(list(args))

# 假设 tooth_normal_utils 包含 find_normal_by_vertices_divide_bayes_iteration
# 这里仅占位，实际需用户提供实现
class ToothNormalUtilsPlaceholder:
    @staticmethod
    def find_normal_by_vertices_divide_bayes_iteration(vertices, normals, plane_normal, centroid, iteration_num,
                                                       top_weight=1, low_weight=0.1):
        # 占位实现，直接返回初始 plane_normal
        return plane_normal

# 绑定 utils 模块
try:
    from utils import tooth_normal_utils, draw_utils, mesh_utils
except ImportError:
    print("Warning: utils module not found. Using placeholder implementations.")
    tooth_normal_utils = ToothNormalUtilsPlaceholder()
    draw_utils = UtilsPlaceholder()
    mesh_utils = UtilsPlaceholder()

# 命令行参数解析
parser = argparse.ArgumentParser(description="Calculate tooth orientation using PCA or vertex normal method.")
parser.add_argument('--method', type=str, default='pca', choices=['pca', 'normal'],
                    help='Method to calculate tooth orientation: "pca" or "normal" (default: pca)')
parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth3.ply",
                    help='Path to the input mesh file (default: specified path)')
parser.add_argument('--iteration_num', type=int, default=2,
                    help='Number of iterations for the normal method (default: 2)')
parser.add_argument('--top_weight', type=float, default=1.0,
                    help='Weight for top plane normals in normal method (default: 1.0)')
parser.add_argument('--low_weight', type=float, default=0.1,
                    help='Weight for bottom plane normals in normal method (default: 0.1)')
parser.add_argument('--show', action='store_true',default=True,
                    help='Show the visualization (default: True)')

args = parser.parse_args()

def main():
    # 读取并预处理网格
    try:
        mesh = o3d.io.read_triangle_mesh(args.file_path)
        if not mesh.has_vertices():
            raise ValueError("Mesh has no vertices.")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.1, 0.7, 0.3])
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # 方法 1: 基于顶点法向量的贝叶斯迭代法
    def find_tooth_normal_by_vertices_divide(mesh, iteration_num, top_weight=1, low_weight=0.1, show=False):
        vertices = mesh_utils.get_mesh_vertices(mesh)
        normals = mesh_utils.get_mesh_vertices_normals(mesh)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms

        centroid = vertices.mean(axis=0)
        average_normal = normals.mean(axis=0)
        average_normal = average_normal / np.linalg.norm(average_normal)

        normal = tooth_normal_utils.find_normal_by_vertices_divide_bayes_iteration(vertices, normals, average_normal, centroid, iteration_num,
                                                                                  top_weight, low_weight)
        if show:
            arrow = draw_utils.draw_line(origin=centroid, direction=normal)
            draw_utils.draw_geometries(arrow, mesh)
        return mesh, normal

    # 方法 2: 基于 PCA 的朝向计算
    def find_tooth_orientation_pca(vertices):
        # 计算质心
        centroid = np.mean(vertices, axis=0)
        
        # 中心化顶点
        centered_vertices = vertices - centroid
        
        # 应用 PCA
        pca = PCA(n_components=3)
        pca.fit(centered_vertices)
        
        # 提取第一主成分作为朝向向量
        components = pca.components_
        dot_products_with_z = np.abs(np.dot(components, [0,0,1]))  # 计算每个主成分与 Z 轴的绝对点积
        best_component_idx = np.argmax(dot_products_with_z)  # 找到点积绝对值最大的索引
        orientation_vector = components[best_component_idx]  # 选择最接近的成分
        
        # 调整方向：确保与 Z 轴点积大于 0
        z_axis = np.array([0, 0, 1])
        if np.dot(orientation_vector, z_axis) < 0:
            orientation_vector = -orientation_vector
        
        return orientation_vector, centroid

    # 根据命令行选择方法
    if args.method.lower() == 'normal':
        # 使用原有方法
        _, normal = find_tooth_normal_by_vertices_divide(mesh, iteration_num=args.iteration_num,
                                                        top_weight=args.top_weight, low_weight=args.low_weight,
                                                        show=args.show)
        centroid = np.mean(np.asarray(mesh.vertices), axis=0)
    else:
        # 使用 PCA 方法
        vertices = np.asarray(mesh.vertices)
        normal, centroid = find_tooth_orientation_pca(vertices)

    # 创建箭头
    try:
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04,
                                                       cylinder_height=8.0, cone_height=2.0)
        arrow.translate(centroid)
        # 使用旋转矩阵调整箭头方向
        rotation_axis = np.cross([0, 0, 1], normal)
        rotation_angle = np.arccos(np.clip(np.dot([0, 0, 1], normal) / (np.linalg.norm(normal) * 1.0), -1.0, 1.0))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        arrow.rotate(rotation_matrix, center=centroid)
        arrow.paint_uniform_color([1, 0, 0])
    except Exception as e:
        print(f"Error creating arrow: {e}")
        return

    # 显示结果
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    geometries = [mesh, arrow, axis_pcd]
    if args.show:
        try:
            o3d.visualization.draw_geometries(geometries)
        except Exception as e:
            print(f"Error in visualization: {e}")

if __name__ == "__main__":
    main()