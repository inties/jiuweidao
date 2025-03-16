import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import argparse
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # 设置为你的 CPU 核心数
# 用户提供的加载模型函数
def load_obj_model(file_path):
    if file_path.lower().endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError("无法加载.obj文件或文件中没有顶点。")
    elif file_path.lower().endswith('.ply'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            point_cloud = o3d.io.read_point_cloud(file_path)
            if not point_cloud.has_points():
                raise ValueError("无法加载.ply文件或文件中没有点。")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = point_cloud.points
        else:
            mesh.compute_vertex_normals()
    else:
        raise ValueError("不支持的文件格式，仅支持.obj和.ply。")
    
    mesh.paint_uniform_color([0.1, 0.7, 0.3])  # 设置模型颜色
    return mesh

# 计算牙齿朝向的主函数
def compute_tooth_orientation(mesh, delta=0.01, eps=0.3, min_samples=5):
    # 提取顶点并确保顶点法线可用
    vertices = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()  # 如果没有法线，计算法线
    normals = np.asarray(mesh.vertex_normals)

    # 找到咬合面顶点
    z_values = vertices[:, 2]  # 提取z坐标
    max_z = np.max(z_values)   # 找到z坐标最大值
    occlusal_mask = (z_values >= max_z - delta) & (z_values <= max_z)  # 咬合面顶点掩码
    occlusal_indices = np.where(occlusal_mask)[0]
    occlusal_normals = normals[occlusal_indices]  # 获取咬合面顶点的法线

    # 使用DBSCAN对法线进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(occlusal_normals)
    labels = db.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 排除噪声点（label = -1）
    if -1 in unique_labels:
        counts = counts[unique_labels != -1]
        unique_labels = unique_labels[unique_labels != -1]

    # 找到最大的簇
    max_cluster_label = unique_labels[np.argmax(counts)]
    cluster_normals = occlusal_normals[labels == max_cluster_label]

    # 计算牙齿朝向（最大簇法线的均值）
    mean_normal = np.mean(cluster_normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)  # 归一化为单位向量

    # 计算牙齿质心
    centroid = np.mean(vertices, axis=0)

    return centroid, mean_normal, vertices, occlusal_mask

# 可视化函数（修改以不同颜色标记咬合面顶点）
def visualize_tooth(vertices, occlusal_mask, centroid, orientation_vector, line_length=10):
    # 分离咬合面顶点和非咬合面顶点
    occlusal_vertices = vertices[occlusal_mask]  # 咬合面顶点
    non_occlusal_vertices = vertices[~occlusal_mask]  # 非咬合面顶点

    # 创建Plotly图形
    fig = go.Figure()

    # 绘制非咬合面顶点（绿色）
    if len(non_occlusal_vertices) > 0:
        fig.add_trace(go.Scatter3d(
            x=non_occlusal_vertices[:, 0], y=non_occlusal_vertices[:, 1], z=non_occlusal_vertices[:, 2],
            mode='markers', marker=dict(size=1, color='green'), name='非咬合面顶点'
        ))

    # 绘制咬合面顶点（蓝色）
    if len(occlusal_vertices) > 0:
        fig.add_trace(go.Scatter3d(
            x=occlusal_vertices[:, 0], y=occlusal_vertices[:, 1], z=occlusal_vertices[:, 2],
            mode='markers', marker=dict(size=1, color='blue'), name='咬合面顶点'
        ))

    # 绘制朝向直线（红色）
    line_points = np.array([centroid, centroid + line_length * orientation_vector])
    fig.add_trace(go.Scatter3d(
        x=line_points[:, 0], y=line_points[:, 1], z=line_points[:, 2],
        mode='lines', line=dict(color='red', width=5), name='牙齿朝向'
    ))

    # 设置图形布局
    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ))
    fig.show()

# 主函数
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="计算牙齿朝向并可视化")
    parser.add_argument('--file', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth6.ply", help='模型文件路径（.obj或.ply）')
    parser.add_argument('--delta', type=float, default=1, help='咬合面z值范围（默认0.01）')
    parser.add_argument('--eps', type=float, default=0.3, help='DBSCAN聚类eps参数（默认0.3）')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN聚类最小样本数（默认5）')
    parser.add_argument('--line_length', type=float, default=10, help='朝向直线长度（默认10）')

    args = parser.parse_args()

    # 加载模型
    mesh = load_obj_model(args.file)

    # 计算牙齿朝向
    centroid, mean_normal, vertices, occlusal_mask = compute_tooth_orientation(mesh, args.delta, args.eps, args.min_samples)

    # 打印结果
    print("牙齿质心:", centroid)
    print("牙齿朝向（单位向量）:", mean_normal)

    # 可视化
    visualize_tooth(vertices, occlusal_mask, centroid, mean_normal, args.line_length)

if __name__ == "__main__":
    main()