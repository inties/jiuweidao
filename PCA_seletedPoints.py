import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import argparse

def remove_outlier_clusters(filtered_vertices, eps=1.0, min_samples=5):
    """
    使用 DBSCAN 聚类剔除离群簇，保留最大的簇
    参数：
        filtered_vertices: 筛选后的顶点
        eps: DBSCAN 的邻域半径
        min_samples: DBSCAN 的最小点数（形成簇的最小点数）
    返回：
        剔除离群簇后的顶点
    """
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
    
    # 使用 DBSCAN 进行聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
    
    # 如果没有簇（全部标记为噪声），返回原始点
    if (labels < 0).all():
        print("Warning: DBSCAN found no clusters. Returning original points.")
        return filtered_vertices
    
    # 统计每个簇的点数
    cluster_counts = np.bincount(labels[labels >= 0])
    if len(cluster_counts) == 0:
        print("Warning: No valid clusters found. Returning original points.")
        return filtered_vertices
    
    # 找到最大的簇
    largest_cluster = np.argmax(cluster_counts)
    
    # 保留最大簇的点
    mask = (labels == largest_cluster)
    return filtered_vertices[mask]

def find_tooth_orientation_filtered_pca(mesh, dot_threshold=0.5, eps=1.0, min_samples=5, show=True):
    # 获取顶点和法向量
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    
    # 确保法向量已归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    # 定义参考方向（Z 轴正方向）
    dir = np.array([0, 0, 1])
    
    # 计算法向量与 Z 轴的点积
    dot_products = np.dot(normals, dir)
    
    # 筛选条件：0 < dot(normal, dir) < dot_threshold
    mask = (dot_products > 0) & (dot_products < dot_threshold)
    filtered_vertices = vertices[mask]
    
    if len(filtered_vertices) < 3:
        print("Warning: Too few vertices after filtering. Using all vertices instead.")
        filtered_vertices = vertices
    
    # 剔除离群簇（基于 DBSCAN）
    filtered_vertices = remove_outlier_clusters(filtered_vertices, eps=eps, min_samples=min_samples)
    
    if len(filtered_vertices) < 3:
        print("Warning: Too few vertices after outlier removal. Using all vertices instead.")
        filtered_vertices = vertices
    
    # 计算质心
    centroid = np.mean(filtered_vertices, axis=0)
    
    # 中心化顶点
    centered_vertices = filtered_vertices - centroid
    
    # 应用 PCA
    pca = PCA(n_components=3)
    pca.fit(centered_vertices)
    
   
    # 提取第一主成分作为朝向向量
    components = pca.components_
    dot_products_with_z = np.abs(np.dot(components, dir))  # 计算每个主成分与 Z 轴的绝对点积
    best_component_idx = np.argmax(dot_products_with_z)  # 找到点积绝对值最大的索引
    orientation_vector = components[best_component_idx]  # 选择最接近的成分
    
    # 调整方向：确保与 Z 轴点积大于 0
    if np.dot(orientation_vector, dir) < 0:
        orientation_vector = -orientation_vector
    
    # 计算朝向与 Z 轴的夹角
    angle_deg = np.degrees(np.arccos(np.dot(orientation_vector, dir)))
    print(f"Orientation vector: {orientation_vector}")
    print(f"Centroid: {centroid}")
    print(f"Angle with Z-axis: {angle_deg:.2f}°")
    
    # 可视化
    if show:
        # 创建只包含筛选顶点的点云
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
        filtered_pcd.paint_uniform_color([0.1, 0.7, 0.3])  # 绿色表示筛选出的顶点
        
        # 创建箭头表示朝向
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02, cone_radius=0.04,
            cylinder_height=8.0, cone_height=2.0
        )
        arrow.translate(centroid)
        rotation_axis = np.cross([0, 0, 1], orientation_vector)
        rotation_angle = np.arccos(np.clip(np.dot([0, 0, 1], orientation_vector), -1.0, 1.0))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        arrow.rotate(rotation_matrix, center=centroid)
        arrow.paint_uniform_color([1, 0, 0])  # 红色箭头
        
        # 创建坐标轴
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
        
        # 显示筛选出的顶点、朝向箭头和坐标轴
        o3d.visualization.draw_geometries([filtered_pcd, arrow, axis_pcd])
    
    return orientation_vector, centroid

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Calculate tooth orientation using filtered PCA with DBSCAN-based outlier cluster removal.")
    parser.add_argument('--file_path', type=str, 
                        default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth3.ply",
                        help='Path to the input mesh file')
    parser.add_argument('--dot_threshold', type=float, default=0.5,
                        help='Dot product threshold for filtering vertices (0 to 1, default: 0.5)')
    parser.add_argument('--eps', type=float, default=0.8,
                        help='Epsilon (neighborhood radius) for DBSCAN clustering (default: 1.0)')
    parser.add_argument('--min_samples', type=int, default=30,
                        help='Minimum number of points to form a cluster in DBSCAN (default: 5)')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Show the visualization (default: True)')
    
    args = parser.parse_args()
    
    # 验证参数范围
    if not 0 <= args.dot_threshold <= 1:
        print("Error: dot_threshold must be between 0 and 1.")
        return
    if args.eps <= 0:
        print("Error: eps must be positive.")
        return
    if args.min_samples < 1:
        print("Error: min_samples must be at least 1.")
        return
    
    # 读取网格
    mesh = o3d.io.read_triangle_mesh(args.file_path)
    if not mesh.has_vertices():
        print("Error: Mesh has no vertices.")
        return
    mesh.compute_vertex_normals()
    
    # 计算朝向
    orientation_vector, centroid = find_tooth_orientation_filtered_pca(
        mesh, dot_threshold=args.dot_threshold, eps=args.eps, 
        min_samples=args.min_samples, show=args.show
    )

if __name__ == "__main__":
    main()