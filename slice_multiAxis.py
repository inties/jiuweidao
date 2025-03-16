import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import argparse

####实现沿着多个轴进行切片，计算加权平均方向向量，并可视化结果####

def generate_random_axes(base_axis=np.array([0, 0, 1]), max_angle_deg=10, num_axes=20):
    """生成与基轴偏差在指定角度内的随机方向向量"""
    axes = []
    max_angle_rad = np.deg2rad(max_angle_deg)
    
    for _ in range(num_axes):
        # 在球坐标系中生成随机方向
        theta = np.random.uniform(0, max_angle_rad)  # 角度偏差
        phi = np.random.uniform(0, 2 * np.pi)       # 方位角
        
        # 转换为笛卡尔坐标
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        random_axis = np.array([x, y, z])
        
        # 确保方向向量与基轴方向一致（z > 0）
        if np.dot(random_axis, base_axis) < 0:
            random_axis = -random_axis
            
        axes.append(random_axis / np.linalg.norm(random_axis))
    return np.array(axes)

def load_obj_model(file_path):
    # 同原函数
    if file_path.lower().endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            raise ValueError("Failed to load .obj file or it contains no vertices.")
    elif file_path.lower().endswith('.ply'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices():
            point_cloud = o3d.io.read_point_cloud(file_path)
            if not point_cloud.has_points():
                raise ValueError("Failed to load .ply file or it contains no points.")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = point_cloud.points
        else:
            mesh.compute_vertex_normals()
    else:
        raise ValueError("Unsupported file format. Only .obj and .ply are supported.")
    
    mesh.paint_uniform_color([0.1, 0.7, 0.3])
    return mesh

def compute_centroid(points):
    if points.size == 0:
        return None
    return np.mean(points, axis=0)

def slice_model(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None):
    max_index = num_slices if max_index is None else max_index
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Axis cannot be a zero vector.")
    axis = axis / axis_norm
    
    projections = np.dot(vertices, axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    if min_proj == max_proj:
        raise ValueError("All points project to the same value along the axis.")
    
    slice_width = (max_proj - min_proj) / num_slices
    slice_edges = min_proj + np.arange(num_slices + 1) * slice_width
    slice_indices = np.digitize(projections, slice_edges) - 1
    
    slice_centroids = []
    slice_point_counts = []
    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        slice_points = vertices[slice_indices == i]
        if slice_points.size > 0:
            centroid = np.mean(slice_points, axis=0)
            slice_centroids.append(centroid)
            slice_point_counts.append(len(slice_points))
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice centroids found within the specified range.")
    
    # 检查切片点数，丢弃明显偏少的切片
    if len(slice_point_counts) > 1:
        mean_count = np.mean(slice_point_counts)
        std_count = np.std(slice_point_counts)
        threshold = mean_count - 2 * std_count  # 假设低于均值-2倍标准差为异常
        valid_indices = [i for i, count in enumerate(slice_point_counts) if count >= max(threshold, 1)]
        slice_centroids = [slice_centroids[i] for i in valid_indices]
    
    return np.array(slice_centroids)

def fit_line(points, vertices, extension_factor=5, inlier_threshold=0.7):
    """拟合直线并返回方向向量和内点占比"""
    if points.size == 0:
        raise ValueError("No valid points provided for line fitting.")
    
    model = RANSACRegressor(
        min_samples=int(len(points) * 0.5),
        residual_threshold=0.5,
        max_trials=1000
    )
    X = points[:, 2].reshape(-1, 1)
    y = points[:, [0, 1]]
    model.fit(X, y)
    
    inlier_mask = model.inlier_mask_
    inlier_ratio = np.sum(inlier_mask) / len(points)
    
    if inlier_ratio < inlier_threshold:
        return None, inlier_ratio  # 内点占比不足，返回 None
    
    inlier_points = points[inlier_mask]
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    extend_length = (max_z - min_z) * extension_factor
    extended_min_z = min_z - extend_length
    extended_max_z = max_z + extend_length
    
    z_vals = np.linspace(extended_min_z, extended_max_z, 100)
    xy_vals = model.predict(z_vals.reshape(-1, 1))
    fit_points = np.column_stack((xy_vals[:, 0], xy_vals[:, 1], z_vals))
    direction = fit_points[-1] - fit_points[0]
    direction = direction / np.linalg.norm(direction)
    
    return direction, inlier_ratio

def create_arrow_from_direction(origin, direction, length=10.0):
    # 同原函数
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, 
                                                   cylinder_height=length * 0.8, cone_height=length * 0.2)
    arrow.translate(origin)
    default_dir = np.array([0, 0, 1])
    direction = direction / np.linalg.norm(direction)
    axis = np.cross(default_dir, direction)
    if np.linalg.norm(axis) > 0:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(default_dir, direction))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(rotation_matrix, center=origin)
    arrow.paint_uniform_color([1, 0, 0])
    return arrow

def compute_final_direction(vertices, num_slices, max_angle_deg=10, num_axes=20, inlier_threshold=0.7):
    """计算加权平均方向向量"""
    axes = generate_random_axes(max_angle_deg=max_angle_deg, num_axes=num_axes)
    directions = []
    weights = []
    
    for axis in axes:
        try:
            slice_centroids = slice_model(vertices, axis=axis, num_slices=num_slices)
            direction, inlier_ratio = fit_line(slice_centroids, vertices, inlier_threshold=inlier_threshold)
            if direction is not None:
                directions.append(direction)
                weights.append(inlier_ratio)  # 使用内点占比作为权重
        except ValueError:
            continue
    
    if not directions:
        raise ValueError("No valid directions computed from any axis.")
    
    directions = np.array(directions)
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # 归一化权重
    final_direction = np.average(directions, weights=weights, axis=0)
    final_direction = final_direction / np.linalg.norm(final_direction)
    return final_direction

def visualize_open3d(mesh, final_direction):
    centroid = compute_centroid(np.asarray(mesh.vertices))
    arrow = create_arrow_from_direction(centroid, final_direction, length=10.0)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, arrow, axis_pcd])

def main():
    parser = argparse.ArgumentParser(description='3D Model Visualization with multi-axis slicing')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth1.ply", help='Path to the 3D model file (.obj or .ply)')
    parser.add_argument('--num_slices', type=int, default=4, help='Number of slices (default: 4)')
    parser.add_argument('--max_angle_deg', type=float, default=10, help='Max angle deviation from Z-axis in degrees (default: 10)')
    parser.add_argument('--num_axes', type=int, default=20, help='Number of random axes (default: 20)')
    parser.add_argument('--inlier_threshold', type=float, default=0.7, help='Inlier ratio threshold for RANSAC (default: 0.7)')

    args = parser.parse_args()

    # 加载模型并计算最终方向
    mesh = load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    final_direction = compute_final_direction(
        vertices, 
        num_slices=args.num_slices, 
        max_angle_deg=args.max_angle_deg, 
        num_axes=args.num_axes, 
        inlier_threshold=args.inlier_threshold
    )
    
    # 可视化
    visualize_open3d(mesh, final_direction)

if __name__ == "__main__":
    main()