import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
import argparse
import plotly.graph_objects as go

####实现累计切片求质心和拟合直线，并可视化结果####

def load_obj_model(file_path):
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

def slice_model_individual(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None):
    """原始方法：每个切片独立计算质心"""
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
    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        slice_points = vertices[slice_indices == i]
        if slice_points.size > 0:
            centroid = np.mean(slice_points, axis=0)
            slice_centroids.append(centroid)
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice centroids found within the specified range.")
    
    return np.array(slice_centroids)

def slice_model_cumulative(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None):
    """新方法：累积合并切片计算质心"""
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
    accumulated_points = np.array([]).reshape(0, 3)
    
    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        current_slice_points = vertices[slice_indices == i]
        if current_slice_points.size > 0:
            accumulated_points = np.vstack([accumulated_points, current_slice_points])
            centroid = np.mean(accumulated_points, axis=0)
            slice_centroids.append(centroid)
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice centroids found within the specified range.")
    
    return np.array(slice_centroids)
#修改后的拟合直线函数，返回拟合直线点和方向向量
def fit_line(points, vertices, extension_factor=5):
    if points.size == 0 or len(points) < 2:
        raise ValueError("提供的点不足以拟合直线（至少需要 2 个点）。")
    
    model = RANSACRegressor(
        min_samples=int(len(points) * 0.5),
        residual_threshold=0.5,
        max_trials=1000
    )
    X = points[:, 2].reshape(-1, 1)  # Z 坐标
    y = points[:, [0, 1]]            # X, Y 坐标
    model.fit(X, y)
    
    inlier_mask = model.inlier_mask_
    inlier_points = points[inlier_mask]
    
    if len(inlier_points) < 2:
        raise ValueError("RANSAC 未能找到足够的内点来拟合直线（至少需要 2 个内点）。")
    
    # 提取斜率 (dx/dz, dy/dz)，coef_ 应为 (2, 1) 形状
    slope = model.estimator_.coef_
    if slope.shape != (2, 1):
        raise ValueError(f"斜率形状异常：{slope.shape}，预期为 (2, 1)。")
    
    # 创建方向向量 [dx/dz, dy/dz, 1]
    direction = np.array([slope[0, 0], slope[1, 0], 1.0])
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("计算出的方向向量长度为零。")
    direction = direction / direction_norm  # 归一化
    
    # 确保 Z 分量大于 0
    if direction[2] < 0:
        direction = -direction
    
    # 计算延伸的直线点
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    extend_length = (max_z - min_z) * extension_factor
    extended_min_z = min_z - extend_length
    extended_max_z = max_z + extend_length
    
    z_vals = np.linspace(extended_min_z, extended_max_z, 100)
    xy_vals = model.predict(z_vals.reshape(-1, 1))
    fit_line_points = np.column_stack((xy_vals[:, 0], xy_vals[:, 1], z_vals))
    
    return fit_line_points, direction

def visualize_open3d_with_fixed_z_ticks(mesh, slice_centroids, fit_line_points):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    
    slice_centroids_pcd = o3d.geometry.PointCloud()
    slice_centroids_pcd.points = o3d.utility.Vector3dVector(slice_centroids)
    slice_centroids_pcd.paint_uniform_color([1, 0, 0])

    fit_line_pcd = o3d.geometry.PointCloud()
    fit_line_pcd.points = o3d.utility.Vector3dVector(fit_line_points)
    fit_line_pcd.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([mesh, slice_centroids_pcd, fit_line_pcd, axis_pcd])

def visualize_plotly(vertices, slice_centroids, fit_line_points):
    model_points = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=1, color='gray', opacity=0.1),
        name='Model Points'
    )
    
    centroids = go.Scatter3d(
        x=slice_centroids[:, 0],
        y=slice_centroids[:, 1],
        z=slice_centroids[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Slice Centroids'
    )
    
    fitted_line = go.Scatter3d(
        x=fit_line_points[:, 0],
        y=fit_line_points[:, 1],
        z=fit_line_points[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Fitted Line'
    )
    
    z_min = min(np.min(vertices[:, 2]), np.min(slice_centroids[:, 2]), np.min(fit_line_points[:, 2]))
    z_max = max(np.max(vertices[:, 2]), np.max(slice_centroids[:, 2]), np.max(fit_line_points[:, 2]))
    z_ticks = np.arange(np.floor(z_min / 2) * 2, np.ceil(z_max / 2) * 2 + 2, 2)

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            zaxis=dict(tickvals=z_ticks, ticktext=[f"{z:.0f}" for z in z_ticks]),
            aspectmode='data'
        ),
        title='3D Point Cloud Visualization'
    )
    
    fig = go.Figure(data=[model_points, centroids, fitted_line], layout=layout)
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='3D Model Visualization with slicing and line fitting')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth7.ply", 
                        help='Path to the 3D model file (.obj or .ply)')
    parser.add_argument('--viz_method', type=str, choices=['open3d', 'plotly'], default='open3d',
                        help='Visualization method: "open3d" or "plotly"')
    parser.add_argument('--slice_method', type=str, choices=['individual', 'cumulative'], default='individual',
                        help='Slice method: "individual" (original) or "cumulative" (new)')
    parser.add_argument('--extension_factor', type=float, default=2, help='Line extension factor')
    parser.add_argument('--min_index', type=int, default=22, help='Minimum slice index to retain')
    parser.add_argument('--max_index', type=int, default=30, help='Maximum slice index to retain')
    parser.add_argument('--num_slices', type=int, default=40, help='Number of slices')  # 添加默认值

    args = parser.parse_args()

    mesh = load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    axis = np.array(  [ 0.45146372,0.1956906 ,0.89207486])
    
    max_index = args.max_index if args.max_index is not None else args.num_slices - 1
    
    # 根据选择的切片方法计算质心
    if args.slice_method == 'individual':
        slice_centroids = slice_model_individual(
            vertices, 
            axis=axis, 
            num_slices=args.num_slices, 
            min_index=args.min_index, 
            max_index=max_index
        )
    else:  # cumulative
        slice_centroids = slice_model_cumulative(
            vertices, 
            axis=axis, 
            num_slices=args.num_slices, 
            min_index=args.min_index, 
            max_index=max_index
        )
    
    # 调用修改后的 fit_line 函数
    fit_line_points, direction = fit_line(slice_centroids, vertices, extension_factor=args.extension_factor)
    
    # 打印方向向量（可选）
    print(f"Fitted line direction vector (Z > 0): {direction}")

    # 可视化
    if args.viz_method == 'open3d':
        visualize_open3d_with_fixed_z_ticks(mesh, slice_centroids, fit_line_points)
    elif args.viz_method == 'plotly':
        visualize_plotly(vertices, slice_centroids, fit_line_points)

if __name__ == "__main__":
    main()

