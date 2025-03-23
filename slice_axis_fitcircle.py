import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
import argparse
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
###点云降采样、使用样条曲线、拟合切片质心。
#### 加载模型函数保持不变 ####
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

#### 计算加权质心（不变） ####
def compute_weighted_centroid(points, k=10):
    if points.size == 0:
        return None
    k = min(k, len(points) - 1)
    if k < 1:
        return np.mean(points, axis=0)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    densities = np.mean(distances, axis=1)
    weights = 1 / (densities + 1e-6)
    weights /= np.sum(weights)
    return np.average(points, axis=0, weights=weights)

#### 新增：基于最近邻的顺序连接并拟合样条曲线 ####
def fit_spline_curve_nearest_neighbor(points, n_points=100, smooth_factor=0):
    """
    使用最近邻算法确定点连接顺序，然后拟合样条曲线。
    返回拟合后的闭合曲线点。
    """
    if len(points) < 3:
        raise ValueError("Point cloud has too few points to form a closed curve.")

    # 初始化
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    order = []
    distances = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)

    # 从随机点开始
    current_idx = np.random.randint(n)
    order.append(current_idx)
    visited[current_idx] = True

    # 最近邻连接
    while len(order) < n:
        current_distances = distances[current_idx]
        current_distances[visited] = np.inf  # 排除已访问的点
        next_idx = np.argmin(current_distances)
        order.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    # 转换为有序点云
    ordered_points = points[order]
    # 闭合曲线：添加首点
    ordered_points_closed = np.vstack([ordered_points, ordered_points[0]])

    # 样条拟合（直接在三维空间中进行）
    tck, u = splprep(ordered_points_closed.T, s=smooth_factor, per=1)  # per=1 表示周期性
    u_fine = np.linspace(0, 1, n_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    fit_points = np.column_stack([x_fine, y_fine, z_fine])
    return fit_points

#### 修改后的切片函数 ####
def slice_model_individual(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None, 
                          voxel_size=0.5, k_neighbors=10, use_spline_fit=False):
    max_index = num_slices if max_index is None else max_index
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Axis cannot be a zero vector.")
    axis = axis / axis_norm
    
    projections = np.dot(vertices, axis)
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    slice_width = (max_proj - min_proj) / num_slices
    slice_edges = min_proj + np.arange(num_slices + 1) * slice_width
    slice_indices = np.digitize(projections, slice_edges) - 1
    
    slice_points_list = []
    slice_centroids = []
    spline_fit_points_list = []

    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        slice_points = vertices[slice_indices == i]
        if slice_points.size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(slice_points)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            downsampled_points = np.asarray(pcd_down.points)
            
            if downsampled_points.size > 0:
                slice_points_list.append(downsampled_points)
                if use_spline_fit:
                    fit_points = fit_spline_curve_nearest_neighbor(downsampled_points)
                    spline_fit_points_list.append(fit_points)
                    centroid = compute_weighted_centroid(fit_points, k=k_neighbors)
                else:
                    spline_fit_points_list.append(None)
                    centroid = compute_weighted_centroid(downsampled_points, k=k_neighbors)
                if centroid is not None:
                    slice_centroids.append(centroid)
    
    return slice_points_list, np.array(slice_centroids), spline_fit_points_list

#### 直线拟合函数（不变） ####
def fit_line(points, vertices, extension_factor=5):
    model = RANSACRegressor(min_samples=int(len(points) * 0.5), residual_threshold=0.5, max_trials=1000)
    X = points[:, 2].reshape(-1, 1)
    y = points[:, [0, 1]]
    model.fit(X, y)
    slope = model.estimator_.coef_.flatten()
    direction = np.array([slope[0], slope[1], 1.0])
    direction = direction / np.linalg.norm(direction)
    if direction[2] < 0:
        direction = -direction
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    extend_length = (max_z - min_z) * extension_factor
    z_vals = np.linspace(min_z - extend_length, max_z + extend_length, 100)
    xy_vals = model.predict(z_vals.reshape(-1, 1))
    fit_line_points = np.column_stack((xy_vals[:, 0], xy_vals[:, 1], z_vals))
    return fit_line_points, direction

#### Plotly 可视化（不变，略） ####
def visualize_plotly(vertices, slice_points_list, slice_centroids, fit_line_points, spline_fit_points_list, args):
    model_points = go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                mode='markers', marker=dict(size=1, color='gray', opacity=0.1), name='Model Points')
    
    slice_traces = []
    spline_traces = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'darkred', 'darkgreen']
    for i, (points, spline_points) in enumerate(zip(slice_points_list, spline_fit_points_list)):
        trace = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                             mode='markers', marker=dict(size=2, color=colors[i % len(colors)], opacity=0.8),
                             name=f'Slice {i + args.min_index} Points', visible=True)
        slice_traces.append(trace)
        if spline_points is not None:
            spline_trace = go.Scatter3d(x=spline_points[:, 0], y=spline_points[:, 1], z=spline_points[:, 2],
                                        mode='lines', line=dict(color=colors[i % len(colors)], width=4),
                                        name=f'Slice {i + args.min_index} Spline', visible=False)
            spline_traces.append(spline_trace)
    
    centroids = go.Scatter3d(x=slice_centroids[:, 0], y=slice_centroids[:, 1], z=slice_centroids[:, 2],
                             mode='markers', marker=dict(size=5, color='black'), name='Slice Centroids')
    fitted_line = go.Scatter3d(x=fit_line_points[:, 0], y=fit_line_points[:, 1], z=fit_line_points[:, 2],
                               mode='lines', line=dict(color='blue', width=4), name='Fitted Line')
    
    z_min = min(np.min(vertices[:, 2]), np.min(slice_centroids[:, 2]), np.min(fit_line_points[:, 2]))
    z_max = max(np.max(vertices[:, 2]), np.max(slice_centroids[:, 2]), np.max(fit_line_points[:, 2]))
    z_ticks = np.arange(np.floor(z_min / 2) * 2, np.ceil(z_max / 2) * 2 + 2, 2)

    # 确保长度正确
    visibility_points = [True] * len(slice_traces) + [False] * len(spline_traces) + [True, True, True]  # model, centroids, line
    visibility_spline = [False] * len(slice_traces) + [True] * len(spline_traces) + [True, True, True]
    
    updatemenus = [dict(
        buttons=[
            dict(label="Show Points", method="update", args=[{"visible": visibility_points}]),
            dict(label="Show Spline", method="update", args=[{"visible": visibility_spline}])
        ],
        direction="down", showactive=True, x=0.1, xanchor="left", y=1.1, yanchor="top"
    )]

    layout = go.Layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                   zaxis=dict(tickvals=z_ticks, ticktext=[f"{z:.0f}" for z in z_ticks]), aspectmode='data'),
        title='3D Point Cloud Visualization',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        updatemenus=updatemenus
    )
    
    fig = go.Figure(data=[model_points] + slice_traces + spline_traces + [centroids, fitted_line], layout=layout)
    fig.show()
#### 添加 Open3D 可视化函数 ####
def visualize_open3d(mesh, slice_points_list, slice_centroids, fit_line_points, spline_fit_points_list):
    """使用 Open3D 可视化模型、切片点、质心和拟合直线"""
    # 创建坐标系
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    
    # 创建颜色列表
    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
        [1, 0, 1], [0, 1, 1], [0.5, 0, 0], [0, 0.5, 0]
    ]
    
    # 为每个切片的点创建点云
    slice_clouds = []
    for i, points in enumerate(slice_points_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color_idx = i % len(colors)
        pcd.paint_uniform_color(colors[color_idx])
        slice_clouds.append(pcd)
    
    # 为样条拟合点创建点云（如果存在）
    spline_clouds = []
    if spline_fit_points_list and any(p is not None for p in spline_fit_points_list):
        for i, spline_points in enumerate(spline_fit_points_list):
            if spline_points is not None:
                spline_pcd = o3d.geometry.PointCloud()
                spline_pcd.points = o3d.utility.Vector3dVector(spline_points)
                color_idx = i % len(colors)
                # 使用稍微不同的颜色
                spline_color = [c * 0.7 for c in colors[color_idx]]
                spline_pcd.paint_uniform_color(spline_color)
                spline_clouds.append(spline_pcd)
    
    # 可视化质心
    slice_centroids_pcd = o3d.geometry.PointCloud()
    slice_centroids_pcd.points = o3d.utility.Vector3dVector(slice_centroids)
    slice_centroids_pcd.paint_uniform_color([0, 0, 0])  # 质心用黑色表示
    
    # 可视化拟合直线
    fit_line_pcd = o3d.geometry.PointCloud()
    fit_line_pcd.points = o3d.utility.Vector3dVector(fit_line_points)
    fit_line_pcd.paint_uniform_color([0, 0, 1])  # 蓝色直线

    # 组合所有几何体
    geometries = [mesh] + slice_clouds + spline_clouds + [slice_centroids_pcd, fit_line_pcd, axis_pcd]
    o3d.visualization.draw_geometries(geometries)

#### 修改主函数以支持 Open3D 可视化 ####
def main():
    parser = argparse.ArgumentParser(description='3D Model Visualization with Spline Fitting')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth4.ply")
    parser.add_argument('--viz_method', type=str, choices=['open3d', 'plotly'], default='open3d', help='Visualization method: open3d or plotly')
    parser.add_argument('--slice_method', type=str, choices=['individual', 'cumulative'], default='individual')
    parser.add_argument('--use_spline_fit', action='store_true', help='Use spline fitting for centroid calculation')
    parser.add_argument('--extension_factor', type=float, default=2)
    parser.add_argument('--min_index', type=int, default=10)
    parser.add_argument('--max_index', type=int, default=15)
    parser.add_argument('--num_slices', type=int, default=20)
    parser.add_argument('--voxel_size', type=float, default=1)
    parser.add_argument('--k_neighbors', type=int, default=3)

    args = parser.parse_args()
    mesh = load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    axis = np.array([0, 0, 1])
    max_index = args.max_index if args.max_index is not None else args.num_slices - 1
    
    slice_points_list, slice_centroids, spline_fit_points_list = slice_model_individual(
        vertices, axis=axis, num_slices=args.num_slices, min_index=args.min_index, max_index=max_index,
        voxel_size=args.voxel_size, k_neighbors=args.k_neighbors, use_spline_fit=args.use_spline_fit
    )
    
    fit_line_points, direction = fit_line(slice_centroids, vertices, extension_factor=args.extension_factor)
    
    print(f"Fitted line direction vector (Z > 0): {direction}")
    if args.use_spline_fit:
        print("Centroids calculated using spline-fitted points.")
    else:
        print("Centroids calculated using original downsampled points.")

    if args.viz_method == 'plotly':
        visualize_plotly(vertices, slice_points_list, slice_centroids, fit_line_points, spline_fit_points_list, args)
    elif args.viz_method == 'open3d':
        visualize_open3d(mesh, slice_points_list, slice_centroids, fit_line_points, spline_fit_points_list)

if __name__ == "__main__":
    main()