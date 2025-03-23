##可视化每个切片，以检查中间是否存在问题
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
import argparse
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

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

# 计算加权质心的辅助函数
def fit_ellipse_spline(points, num_points=100):
    """使用B样条曲线拟合椭圆"""
    if len(points) < 4:
        return None, None
    
    # 计算点云的质心
    centroid = np.mean(points, axis=0)
    
    # 使用最近邻算法重新排序点
    ordered_points = order_points_by_nearest_neighbor(points)
    
    # B样条拟合
    try:
        # 确保点是闭合的（首尾相连）
        closed_points = np.vstack([ordered_points, ordered_points[0]])
        
        # 使用参数化样条拟合
        tck, u = splprep([closed_points[:, 0], closed_points[:, 1], closed_points[:, 2]], s=0, k=3, per=1)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new, z_new = splev(u_new, tck)
        
        spline_points = np.column_stack((x_new, y_new, z_new))
        return spline_points, centroid
    except Exception as e:
        print(f"样条拟合失败: {e}")
        return None, None

# 添加新函数：基于最近邻的点排序
def order_points_by_nearest_neighbor(points):
    """使用最近邻算法对点进行排序，确保相邻点在空间上也相邻"""
    if len(points) <= 1:
        return points
    
    # 创建点的副本，避免修改原始数据
    remaining_points = points.copy()
    
    # 从第一个点开始（可以选择任意点作为起点）
    ordered_indices = [0]
    current_point_idx = 0
    
    # 构建KD树用于快速查找最近邻
    tree = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(remaining_points)
    
    # 循环直到所有点都被处理
    while len(ordered_indices) < len(points):
        # 查找当前点的最近邻
        distances, indices = tree.kneighbors([remaining_points[current_point_idx]])
        
        # 获取最近邻的索引（排除自身）
        nearest_idx = indices[0][1] if indices[0][0] == current_point_idx else indices[0][0]
        
        # 将最近邻添加到有序列表
        ordered_indices.append(nearest_idx)
        
        # 更新当前点为最近邻
        current_point_idx = nearest_idx
        
        # 如果所有点都已处理，跳出循环
        if len(ordered_indices) == len(points):
            break
        
        # 重新构建KD树，排除已处理的点
        mask = np.ones(len(remaining_points), dtype=bool)
        for idx in ordered_indices:
            if idx < len(mask):
                mask[idx] = False
        
        if not np.any(mask):
            break
            
        remaining_subset = remaining_points[mask]
        if len(remaining_subset) == 0:
            break
            
        tree = NearestNeighbors(n_neighbors=min(2, len(remaining_subset)), algorithm='ball_tree').fit(remaining_subset)
        
        # 映射当前点索引到新的子集索引
        current_point_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(mask)[0])}
        if current_point_idx in current_point_map:
            current_point_idx = current_point_map[current_point_idx]
        else:
            # 如果当前点不在新子集中，选择第一个未处理的点
            current_point_idx = 0
    
    # 返回排序后的点
    return points[ordered_indices]

def compute_weighted_centroid(points, k=10, use_spline=False):
    if points.size == 0:
        return None
    
    if use_spline:
        # 使用样条曲线拟合椭圆并获取椭圆上的点
        spline_points, centroid = fit_ellipse_spline(points)
        if spline_points is not None:
            points = spline_points
    
    # 使用 k 近邻计算每个点的局部密度
    k = min(k, len(points) - 1)  # 确保 k 不超过点数
    if k < 1:
        return np.mean(points, axis=0)  # 如果点数太少，直接返回平均值
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    # 取平均距离作为密度指标（距离越大，密度越小）
    densities = np.mean(distances, axis=1)
    # 权重与密度倒数成正比
    weights = 1 / (densities + 1e-6)  # 避免除以 0
    weights /= np.sum(weights)  # 归一化权重
    # 加权平均
    weighted_centroid = np.average(points, axis=0, weights=weights)
    return weighted_centroid

# 修改后的切片函数（独立切片）
def slice_model_individual(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None, voxel_size=0.5, k_neighbors=10, use_spline=False):
    """原始方法：每个切片独立计算质心，应用降采样和加权质心计算"""
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
    
    slice_points_list = []
    slice_centroids = []
    slice_splines = []
    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        slice_points = vertices[slice_indices == i]
        if slice_points.size > 0:
            # 步骤 1：降采样
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(slice_points)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            downsampled_points = np.asarray(pcd_down.points)
            
            if downsampled_points.size > 0:
                # 保存降采样后的点用于可视化
                slice_points_list.append(downsampled_points)
                
                # 步骤 2：如果启用样条拟合，拟合椭圆
                if use_spline:
                    spline_points, _ = fit_ellipse_spline(downsampled_points)
                    if spline_points is not None:
                        slice_splines.append(spline_points)
                        downsampled_points = spline_points
                
                # 步骤 3：基于点计算加权质心
                centroid = compute_weighted_centroid(downsampled_points, k=k_neighbors)
                if centroid is not None:
                    slice_centroids.append(centroid)
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice points or centroids found within the specified range.")
    
    return slice_points_list, np.array(slice_centroids), slice_splines if use_spline else None

# 修改后的切片函数（累积切片）
def slice_model_cumulative(vertices, axis=np.array([0, 0, 1]), num_slices=5, min_index=0, max_index=None, voxel_size=0.5, k_neighbors=10, use_spline=False):
    """新方法：累积合并切片计算质心，应用降采样和加权质心计算"""
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
    
    slice_points_list = []
    slice_centroids = []
    slice_splines = []
    accumulated_points = np.array([]).reshape(0, 3)
    
    for i in range(min_index, max_index + 1):
        if i < 0 or i >= num_slices:
            continue
        current_slice_points = vertices[slice_indices == i]
        if current_slice_points.size > 0:
            # 步骤 1：降采样当前切片点
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(current_slice_points)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            downsampled_points = np.asarray(pcd_down.points)
            
            if downsampled_points.size > 0:
                # 保存降采样后的点用于可视化
                slice_points_list.append(downsampled_points)
                
                # 步骤 2：如果启用样条拟合，拟合椭圆
                if use_spline:
                    spline_points, _ = fit_ellipse_spline(downsampled_points)
                    if spline_points is not None:
                        slice_splines.append(spline_points)
                        downsampled_points = spline_points
                
                # 累积点
                accumulated_points = np.vstack([accumulated_points, downsampled_points])
                # 步骤 3：基于累积点计算加权质心
                centroid = compute_weighted_centroid(accumulated_points, k=k_neighbors)
                if centroid is not None:
                    slice_centroids.append(centroid)
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice points or centroids found within the specified range.")
    
    return slice_points_list, np.array(slice_centroids), slice_splines if use_spline else None
# 修改 open3d 可视化函数以显示所有切片点
def visualize_open3d_with_fixed_z_ticks(mesh, slice_points_list, slice_centroids, fit_line_points):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    
    # 创建颜色列表，确保相邻切片颜色不同
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
    
    # 可视化质心（可选）
    slice_centroids_pcd = o3d.geometry.PointCloud()
    slice_centroids_pcd.points = o3d.utility.Vector3dVector(slice_centroids)
    slice_centroids_pcd.paint_uniform_color([0, 0, 0])  # 质心用黑色表示
    
    fit_line_pcd = o3d.geometry.PointCloud()
    fit_line_pcd.points = o3d.utility.Vector3dVector(fit_line_points)
    fit_line_pcd.paint_uniform_color([0, 0, 1])

    geometries = [mesh] + slice_clouds + [slice_centroids_pcd, fit_line_pcd, axis_pcd]
    o3d.visualization.draw_geometries(geometries)

# 修改 plotly 可视化函数以显示所有切片点
def visualize_plotly(vertices, slice_points_list, slice_centroids, fit_line_points, slice_splines, args):
    model_points = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=1, color='gray', opacity=0.1),
        name='Model Points',
        visible=True
    )
    
    # 为每个切片的点创建散点图和椭圆拟合曲线
    slice_traces = []
    spline_traces = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'darkred', 'darkgreen']
    
    for i, points in enumerate(slice_points_list):
        # 原始点云
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[i % len(colors)], opacity=0.8),
            name=f'Slice {i + args.min_index}',
            visible=True
        )
        slice_traces.append(trace)
        
        # 拟合椭圆（如果存在）
        if slice_splines is not None and i < len(slice_splines):
            spline_points = slice_splines[i]
            spline_trace = go.Scatter3d(
                x=spline_points[:, 0],
                y=spline_points[:, 1],
                z=spline_points[:, 2],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=4),
                name=f'Fitted Ellipse {i + args.min_index}',
                visible=False
            )
            spline_traces.append(spline_trace)
    
    # 可视化质心
    centroids = go.Scatter3d(
        x=slice_centroids[:, 0],
        y=slice_centroids[:, 1],
        z=slice_centroids[:, 2],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Slice Centroids',
        visible=True
    )
    
    fitted_line = go.Scatter3d(
        x=fit_line_points[:, 0],
        y=fit_line_points[:, 1],
        z=fit_line_points[:, 2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Fitted Line',
        visible=True
    )
    
    z_min = min(np.min(vertices[:, 2]), np.min(slice_centroids[:, 2]), np.min(fit_line_points[:, 2]))
    z_max = max(np.max(vertices[:, 2]), np.max(slice_centroids[:, 2]), np.max(fit_line_points[:, 2]))
    z_ticks = np.arange(np.floor(z_min / 2) * 2, np.ceil(z_max / 2) * 2 + 2, 2)

    # 创建下拉菜单选项
    updatemenus = list([
        dict(
            buttons=list([
                dict(
                    args=[{'visible': [True] * len(slice_traces) + [False] * len(spline_traces) + [True] * 3}],
                    label='Show Original Points',
                    method='restyle'
                ),
                dict(
                    args=[{'visible': [False] * len(slice_traces) + [True] * len(spline_traces) + [True] * 3}],
                    label='Show Fitted Ellipses',
                    method='restyle'
                )
            ]),
            direction='down',
            showactive=True,
            x=0.1,
            y=1.1,
            xanchor='left',
            yanchor='top'
        ),
    ])

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            zaxis=dict(tickvals=z_ticks, ticktext=[f"{z:.0f}" for z in z_ticks]),
            aspectmode='data'
        ),
        updatemenus=updatemenus,
        title='3D Point Cloud Visualization',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    all_traces = [model_points] + slice_traces + spline_traces + [centroids, fitted_line]
    fig = go.Figure(data=all_traces, layout=layout)
    fig.show()

# 其余函数保持不变，但需要更新 main 函数以传递新参数
def main():
    parser = argparse.ArgumentParser(description='3D Model Visualization with slicing and line fitting')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth6.ply", 
                        help='Path to the 3D model file (.obj or .ply)')
    parser.add_argument('--viz_method', type=str, choices=['open3d', 'plotly'], default='plotly',
                        help='Visualization method: "open3d" or "plotly"')
    parser.add_argument('--slice_method', type=str, choices=['individual', 'cumulative'], default='individual',
                        help='Slice method: "individual" (original) or "cumulative" (new)')
    parser.add_argument('--extension_factor', type=float, default=2, help='Line extension factor')
    parser.add_argument('--min_index', type=int, default=10, help='Minimum slice index to retain')
    parser.add_argument('--max_index', type=int, default=16, help='Maximum slice index to retain')
    parser.add_argument('--num_slices', type=int, default=20, help='Number of slices')
    parser.add_argument('--voxel_size', type=float, default=1, help='Voxel size for downsampling')
    parser.add_argument('--k_neighbors', type=int, default=3, help='Number of neighbors for density weighting')
    parser.add_argument('--use_spline', action='store_true', help='Use spline fitting for ellipse')

    args = parser.parse_args()

    mesh = load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    axis = np.array([0, 0, 1])
    
    max_index = args.max_index if args.max_index is not None else args.num_slices - 1
    
    # 根据选择的切片方法计算质心和切片点
    if args.slice_method == 'individual':
        slice_points_list, slice_centroids, slice_splines = slice_model_individual(
            vertices, 
            axis=axis, 
            num_slices=args.num_slices, 
            min_index=args.min_index, 
            max_index=max_index,
            voxel_size=args.voxel_size,
            k_neighbors=args.k_neighbors,
            use_spline=args.use_spline
        )
    else:  # cumulative
        slice_points_list, slice_centroids, slice_splines = slice_model_cumulative(
            vertices, 
            axis=axis, 
            num_slices=args.num_slices, 
            min_index=args.min_index, 
            max_index=max_index,
            voxel_size=args.voxel_size,
            k_neighbors=args.k_neighbors,
            use_spline=args.use_spline
        )
    
    fit_line_points, direction = fit_line(slice_centroids, vertices, extension_factor=args.extension_factor)
    
    print(f"Fitted line direction vector (Z > 0): {direction}")

    if args.viz_method == 'open3d':
        visualize_open3d_with_fixed_z_ticks(mesh, slice_points_list, slice_centroids, fit_line_points)
    elif args.viz_method == 'plotly':
        visualize_plotly(vertices, slice_points_list, slice_centroids, fit_line_points, slice_splines, args)

if __name__ == "__main__":
    main()