import numpy as np
import argparse
from sklearn.cluster import DBSCAN
import slice_axis as sa  # 导入 slice_Axis.py 中的方法
import open3d as o3d
import plotly.graph_objects as go

# 随机生成与 baseaxis 夹角小于 angle 的方向向量
def generate_axes(baseaxis, angle, axis_num):
    baseaxis = baseaxis / np.linalg.norm(baseaxis)  # 归一化基础轴
    axes = []
    
    while len(axes) < axis_num:
        # 随机生成一个单位向量
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        
        # 计算与 baseaxis 的夹角（弧度）
        cos_theta = np.dot(v, baseaxis)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # 如果夹角小于指定角度，添加到列表
        if theta <= np.radians(angle):
            axes.append(v)
    
    return np.array(axes)

# 多轴切片并计算方向向量
def slice_multiAxis(baseaxis, angle, axis_num, vertices, num_slices=5, min_index=0, max_index=None, 
                    slice_method='cumulative', extension_factor=5):
    # 生成多个轴
    axes = generate_axes(baseaxis, angle, axis_num)
    
    directions = []
    fit_lines = []
    
    # 对每个轴进行切片和拟合
    for axis in axes:
        try:
            if slice_method == 'cumulative':
                centroids = sa.slice_model_cumulative(vertices, axis=axis, num_slices=num_slices, 
                                                     min_index=min_index, max_index=max_index)
            else:  # individual
                centroids = sa.slice_model_individual(vertices, axis=axis, num_slices=num_slices, 
                                                     min_index=min_index, max_index=max_index)
            
            # 拟合直线并获取方向向量
            fit_line_points, direction = sa.fit_line(centroids, vertices, extension_factor=extension_factor)
            directions.append(direction)
            fit_lines.append(fit_line_points)
        except ValueError as e:
            print(f"轴 {axis} 切片或拟合失败: {e}")
            continue
    
    if not directions:
        raise ValueError("所有轴的切片和拟合均失败，无法计算方向。")
    
    # 计算最终方向和拟合直线
    final_direction, final_fit_line = calDir(directions, fit_lines, vertices, extension_factor)
    
    return final_direction, final_fit_line, directions, fit_lines

# 计算最终方向（聚类并取最大簇平均值）
def calDir(directions, fit_lines, vertices, extension_factor):
    directions = np.array(directions)
    
    # 使用 DBSCAN 进行聚类
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(directions)
    labels = clustering.labels_
    
    # 统计每个簇的大小
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        # 如果没有有效簇，取所有方向的平均值
        final_direction = np.mean(directions, axis=0)
    else:
        # 选择最大簇
        max_cluster_label = unique_labels[np.argmax(counts)]
        cluster_directions = directions[labels == max_cluster_label]
        final_direction = np.mean(cluster_directions, axis=0)
    
    # 归一化并确保 Z 分量大于 0
    final_direction = final_direction / np.linalg.norm(final_direction)
    if final_direction[2] < 0:
        final_direction = -final_direction
    
    # 根据最终方向生成拟合直线
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    extend_length = (max_z - min_z) * extension_factor
    extended_min_z = min_z - extend_length
    extended_max_z = max_z + extend_length
    z_vals = np.linspace(extended_min_z, extended_max_z, 100)
    
    # 使用方向向量生成直线（假设起点为模型中心）
    center = np.mean(vertices, axis=0)
    final_fit_line = center + np.outer(z_vals - center[2], final_direction)
    
    return final_direction, final_fit_line

# Open3D 可视化
def visualize_open3d(mesh, final_fit_line, directions, fit_lines):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    
    # 最终拟合直线
    final_line_pcd = o3d.geometry.PointCloud()
    final_line_pcd.points = o3d.utility.Vector3dVector(final_fit_line)
    final_line_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    
    geometries = [mesh, final_line_pcd, axis_pcd]
    
    # 每个轴的方向向量和拟合直线
    for i, (direction, fit_line) in enumerate(zip(directions, fit_lines)):
        # 拟合直线
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(fit_line)
        line_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
        
        # 方向向量（显示为箭头）
        origin = np.mean(fit_line, axis=0)
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.3, cone_radius=0.6, 
                                                       cylinder_height=5.0, cone_height=2.0)
        arrow.paint_uniform_color([1, 0, 0])  # 红色
        arrow.translate(origin)
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        if np.linalg.norm(rotation_axis) > 0:
            arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis / np.linalg.norm(rotation_axis) * angle))
        
        geometries.extend([line_pcd, arrow])
    
    o3d.visualization.draw_geometries(geometries)

# Plotly 可视化
def visualize_plotly(vertices, final_fit_line, directions, fit_lines):
    # 模型点
    model_points = go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                                mode='markers', marker=dict(size=1, color='gray', opacity=0.1), name='Model Points')
    
    # 最终拟合直线
    final_line = go.Scatter3d(x=final_fit_line[:, 0], y=final_fit_line[:, 1], z=final_fit_line[:, 2], 
                              mode='lines', line=dict(color='blue', width=4), name='Final Fitted Line')
    
    data = [model_points, final_line]
    
    # 每个轴的拟合直线和方向向量
    for i, (direction, fit_line) in enumerate(zip(directions, fit_lines)):
        # 拟合直线
        line = go.Scatter3d(x=fit_line[:, 0], y=fit_line[:, 1], z=fit_line[:, 2], 
                            mode='lines', line=dict(color='gray', width=2, dash='dash'), 
                            name=f'Axis {i} Fit Line')
        
        # 方向向量
        origin = np.mean(fit_line, axis=0)
        arrow_length = 10
        arrow_end = origin + direction * arrow_length
        arrow = go.Scatter3d(x=[origin[0], arrow_end[0]], y=[origin[1], arrow_end[1]], z=[origin[2], arrow_end[2]], 
                             mode='lines+markers', line=dict(color='red', width=3), marker=dict(size=5), 
                             name=f'Axis {i} Direction')
        
        data.extend([line, arrow])
    
    # 设置 Z 轴刻度
    z_min = min(np.min(vertices[:, 2]), np.min(final_fit_line[:, 2]))
    z_max = max(np.max(vertices[:, 2]), np.max(final_fit_line[:, 2]))
    z_ticks = np.arange(np.floor(z_min / 2) * 2, np.ceil(z_max / 2) * 2 + 2, 2)

    layout = go.Layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
                   zaxis=dict(tickvals=z_ticks, ticktext=[f"{z:.0f}" for z in z_ticks]), 
                   aspectmode='data'),
        title='3D Multi-Axis Slicing and Fitting Visualization'
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# 主函数
def main():
    parser = argparse.ArgumentParser(description='多轴切片与方向拟合可视化')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth6.ply", 
                        help='3D 模型文件路径 (.obj 或 .ply)')
    parser.add_argument('--viz', type=str, choices=['open3d', 'plotly'], default='plotly',
                        help='可视化方法: "open3d" 或 "plotly"')
    parser.add_argument('--slice_method', type=str, choices=['individual', 'cumulative'], default='cumulative',
                        help='切片方法: "individual" 或 "cumulative"')
    parser.add_argument('--baseaxis', type=float, nargs=3, default=[ -0.42308215,0.32594583, 0.84543528], 
                        help='基础轴向量，例如: 0 0 1')
    parser.add_argument('--angle', type=float, default=10, 
                        help='与基础轴的最大夹角（度）')
    parser.add_argument('--axis_num', type=int, default=20, 
                        help='随机轴的数量')
    parser.add_argument('--num_slices', type=int, default=40, 
                        help='每个轴的切片数量')
    parser.add_argument('--min_index', type=int, default=20, 
                        help='最小切片索引')
    parser.add_argument('--max_index', type=int, default=30, 
                        help='最大切片索引')
    parser.add_argument('--extension_factor', type=float, default=5, 
                        help='直线延伸因子')

    args = parser.parse_args()

    # 加载模型
    mesh = sa.load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    baseaxis = np.array(args.baseaxis)
    
    # 多轴切片和拟合
    final_direction, final_fit_line, directions, fit_lines = slice_multiAxis(
        baseaxis=baseaxis, angle=args.angle, axis_num=args.axis_num, vertices=vertices,
        num_slices=args.num_slices, min_index=args.min_index, max_index=args.max_index,
        slice_method=args.slice_method, extension_factor=args.extension_factor
    )
    
    print(f"最终拟合方向向量: {final_direction}")
    
    # 可视化
    if args.viz == 'open3d':
        visualize_open3d(mesh, final_fit_line, directions, fit_lines)
    elif args.viz == 'plotly':
        visualize_plotly(vertices, final_fit_line, directions, fit_lines)

if __name__ == "__main__":
    main()