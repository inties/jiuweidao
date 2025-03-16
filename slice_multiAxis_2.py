import numpy as np
import argparse
from sklearn.cluster import DBSCAN
import slice_axis as sa
import open3d as o3d
import plotly.graph_objects as go
####沿着baseAxis周围随机生成多个轴进行切片，并可视化结果####
# generate_axes 和 slice_multiAxis 函数保持不变
def generate_axes(baseaxis, angle, axis_num):
    baseaxis = baseaxis / np.linalg.norm(baseaxis)
    axes = []
    while len(axes) < axis_num:
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        cos_theta = np.dot(v, baseaxis)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        if theta <= np.radians(angle):
            axes.append(v)
    return np.array(axes)

def slice_multiAxis(baseaxis, angle, axis_num, vertices, num_slices=5, min_index=0, max_index=None, 
                    slice_method='cumulative', extension_factor=5):
    axes = generate_axes(baseaxis, angle, axis_num)
    directions = []
    fit_lines = []
    
    for axis in axes:
        try:
            if slice_method == 'cumulative':
                centroids = sa.slice_model_cumulative(vertices, axis=axis, num_slices=num_slices, 
                                                     min_index=min_index, max_index=max_index)
            else:
                centroids = sa.slice_model_individual(vertices, axis=axis, num_slices=num_slices, 
                                                     min_index=min_index, max_index=max_index)
            fit_line_points, direction = sa.fit_line(centroids, vertices, extension_factor=extension_factor)
            directions.append(direction)
            fit_lines.append(fit_line_points)
        except ValueError as e:
            print(f"轴 {axis} 切片或拟合失败: {e}")
            continue
    
    if not directions:
        raise ValueError("所有轴的切片和拟合均失败，无法计算方向。")
    
    final_direction, final_fit_line = calDir(directions, fit_lines, vertices, extension_factor)
    return final_direction, final_fit_line, fit_lines

def calDir(directions, fit_lines, vertices, extension_factor):
    directions = np.array(directions)
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(directions)
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        final_direction = np.mean(directions, axis=0)
    else:
        max_cluster_label = unique_labels[np.argmax(counts)]
        cluster_directions = directions[labels == max_cluster_label]
        final_direction = np.mean(cluster_directions, axis=0)
    
    final_direction = final_direction / np.linalg.norm(final_direction)
    if final_direction[2] < 0:
        final_direction = -final_direction
    
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    extend_length = (max_z - min_z) * extension_factor
    extended_min_z = min_z - extend_length
    extended_max_z = max_z + extend_length
    z_vals = np.linspace(extended_min_z, extended_max_z, 100)
    center = np.mean(vertices, axis=0)
    final_fit_line = center + np.outer(z_vals - center[2], final_direction)
    
    return final_direction, final_fit_line

# Open3D 可视化（添加控制台控制）
def visualize_open3d(mesh, final_fit_line, fit_lines, show_intermediate=True):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
    final_line_pcd = o3d.geometry.PointCloud()
    final_line_pcd.points = o3d.utility.Vector3dVector(final_fit_line)
    final_line_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
    
    geometries = [mesh, final_line_pcd, axis_pcd]
    
    if show_intermediate:
        for i, fit_line in enumerate(fit_lines):
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(fit_line)
            line_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
            geometries.append(line_pcd)
    
    o3d.visualization.draw_geometries(geometries)

# Plotly 可视化（保持带按钮）
def visualize_plotly(vertices, final_fit_line, fit_lines):
    model_points = go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                                mode='markers', marker=dict(size=1, color='gray', opacity=0.1), name='Model Points')
    final_line = go.Scatter3d(x=final_fit_line[:, 0], y=final_fit_line[:, 1], z=final_fit_line[:, 2], 
                              mode='lines', line=dict(color='blue', width=4), name='Final Fitted Line')
    
    intermediate_lines = []
    for i, fit_line in enumerate(fit_lines):
        line = go.Scatter3d(x=fit_line[:, 0], y=fit_line[:, 1], z=fit_line[:, 2], 
                            mode='lines', line=dict(color='gray', width=2, dash='dash'), 
                            name=f'Axis {i} Fit Line')
        intermediate_lines.append(line)
    
    z_min = min(np.min(vertices[:, 2]), np.min(final_fit_line[:, 2]))
    z_max = max(np.max(vertices[:, 2]), np.max(final_fit_line[:, 2]))
    z_ticks = np.arange(np.floor(z_min / 2) * 2, np.ceil(z_max / 2) * 2 + 2, 2)

    data = [model_points, final_line] + intermediate_lines
    
    updatemenus = [
        dict(
            buttons=[
                dict(label="显示中间直线",
                     method="update",
                     args=[{"visible": [True] * len(data)},
                           {"title": "3D Multi-Axis Slicing (中间直线可见)"}]),
                dict(label="隐藏中间直线",
                     method="update",
                     args=[{"visible": [True, True] + [False] * len(intermediate_lines)},
                           {"title": "3D Multi-Axis Slicing (仅最终直线)"}])
            ],
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ]
    
    layout = go.Layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
                   zaxis=dict(tickvals=z_ticks, ticktext=[f"{z:.0f}" for z in z_ticks]), 
                   aspectmode='data'),
        title='3D Multi-Axis Slicing and Fitting Visualization',
        updatemenus=updatemenus
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# 主函数（添加控制台参数）
def main():
    parser = argparse.ArgumentParser(description='多轴切片与方向拟合可视化')
    parser.add_argument('--file_path', type=str, default=r"E:\MylabProjects\jiuweidao_deteting\wzpdata\teeth2\repaired\repaired_teeth8.ply", 
                        help='3D 模型文件路径 (.obj 或 .ply)')
    parser.add_argument('--viz_method', type=str, choices=['open3d', 'plotly'], default='plotly',
                        help='可视化方法: "open3d" 或 "plotly"')
    parser.add_argument('--slice_method', type=str, choices=['individual', 'cumulative'], default='cumulative',
                        help='切片方法: "individual" 或 "cumulative"')
    parser.add_argument('--baseaxis', type=float, nargs=3, default= [ 0.07289371,-0.3424829 ,0.93669204], 
                        help='基础轴向量，例如: 0 0 1')
    parser.add_argument('--angle', type=float, default=3, 
                        help='与基础轴的最大夹角（度）')
    parser.add_argument('--axis_num', type=int, default=40, 
                        help='随机轴的数量')
    parser.add_argument('--num_slices', type=int, default=40, 
                        help='每个轴的切片数量')
    parser.add_argument('--min_index', type=int, default=21, 
                        help='最小切片索引')
    parser.add_argument('--max_index', type=int, default=29, 
                        help='最大切片索引')
    parser.add_argument('--extension_factor', type=float, default=5, 
                        help='直线延伸因子')
    parser.add_argument('--show_intermediate', type=bool, default=False, 
                        help='是否显示中间过程的拟合直线 (True/False，仅对 Open3D 有效)')

    args = parser.parse_args()
    mesh = sa.load_obj_model(args.file_path)
    vertices = np.asarray(mesh.vertices)
    baseaxis = np.array(args.baseaxis)
    
    final_direction, final_fit_line, fit_lines = slice_multiAxis(
        baseaxis=baseaxis, angle=args.angle, axis_num=args.axis_num, vertices=vertices,
        num_slices=args.num_slices, min_index=args.min_index, max_index=args.max_index,
        slice_method=args.slice_method, extension_factor=args.extension_factor
    )
    
    print(f"最终拟合方向向量: {final_direction}")
    
    if args.viz_method == 'open3d':
        visualize_open3d(mesh, final_fit_line, fit_lines, show_intermediate=args.show_intermediate)
    elif args.viz_method == 'plotly':
        visualize_plotly(vertices, final_fit_line, fit_lines)

if __name__ == "__main__":
    main()