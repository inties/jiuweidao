"""
(1) 功能说明：模型处理的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

from utils import mesh_utils, direction_utils, geometry_utils, tooth_cuff_utils, json_utils
import trimesh


# 获取某点附近圆内所有顶点
def get_near_vertices(mesh, r, center):
    vertices_teeth = mesh_utils.get_mesh_vertices(mesh)

    # 计算每个顶点与中心点的距离
    distances = np.linalg.norm(vertices_teeth - center, axis=1)
    # 找出距离小于半径的顶点索引
    indices_within_radius = np.where(distances < r)[0]

    return mesh, vertices_teeth[indices_within_radius]


# 定义某个点坐标找到这点某个半径圆内所有的面片和顶点等相关mesh信息
def extract_vertices_and_faces_with_shared_vertices(mesh, target_point, radius):
    # 获取顶点和三角形面数据
    vertices = mesh_utils.get_mesh_vertices(mesh)
    triangles = mesh_utils.get_mesh_triangles(mesh)

    # 计算所有顶点到目标点的距离
    distances = np.linalg.norm(vertices - target_point, axis=1)

    # 找出所有在半径范围内的顶点索引
    selected_indices = np.where(distances <= radius)[0]
    selected_set = set(selected_indices)

    # 创建索引映射，用于更新新模型的面索引
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    # 提取在范围内的三角形面
    new_triangles = []
    for tri in triangles:
        if all(idx in selected_set for idx in tri):
            new_triangles.append([index_map[idx] for idx in tri])

    # 创建新的顶点和面数据
    new_vertices = vertices[selected_indices]
    new_triangles = np.array(new_triangles)

    # 去除重复顶点，确保共享顶点
    all_vertices = new_vertices
    unique_vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)

    # 更新面索引
    updated_triangles = inverse_indices[new_triangles]

    # 创建新的网格
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(unique_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(updated_triangles)

    # 保持原始的其他属性
    if mesh.has_vertex_normals():
        vertex_normals = mesh_utils.get_mesh_vertices_normals(mesh)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals[selected_indices])

    if mesh.has_vertex_colors():
        vertex_colors = mesh_utils.get_mesh_vertex_colors(mesh)
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[selected_indices])

    if mesh.has_textures():
        # 保持纹理坐标（如果存在）
        new_mesh.textures = mesh_utils.get_mesh_texture(mesh)

    return new_mesh


# 把两个模型合并成一个模型
def merge_meshes(mesh1, mesh2):
    # 提取顶点和三角形面数据
    vertices1 = np.asarray(mesh1.vertices)
    triangles1 = np.asarray(mesh1.triangles)

    vertices2 = np.asarray(mesh2.vertices)
    triangles2 = np.asarray(mesh2.triangles)

    # 更新第二个模型的三角形索引
    offset = len(vertices1)
    triangles2_updated = triangles2 + offset

    # 合并顶点和三角形面
    merged_vertices = np.vstack((vertices1, vertices2))
    merged_triangles = np.vstack((triangles1, triangles2_updated))

    # 创建新的网格
    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(merged_vertices)
    merged_mesh.triangles = o3d.utility.Vector3iVector(merged_triangles)

    mesh_utils.save_triangle_mesh("merged_mesh.obj", merged_mesh)
    return merged_mesh


# stl格式转为off格式
def stl_to_off(input_path, output_path=""):
    if len(output_path) == 0:
        output_path = input_path[:-4]
    # 1. 读取 STL 文件
    stl_mesh = o3d.io.read_triangle_mesh(input_path)

    # 2. 将网格保存为 OFF 文件
    o3d.io.write_triangle_mesh(output_path + ".off", stl_mesh)

    print("转换成功！")


# 通过trimesh包加载的模型得到uv展开的坐标
def get_mesh_uv(path, show=False):
    mesh = trimesh.load_mesh(path)

    # 执行UV展开
    mesh = mesh.unwrap()

    # 获取展开后的UV信息
    uvs = mesh.visual.uv  # UV坐标

    if show:
        faces = mesh.faces  # 顶点索引

        # 创建一个空白的UV图像
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 绘制每个面对应的UV三角形
        for face in faces:
            uv0, uv1, uv2 = uvs[face]
            triangle = np.array([uv0, uv1, uv2])
            ax.fill(triangle[:, 0], triangle[:, 1], edgecolor='black', fill=False)

        # 可选：添加每个UV点作为小圆点
        for uv in uvs:
            ax.scatter(uv[0], uv[1], color='red', s=10)

        # 显示UV图
        plt.gca().invert_yaxis()  # 反转Y轴（有些UV图坐标系Y轴方向不同）
        plt.title('UV Map')
        plt.show()
    return mesh, uvs


# 添加质心三角形面片
def add_centroid_to_mesh(input_path, output_path, show=False, save_to_json=False):
    # 加载现有的模型
    mesh = mesh_utils.get_mesh(input_path)

    # 获取现有网格的顶点和三角形面
    vertices = mesh_utils.get_mesh_vertices(mesh)
    triangles = mesh_utils.get_mesh_triangles(mesh)
    triangles_normals = mesh_utils.get_mesh_triangle_normals(mesh)

    # 执行计算
    shared_vertices = tooth_cuff_utils.calculate_adjacent_face_angles_and_shared_vertices(triangles, triangles_normals)

    # 记录满足条件的点坐标
    points_coordinate = set()
    for edge in shared_vertices:
        points_coordinate.add(edge[0])
        points_coordinate.add(edge[1])

    points = [vertices[i] for i in points_coordinate]
    center_point = np.mean(points, axis=0)
    # 平面方程参数
    point_on_plane = np.asarray(center_point)  # 平面上的一点
    direction = direction_utils.get_direction_vector_by_pca("../teeth/马燕婷46/Lower AbutmentAlignmentScan.stl")
    normal_vector = np.array(direction)  # 平面的法向量
    d = np.dot(normal_vector, point_on_plane)  # 计算平面方程的常数d

    # 找到所有距离小于0.5的点
    close_points_coordinate = []
    for i in points_coordinate:
        distance = geometry_utils.point_to_plane_distance(vertices[i], normal_vector, d)
        if distance < 0.5:
            close_points_coordinate.append(int(i))

    close_points = [vertices[i] for i in close_points_coordinate]
    close_points_center = np.mean(close_points, axis=0)

    # 定义质心三角形顶点
    new_vertices = np.array([
        [close_points_center[0], close_points_center[1] + 0.001, close_points_center[2]],
        # [close_points_center[0] + 0.001, close_points_center[1], close_points_center[2]],
        [close_points_center[0], close_points_center[1], close_points_center[2]]
    ])

    # 将新顶点添加到现有顶点列表中
    vertices = np.vstack([vertices, new_vertices])

    # 定义新的三角形的面
    new_triangle = np.array([
        [len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]  # 索引为最后三个顶点
    ])

    # 将新的面添加到现有三角形面列表中
    triangles = np.vstack([triangles, new_triangle])

    # 更新网格
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 确保网格有顶点和颜色属性
    # if not mesh.has_vertex_colors():
    #     mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)))  # 默认给顶点上白色

    close_points_color = []
    red = [255, 0, 0]
    for i in close_points_coordinate:
        # mesh.vertex_colors[i] = red
        close_points_color.append(red)
    blue = [0, 0, 255]
    # mesh.vertex_colors[-1] = blue

    # 重新计算法线
    mesh.compute_vertex_normals()

    if save_to_json:
        close_points_coordinate.append(len(vertices) - 1)
        close_points.append(close_points_center)
        close_points_color.append(blue)
        close_points_list = [point.tolist() for point in close_points]
        json_utils.save_cuff_vertices_to_json(close_points_coordinate, close_points_list, close_points_color,
                                              "../cuff_vertices.json")

    if show:
        # 可视化添加后的模型
        o3d.visualization.draw_geometries([mesh])

    mesh_utils.save_triangle_mesh(mesh, output_path)


# 给模型添加uv展开所需要的两个fix点
def add_fix_info_to_obj(input_path, output_path):
    # 加载 OBJ 模型
    mesh = trimesh.load_mesh(input_path)

    # 获取顶点坐标
    vertices = mesh.vertices

    # 找到最左下角和最右上角的顶点
    min_vertex_index = np.argmin(vertices[:, 0] + vertices[:, 1])  # 最左下角
    max_vertex_index = np.argmax(vertices[:, 0] + vertices[:, 1])  # 最右上角

    min_vertex = vertices[min_vertex_index]
    max_vertex = vertices[max_vertex_index]

    # 打印找到的顶点
    print("最左下角顶点:", min_vertex)
    print("最右上角顶点:", max_vertex)

    # 创建 UV 坐标
    uv_coords = np.zeros((len(vertices), 2))  # 初始化 UV 坐标数组
    uv_coords[min_vertex_index] = [0.0, 0.0]  # 最左下角
    uv_coords[max_vertex_index] = [0.0, 1.0]  # 最右上角

    # 保存修改后的模型
    with open(output_path, 'w') as f:
        # 写入顶点
        for i, vertex in enumerate(vertices):
            if i == min_vertex_index:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} fix {uv_coords[i][0]} {uv_coords[i][1]}\n")
            elif i == max_vertex_index:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} fix {uv_coords[i][0]} {uv_coords[i][1]}\n")
            else:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # 写入 UV 坐标（如果需要）
        for uv in uv_coords:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        # 写入面
        for face in mesh.faces:
            f.write(f"f {' '.join([f'{v_idx + 1}/{v_idx + 1}' for v_idx in face])}\n")


#  提取位于圆柱体内的网格部分
def extract_vertices_and_faces_within_cylinder_from_center(mesh, center, axis, radius, height):
    # 根据圆柱中心点计算底面中心点
    axis = axis / np.linalg.norm(axis)  # 确保轴向单位化
    base_center = center - (axis * (height / 2.0))

    # 获取顶点和三角形面数据
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 找出所有在圆柱体范围内的顶点索引
    selected_indices = [
        i for i, vertex in enumerate(vertices)
        if geometry_utils.is_inside_cylinder(vertex, base_center, axis, radius, height)
    ]
    selected_set = set(selected_indices)

    # 创建索引映射，用于更新新模型的面索引
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    # 提取在范围内的三角形面
    new_triangles = []
    for tri in triangles:
        if all(idx in selected_set for idx in tri):
            new_triangles.append([index_map[idx] for idx in tri])

    # 创建新的顶点和面数据
    new_vertices = vertices[selected_indices]
    new_triangles = np.array(new_triangles)

    # 去除重复顶点，确保共享顶点
    unique_vertices, inverse_indices = np.unique(new_vertices, axis=0, return_inverse=True)

    # 更新面索引
    updated_triangles = inverse_indices[new_triangles]

    # 创建新的网格
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(unique_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(updated_triangles)

    # 保持原始的其他属性
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals[selected_indices])

    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[selected_indices])

    if mesh.has_textures():
        # 保持纹理坐标（如果存在）
        new_mesh.textures = mesh.textures

    new_mesh.compute_vertex_normals()
    return new_mesh


# 提取位于长方体内的网格部分
def extract_vertices_and_faces_within_box(mesh, cube_vertices):
    # 计算长方体的最小坐标和最大坐标
    box_min = np.min(cube_vertices, axis=0)
    box_max = np.max(cube_vertices, axis=0)

    # 获取顶点和三角形面数据
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # 找出所有在长方体范围内的顶点索引
    selected_indices = [i for i, vertex in enumerate(vertices)
                        if geometry_utils.is_inside_box(vertex, box_min, box_max)]
    selected_set = set(selected_indices)

    # 创建索引映射，用于更新新模型的面索引
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    # 提取在范围内的三角形面
    new_triangles = []
    for tri in triangles:
        if all(idx in selected_set for idx in tri):
            new_triangles.append([index_map[idx] for idx in tri])

    # 创建新的顶点和面数据
    new_vertices = vertices[selected_indices]
    new_triangles = np.array(new_triangles)

    # 去除重复顶点，确保共享顶点
    unique_vertices, inverse_indices = np.unique(new_vertices, axis=0, return_inverse=True)

    # 更新面索引
    updated_triangles = inverse_indices[new_triangles]

    # 创建新的网格
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(unique_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(updated_triangles)

    # 保持原始的其他属性
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals[selected_indices])

    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[selected_indices])

    if mesh.has_textures():
        # 保持纹理坐标（如果存在）
        new_mesh.textures = mesh.textures

    new_mesh.compute_vertex_normals()
    return new_mesh
