"""
(1) 功能说明：牙齿袖口的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from utils import geometry_utils, draw_utils, direction_utils, mesh_utils
import numpy as np
import vtk


# 计算相邻三角形的法向量夹角，并保存共享顶点
def calculate_adjacent_face_angles_and_shared_vertices(vertices, triangles, triangles_normals):
    # 为每条边创建一个字典，记录与其共享的面
    edge_to_faces = {}
    for i, triangle in enumerate(triangles):
        # 获取三角形的三条边
        edges = [
            tuple(sorted([triangle[0], triangle[1]])),
            tuple(sorted([triangle[1], triangle[2]])),
            tuple(sorted([triangle[2], triangle[0]]))
        ]

        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(i)

    # 记录共享顶点
    shared_vertices = []
    for edge, face_indices in edge_to_faces.items():
        if len(face_indices) == 2:  # 只有当一个边被两个三角形共享时才考虑
            tri1_idx, tri2_idx = face_indices
            # 计算法向量夹角
            angle = geometry_utils.angle_between_vectors(triangles_normals[tri1_idx], triangles_normals[tri2_idx])
            # 如果夹角超过20度，保存共享的两个顶点
            if angle > 20:
                shared_vertices.append(edge)

    # 提取共享顶点的点
    points = []
    for edge in shared_vertices:
        points.extend([vertices[edge[0]], vertices[edge[1]]])

    points = np.unique(points, axis=0)  # 去除重复点
    return points


# 根据曲率找点
def find_close_points_by_curvature(path_model):
    # 1. 读取OBJ模型
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path_model)
    reader.Update()

    curvature_filter = vtk.vtkCurvatures()
    curvature_filter.SetInputData(reader.GetOutput())

    curvature_filter.SetCurvatureTypeToMean()  # 默认使用平均曲率
    # curvature_filter.SetCurvatureTypeToGaussian()  # 计算高斯曲率
    curvature_filter.Update()

    curvature_output = curvature_filter.GetOutput()

    # 计算得到的曲率值对应的顶点数量
    num_points = curvature_output.GetNumberOfPoints()

    # 查找曲率变化大的顶点
    cuff_points = []
    for i in range(num_points):
        curvature_value = curvature_output.GetPointData().GetScalars().GetValue(i)
        if curvature_value > 0.8:  # 设定阈值
            cuff_points.append(curvature_output.GetPoint(i))

    return cuff_points


# 根据扫描杆排除点
def find_close_points_by_plane_distance(points, path_scanning_rod):
    center_point = np.mean(points, axis=0)

    # 平面方程参数
    point_on_plane = np.array(center_point)  # 平面上的一点
    direction = direction_utils.get_direction_vector_by_pca(path_scanning_rod, False)
    normal_vector = np.array(direction)  # 平面的法向量
    d = np.dot(normal_vector, point_on_plane)  # 计算平面方程的常数d

    # 找到所有距离小于0.5的点
    close_points = []
    for point in points:
        distance = geometry_utils.point_to_plane_distance(point, normal_vector, d)
        if distance < 0.5:
            close_points.append(point)

    return close_points


# 根据质心排除点
def find_close_points_by_distance(points):
    close_points2 = []
    center_point2 = np.mean(points, axis=0)

    r = np.mean(np.linalg.norm(points - center_point2, axis=1))
    for point in points:
        if abs(np.linalg.norm(point - center_point2) - r) < 0.1:
            close_points2.append(point)

    return close_points2


# 根据投影排除点
def find_close_points_by_project(points, center, normal, point=[0, 0, 0], show=False):
    points.append(center)
    projected_points = geometry_utils.project_points_to_plane(points, normal, point)
    projected_points = np.array(list(projected_points.values()))

    if show:
        # 创建一个 3D 图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], color='b', s=1)

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    projected_points2 = projected_points[:-1]
    close_points = []
    r = np.mean(np.linalg.norm(projected_points2 - projected_points[-1], axis=1))

    for i, p in enumerate(projected_points2):
        if abs(np.linalg.norm(p - projected_points[-1]) - r) < 0.5:
            close_points.append(points[i])

    return close_points


# 通过聚类排除点
def find_close_points_by_cluster(points, line_direction, line_point=[0, 0, 0], show=False):
    # 将三维点集投影到直线
    projected_and_origin_points = geometry_utils.project_points_to_line(points, line_direction, line_point)

    projected_points = list(projected_and_origin_points.keys())

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=0.2, min_samples=2)  # 根据实际情况调整 eps 和 min_samples
    labels = dbscan.fit_predict(projected_points)

    # 可视化聚类结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if show:
        # 根据 DBSCAN 标签绘制不同颜色的点
        unique_labels = set(labels)
        for label in unique_labels:
            cluster_points = []
            for i, l in enumerate(labels):
                if l == label:
                    cluster_points.append(projected_points[i])
            cluster_points = np.array(cluster_points)
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {label}")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    # 统计每个簇的大小
    label_counts = Counter(labels)  # 统计每个标签的数量

    # 找到类别最多的簇的标签（排除噪声 -1）
    most_common_label = max(label_counts, key=lambda x: label_counts[x] if x != -1 else -1)

    largest_cluster_points = []
    # 如果有簇（不是噪声），选择该簇的点
    if most_common_label != -1:
        # 提取最大簇的所有点
        for i, l in enumerate(labels):
            if l == most_common_label:
                largest_cluster_points.append(projected_points[i])

    close_points = [projected_and_origin_points[p] for p in largest_cluster_points]

    return close_points


# 根据法向量夹角、扫描杆、质心点找到袖口顶点
def visualize_cuff_vertices_by_geometry(path_scanning_rod, path_model, show=False):
    # 加载三维网格模型
    mesh = mesh_utils.get_mesh(path_model)

    # 获取三角形的顶点和面
    triangles = mesh_utils.get_mesh_triangles(mesh)
    vertices = mesh_utils.get_mesh_vertices(mesh)
    triangles_normals = mesh_utils.get_mesh_triangle_normals(mesh)

    # 计算相邻三角形的法向量夹角，并保存共享顶点
    points = calculate_adjacent_face_angles_and_shared_vertices(vertices, triangles, triangles_normals)
    if show:
        draw_utils.draw_geometries(mesh, draw_utils.draw_points(points))

    # 根据扫描杆排除点
    close_points = find_close_points_by_plane_distance(points, path_scanning_rod)
    if show:
        draw_utils.draw_geometries(mesh, draw_utils.draw_points(close_points))

    # 根据质心排除点
    close_points2 = find_close_points_by_distance(close_points)
    if show:
        draw_utils.draw_geometries(mesh, draw_utils.draw_points(close_points2))

    return np.asarray(close_points2)


# 最近邻排序函数，加入最小距离检查
def nearest_neighbor_sort_with_min_distance_threshold(points):
    if not isinstance(points, np.ndarray):
        points = np.ndarray(points)
    min_distance = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1), axis=0)

    # 初始化已排序点集
    sorted_points = [points[0]]  # 从第一个点开始
    points = points[1:].tolist()  # 剩余点集（不包括已排序的第一个点）

    # 遍历剩余的点集，选择离当前点最近的点
    while points:
        current_point = sorted_points[-1]  # 当前的最后一个点

        if len(sorted_points) > 1:
            angels = []
            distances = []
            current_direction = np.array(sorted_points[-2]) - np.array(current_point)
            for point in points:
                # 计算剩余点集到当前点的距离
                distances.append(geometry_utils.euclidean_distance(np.array(current_point), np.array(point)))
                # 计算当前走向和即将选择下一个点的走向的角度
                angels.append(
                    geometry_utils.angle_between_vectors(current_direction, np.array(current_point - np.array(point))))
            # 检查最小距离，如果最小距离都大于 min_distance，则抛弃剩余点
            if min(distances) > min_distance:
                print(f"剩余点的最小距离都大于 {min_distance}，抛弃剩余点，停止排序。")
                break
            # 距离排序索引
            dis_indices = np.argsort(distances)
            find_point = False
            for i in dis_indices:
                if angels[i] < 50 and distances[i] < min_distance / 2:
                    if distances[dis_indices[0]] > geometry_utils.euclidean_distance(current_point, sorted_points[0]):
                        break
                    # 最小距离对应的点
                    nearest_point = points[i]
                    find_point = True
                    break
            if not find_point:
                if distances[dis_indices[0]] > geometry_utils.euclidean_distance(current_point, sorted_points[0]):
                    break
                nearest_point = points[dis_indices[0]]

        else:
            # 计算剩余点集到当前点的距离
            distances = [geometry_utils.euclidean_distance(np.array(current_point), np.array(point)) for point in
                         points]
            # 检查最小距离，如果最小距离都大于 min_distance，则抛弃剩余点
            if min(distances) > min_distance:
                print(f"剩余点的最小距离都大于 {min_distance}，抛弃剩余点，停止排序。")
                break
            # 找到当前点最近的点
            nearest_point = points[np.argmin(distances)]

        sorted_points.append(nearest_point)  # 加入已排序点集
        points.remove(nearest_point)  # 从剩余点集中移除该点
    sorted_points.append(sorted_points[0])
    return np.array(sorted_points)


def show_cuff(res, vertices, show_cuff_and_vertices=False):
    # 检查点集形状
    if res.ndim != 2 or res.shape[1] != 3:
        raise ValueError("输入点集必须是 (N, 3) 的二维数组！")

    # 确保点数足够多
    if len(res) < 4:
        raise ValueError("输入点集太少，至少需要4个点！")

    # 分离坐标
    x, y, z = res[:, 0], res[:, 1], res[:, 2]

    # 插值拟合
    tck, u = splprep([x, y, z], s=0, per=True)  # per=True 确保曲线闭合
    u_fine = np.linspace(0, 1, 100)  # 生成参数
    x_fine, y_fine, z_fine = splev(u_fine, tck)  # 拟合曲线

    # 绘制原始点和拟合曲线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'ro', label="原始点", markersize=1)  # 原始点
    ax.plot(x_fine, y_fine, z_fine, 'r-', label="拟合曲线")  # 拟合曲线

    # 绘制三维模型（OBJ顶点）
    if show_cuff_and_vertices:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', label='Model', s=1)
    ax.legend()
    plt.show()

    geometry_utils.regression(res, vertices)
