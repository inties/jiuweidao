"""
(1) 功能说明：几何工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import numpy as np
from matplotlib import pyplot as plt


# 3. 计算每个三角形的面积
def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


# 计算两个向量的夹角（以度为单位）
def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 保证数值不超出 [-1, 1]
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


# 计算三角形的法向量
def compute_normal(p1, p2, p3):
    AB = p2 - p1
    AC = p3 - p1
    normal = np.cross(AB, AC)
    return normal


# 计算每个点到平面的距离
def point_to_plane_distance(point, normal_vector, d):
    return np.abs(np.dot(normal_vector, point) - d) / np.linalg.norm(normal_vector)


# 计算两点之间的欧几里得距离
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# 将三维点集投影到一个点法式三维平面上
def project_points_to_plane(points, plane_normal, plane_point=[0, 0, 0]):
    # 将点和法向量转换为 numpy 数组
    points = np.array(points)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)

    # 计算投影
    projected_points_dict = {}
    for point in points:
        # 计算点到平面的向量
        vector_to_plane = point - plane_point

        # 计算点到平面的法向量分量
        distance_to_plane = np.dot(vector_to_plane, plane_normal) / np.linalg.norm(plane_normal) ** 2

        # 投影点
        projected_point = point - distance_to_plane * plane_normal
        # 投影点和原始点键值对
        projected_points_dict[tuple(point)] = tuple(projected_point)
    return projected_points_dict


# 使用回归方式拟合一条曲线
def regression(P, vertices):
    t = np.linspace(0, 2 * np.pi, len(P))  # 统一的参数 t，可以基于点的顺序
    x = np.array([p[0] for p in P])  # 提取 x 维度
    y = np.array([p[1] for p in P])  # 提取 y 维度
    z = np.array([p[2] for p in P])  # 提取 z 维度

    # 对每个维度进行多项式回归拟合
    deg = 3  # 多项式阶数，根据数据复杂度调整
    coeffs_x = np.polyfit(t, x, deg)
    coeffs_y = np.polyfit(t, y, deg)
    coeffs_z = np.polyfit(t, z, deg)

    # 生成三维曲线
    x_fit = np.polyval(coeffs_x, t)
    y_fit = np.polyval(coeffs_y, t)
    z_fit = np.polyval(coeffs_z, t)

    # 可视化拟合结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点集
    ax.scatter(x, y, z, color='r', label='原始点')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', label='Model', s=1)
    # 绘制拟合曲线
    ax.plot(x_fit, y_fit, z_fit, label='拟合曲线', color='b')

    # 添加标签
    ax.legend()

    plt.show()


# 投影到直线的函数，返回一个字典
def project_points_to_line(points, line_direction, line_point=[0, 0, 0]):
    # 确保方向向量是单位向量（标准化）
    line_direction = np.array(line_direction)
    line_direction = line_direction / np.linalg.norm(line_direction)

    line_point = np.array(line_point)
    projection_dict = {}

    for point in points:
        # 计算投影系数 t
        v = np.array(point) - line_point
        t = np.dot(v, line_direction)  # 计算投影系数

        # 计算投影点
        projection = line_point + t * line_direction

        # 将投影点和原始点作为键值对存入字典
        projection_dict[tuple(projection)] = tuple(point)

    return projection_dict


# 判断一个点是否在圆柱体内
def is_inside_cylinder(vertex, center, axis, radius, height):
    # 归一化轴向向量
    axis = axis / np.linalg.norm(axis)

    # 点到中心的向量
    diff = vertex - center

    # 投影到轴向上的高度
    proj_height = np.dot(diff, axis)

    # 判断高度是否在范围内
    if proj_height < 0 or proj_height > height:
        return False

    # 计算点到轴的水平距离
    radial_distance = np.linalg.norm(diff - proj_height * axis)

    # 判断水平距离是否在半径范围内
    return radial_distance <= radius


# 检查顶点是否在长方体内
def is_inside_box(vertex, box_min, box_max):
    return (box_min[0] <= vertex[0] <= box_max[0] and
            box_min[1] <= vertex[1] <= box_max[1] and
            box_min[2] <= vertex[2] <= box_max[2])
