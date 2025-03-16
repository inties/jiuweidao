import math

def calculate_direction_vector(point1, point2):
    """计算二维方向向量"""
    x1, y1 = point1
    x2, y2 = point2
    return (x2 - x1, y2 - y1)

def calculate_direction_vector_3d(point1, point2):
    """计算三维方向向量"""
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return (x2 - x1, y2 - y1, z2 - z1)

def normalize_vector_2d(dx, dy):
    """归一化二维向量"""
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        raise ValueError("方向向量的模为零，无法归一化。")
    return (dx / magnitude, dy / magnitude)

def normalize_vector_3d(dx, dy, dz):
    """归一化三维向量"""
    magnitude = math.sqrt(dx**2 + dy**2 + dz**2)
    if magnitude == 0:
        raise ValueError("方向向量的模为零，无法归一化。")
    return (dx / magnitude, dy / magnitude, dz / magnitude)

# 示例使用
if __name__ == "__main__":
    # 二维示例
  
    # 三维示例
    point1_3d = (1, 2, 3)
    point2_3d = (4, 6, 8)
    dx_3d, dy_3d, dz_3d = calculate_direction_vector_3d(point1_3d, point2_3d)
    unit_vector_3d = normalize_vector_3d(dx_3d, dy_3d, dz_3d)
    print(f"三维单位向量: {unit_vector_3d}")