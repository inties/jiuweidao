import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_obj_model(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    return vertices

def compute_centroid(points):
    if points.size == 0:
        return None
    return np.mean(points, axis=0)

def slice_model(vertices, step=0.2, num_slices=5):
    centroid = compute_centroid(vertices)
    if centroid is None:
        raise ValueError("Model contains no vertices or is invalid.")
    print("centroid: ", centroid)
    slice_centroids = []
    
    # 向上切片
    for i in range(1, num_slices + 1):
        z_level = centroid[2] + i * step  # 沿 z 轴向上移动
        slice_points = vertices[np.abs(vertices[:, 2] - z_level) < step / 2]  # 提取 z 坐标附近的点
        centroid_slice = compute_centroid(slice_points)
        if centroid_slice is not None:
            slice_centroids.append(centroid_slice)
    
    # 质心位置切片
    slice_points = vertices[np.abs(vertices[:, 2] - centroid[2]) < step / 2]  # 提取质心 z 坐标附近的点
    centroid_slice = compute_centroid(slice_points)
    if centroid_slice is not None:
        slice_centroids.append(centroid_slice)
    
    # 向下切片
    for i in range(1, num_slices + 1):
        z_level = centroid[2] - i * step  # 沿 z 轴向下移动
        slice_points = vertices[np.abs(vertices[:, 2] - z_level) < step / 2]  # 提取 z 坐标附近的点
        centroid_slice = compute_centroid(slice_points)
        if centroid_slice is not None:
            slice_centroids.append(centroid_slice)
    
    if len(slice_centroids) == 0:
        raise ValueError("No valid slice centroids found.")
    
    return np.array(slice_centroids)

def fit_line(points, extension_factor=5):
    if points.size == 0:
        raise ValueError("No valid points provided for line fitting.")
    
    reg = LinearRegression()
    reg.fit(points[:, 1].reshape(-1, 1), points[:, [0, 2]])
    
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    extend_length = (max_y - min_y) * extension_factor
    extended_min_y = min_y - extend_length
    extended_max_y = max_y + extend_length
    
    y_vals = np.linspace(extended_min_y, extended_max_y, 100)
    xz_vals = reg.predict(y_vals.reshape(-1, 1))
    return np.column_stack((xz_vals[:, 0], y_vals, xz_vals[:, 1]))

def visualize_results(vertices, slice_centroids, fit_line_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.1, label='Model Points')
    ax.scatter(slice_centroids[:, 0], slice_centroids[:, 1], slice_centroids[:, 2], color='red', label='Slice Centroids')
    ax.plot(fit_line_points[:, 0], fit_line_points[:, 1], fit_line_points[:, 2], color='blue', linewidth=2, label='Fitted Line')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    file_path = "wzpdata\\teeth2\\source\\teeth2.ply"
    vertices = load_obj_model(file_path)
    slice_centroids = slice_model(vertices, step=0.2, num_slices=6)
    fit_line_points = fit_line(slice_centroids, extension_factor=2)
    visualize_results(vertices, slice_centroids, fit_line_points)