"""
(1) 功能说明：单牙切割的工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import json
import numpy as np
import trimesh


# 通过标注文件切割单牙并保存模型
def cut_teeth(obj_file, json_file, teeth_num, output_dir):
    # 文件路径
    output_file = output_dir + "/tooth_" + str(teeth_num) + ".obj"

    # 读取 JSON 文件
    with open(json_file, "r") as f:
        data = json.load(f)

    # 获取顶点标签列表
    labels = data["labels"]

    # 加载 OBJ 文件
    mesh = trimesh.load(obj_file, file_type='obj')

    # 获取所有顶点和面
    vertices = mesh.vertices
    faces = mesh.faces

    # 找到属于 31 号牙齿的顶点索引
    tooth_indices = [i for i, label in enumerate(labels) if label == teeth_num]

    # 筛选面，只保留全部顶点属于 31 的面
    tooth_faces = []
    for face in faces:
        if all(v in tooth_indices for v in face):
            tooth_faces.append(face)

    # 重构为新的网格
    tooth_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(tooth_faces)
    )

    # 保存为新的 OBJ 文件
    tooth_mesh.export(output_file)

    print(f"{teeth_num}号牙齿网格已保存到 {output_file}")
