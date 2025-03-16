"""
(1) 功能说明：json解析工具函数，为result中的结果函数服务
(2) 开发团队：电子科技大学数字媒体技术团队
(3) 作者：晏小虎
(4) 联系方式:1009662182
(5) 创建日期：2025-1-14
(6) 重要修改：
"""
import json


def save_cuff_vertices_to_json(positions, vertices, colors, json_filename):
    # 将数据结构化为字典（可以根据需要调整结构）
    vertices_data = []
    if len(positions) != len(vertices) or len(vertices) != len(colors):
        raise ValueError("数据信息不一致")
    for i in range(len(positions)):
        vertex = {
            'position': positions[i],
            'vertice': vertices[i],
            'color': colors[i]
        }
        vertices_data.append(vertex)

    # 将数据写入JSON文件
    with open(json_filename, 'w') as json_file:
        json.dump(vertices_data, json_file, indent=4)
