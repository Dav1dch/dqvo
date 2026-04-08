import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Bresenham直线算法"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    
    points.append((x1, y1))
    return points

def generate_optimized_octagon_spiral(m: int, n: int, skip_outer: bool = False) -> List[int]:
    """
    优化的八边形螺旋路径生成器
    返回一维索引列表
    
    参数:
        m: 行数
        n: 列数
        skip_outer: 是否跳过最外层
        
    返回:
        一维索引列表，对应扫描顺序
    """
    path_2d = []  # 二维坐标路径
    visited = set()
    
    # 计算层数
    layers = min(m, n) // 2
    
    # 起始层
    start_layer = 0 if not skip_outer else 1
    
    # 逐层生成八边形
    for layer in range(start_layer, layers):
        # 当前层边界
        top = layer
        bottom = m - 1 - layer
        left = layer
        right = n - 1 - layer
        
        # 如果只剩中心点
        if top >= bottom or left >= right:
            if top <= bottom and left <= right:
                # 添加中心区域的所有点
                for r in range(top, bottom + 1):
                    for c in range(left, right + 1):
                        if (r, c) not in visited:
                            path_2d.append((r, c))
                            visited.add((r, c))
            break
        
        # 计算八边形顶点
        # 对于八边形，我们需要切除四个角
        cut_h = max(1, (bottom - top + 1) // 4)  # 水平切除量
        cut_v = max(1, (right - left + 1) // 4)  # 垂直切除量
        
        # 定义八边形的8条边
        edges = [
            # (起点, 终点, 方向)
            ((top + cut_h, left), (top, left + cut_v), (-1, 1)),     # 左上边
            ((top, left + cut_v), (top, right - cut_v), (0, 1)),     # 上边
            ((top, right - cut_v), (top + cut_h, right), (1, 1)),    # 右上边
            ((top + cut_h, right), (bottom - cut_h, right), (1, 0)), # 右边
            ((bottom - cut_h, right), (bottom, right - cut_v), (1, -1)), # 右下边
            ((bottom, right - cut_v), (bottom, left + cut_v), (0, -1)),  # 下边
            ((bottom, left + cut_v), (bottom - cut_h, left), (-1, -1)),  # 左下边
            ((bottom - cut_h, left), (top + cut_h, left), (-1, 0)),  # 左边
        ]
        
        # 生成当前层的八边形
        layer_points = []
        
        for i, (start, end, direction) in enumerate(edges):
            # 使用Bresenham算法连接起点和终点
            line_points = bresenham_line(start[0], start[1], end[0], end[1])
            
            # 添加点（避免重复添加终点）
            for point in line_points:
                if point not in visited:
                    visited.add(point)
                    layer_points.append(point)
        
        # 将当前层添加到路径
        if layer == start_layer:
            # 第一层直接添加
            path_2d.extend(layer_points)
        else:
            # 内层：需要从外层连接到内层
            if path_2d and layer_points:
                # 连接外层和内层
                last_point = path_2d[-1]
                first_point = layer_points[0]
                
                # 计算连接线
                connector = bresenham_line(last_point[0], last_point[1],
                                          first_point[0], first_point[1])
                
                # 添加连接线（跳过第一个点，已经是外层终点）
                for point in connector[1:]:
                    if point not in visited:
                        visited.add(point)
                        path_2d.append(point)
            
            # 添加内层八边形
            path_2d.extend(layer_points)
    
    # 将二维坐标转换为一维索引
    path_1d = [r * n + c for (r, c) in path_2d]
    
    return path_1d
def generate_spiral_index_tensor(m, n, re):
    print(m, n)
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    
    # 首先按正常顺序填充矩阵
    for i in range(m):
        for j in range(n):
            matrix[i][j] = i * n + j
    
    # 然后按螺旋顺序访问这些位置
    spiral_order = []
    top, bottom = 0, m - 1
    left, right = 0, n - 1
    
    while top <= bottom and left <= right:
        # 从左到右访问上边界
        for col in range(left, right + 1):
            spiral_order.append(matrix[top][col])
        top += 1
        
        # 从上到下访问右边界
        for row in range(top, bottom + 1):
            spiral_order.append(matrix[row][right])
        right -= 1
        
        # 从右到左访问下边界（如果还有行）
        if top <= bottom:
            for col in range(right, left - 1, -1):
                spiral_order.append(matrix[bottom][col])
            bottom -= 1
        
        # 从下到上访问左边界（如果还有列）
        if left <= right:
            for row in range(bottom, top - 1, -1):
                spiral_order.append(matrix[row][left])
            left += 1
    # print(spiral_order)
    # if re:
    #     spiral_order.reverse()
    
    return spiral_order


def generate_spiral_index_tensor_2(m, n):
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    current_direction = 0
    current_row, current_col = 0, 0
    count = 0

    for _ in range(m * n):
        matrix[current_row][current_col] = count
        count += 1
        next_row = current_row + directions[current_direction][0]
        next_col = current_col + directions[current_direction][1]

        if 0 <= next_row < m and 0 <= next_col < n and matrix[next_row][next_col] == 0:
            current_row, current_col = next_row, next_col
        else:
            current_direction = (current_direction + 1) % 4
            # 问题在这里：直接移动，没有检查新位置是否有效
            current_row += directions[current_direction][0]
            current_col += directions[current_direction][1]

    # Flatten the matrix into a list
    spiral_order = []
    for row in matrix:
        spiral_order.extend(row)
    # spiral_order.reverse()
    return spiral_order


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Bresenham直线算法"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    
    points.append((x1, y1))
    return points

def generate_octagon_boundary_spiral(m: int, n: int, skip_outer: bool = False) -> List[Tuple[int, int]]:
    """
    生成八边形边界螺旋路径（只走边界，不填充内部）
    
    参数:
        m: 行数
        n: 列数
        skip_outer: 是否跳过最外层
    """
    path = []
    visited = set()
    
    # 计算最大层数（八边形的层数）
    # 每层八边形需要至少3x3的空间
    max_layers = min((m + 1) // 2, (n + 1) // 2)
    
    # 起始层
    start_layer = 1 if skip_outer else 0
    
    # 存储每一层八边形的边界点
    layer_boundaries = []
    
    for layer in range(start_layer, max_layers):
        # 当前层边界
        top = layer
        bottom = m - 1 - layer
        left = layer
        right = n - 1 - layer
        
        # 检查是否还有空间形成八边形
        if bottom - top < 2 or right - left < 2:
            # 太小，直接添加剩余点
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if (r, c) not in visited:
                        path.append((r, c))
                        visited.add((r, c))
            continue
        
        # 计算八边形的8个顶点
        # 水平缩进：切除四个角形成八边形
        h_indent = max(1, (bottom - top) // 3)
        v_indent = max(1, (right - left) // 3)
        
        # 定义八边形的8个顶点（逆时针方向）
        vertices = [
            (top + h_indent, left),           # 左中
            (top, left + v_indent),            # 左上
            (top, right - v_indent),           # 右上
            (top + h_indent, right),           # 右中
            (bottom - h_indent, right),        # 右下
            (bottom, right - v_indent),        # 下右
            (bottom, left + v_indent),         # 下左
            (bottom - h_indent, left),         # 左下
            (top + h_indent, left)             # 闭合
        ]
        
        # 生成当前层的八边形边界
        current_layer_points = []
        for i in range(8):
            start_r, start_c = vertices[i]
            end_r, end_c = vertices[i + 1]
            
            line_points = bresenham_line(start_r, start_c, end_r, end_c)
            # 添加点（不包括最后一个点，避免重复）
            for point in line_points[:-1]:
                if point not in visited:
                    current_layer_points.append(point)
                    visited.add(point)
        
        # 将当前层添加到路径
        # 如果是第一层，直接添加
        # 如果是内层，需要连接到外层
        if layer == start_layer:
            path.extend(current_layer_points)
        else:
            # 从外层终点连接到内层起点
            if path and current_layer_points:
                # 添加连接线
                last_outer = path[-1]
                first_inner = current_layer_points[0]
                connector = bresenham_line(last_outer[0], last_outer[1], 
                                          first_inner[0], first_inner[1])
                for point in connector[1:]:  # 跳过起点（已在外层）
                    if point not in visited:
                        path.append(point)
                        visited.add(point)
            
            # 添加内层八边形
            path.extend(current_layer_points)
        
        layer_boundaries.append(current_layer_points)
    
    return path

def generate_smooth_octagon_spiral(m: int, n: int, skip_outer: bool = False) -> List[Tuple[int, int]]:
    """
    生成平滑的八边形螺旋路径（连续扫描边界）
    这个算法确保路径连续且不形成Z形
    """
    path = []
    
    # 创建访问矩阵
    visited = [[False] * n for _ in range(m)]
    
    # 如果需要跳过外层，标记外层为已访问
    if skip_outer:
        for i in range(m):
            visited[i][0] = True
            visited[i][n-1] = True
        for j in range(n):
            visited[0][j] = True
            visited[m-1][j] = True
    
    # 定义八边形的8个方向（顺时针）
    directions = [
        (0, 1),   # 右
        (1, 1),   # 右下
        (1, 0),   # 下
        (1, -1),  # 左下
        (0, -1),  # 左
        (-1, -1), # 左上
        (-1, 0),  # 上
        (-1, 1)   # 右上
    ]
    
    # 计算起始点
    if skip_outer:
        # 跳过外层，从内部开始
        r, c = 1, 1
    else:
        # 从左上角开始
        r, c = 0, 0
    
    # 确保起始点有效
    if visited[r][c]:
        # 寻找第一个未访问的点
        for i in range(m):
            for j in range(n):
                if not visited[i][j]:
                    r, c = i, j
                    break
            else:
                continue
            break
    
    # 添加起始点
    path.append((r, c))
    visited[r][c] = True
    
    # 初始方向
    dir_idx = 0
    
    # 螺旋参数
    steps_taken = 0
    steps_needed = 1
    segment = 0  # 当前边（八边形有8条边）
    
    while True:
        # 尝试沿当前方向移动
        dr, dc = directions[dir_idx]
        nr, nc = r + dr, c + dc
        
        # 检查下一个点是否有效
        if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
            r, c = nr, nc
            path.append((r, c))
            visited[r][c] = True
            steps_taken += 1
            
            # 检查是否完成当前边
            if steps_taken >= steps_needed:
                # 改变方向
                dir_idx = (dir_idx + 1) % 8
                steps_taken = 0
                segment += 1
                
                # 每完成两个边，增加步长（八边形的特性）
                if segment % 2 == 0:
                    steps_needed += 1
        else:
            # 改变方向
            dir_idx = (dir_idx + 1) % 8
            steps_taken = 0
            segment += 1
            
            # 检查新方向
            dr, dc = directions[dir_idx]
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                r, c = nr, nc
                path.append((r, c))
                visited[r][c] = True
                steps_taken += 1
        
        # 检查是否所有点都被访问
        all_visited = all(all(row) for row in visited)
        if all_visited:
            break
        
        # 安全措施：防止无限循环
        if len(path) > m * n * 2:
            break
    
    return path

def generate_spiral_index_tensor_2(m, n):
    print(m, n)
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    
    # 首先按正常顺序填充矩阵
    for i in range(m):
        for j in range(n):
            matrix[i][j] = i * n + j
    
    # 然后按螺旋顺序访问这些位置
    spiral_order = []
    top, bottom = 0, m - 1
    left, right = 0, n - 1
    
    while top <= bottom and left <= right:
        # 从左到右访问上边界
        for col in range(left, right + 1):
            spiral_order.append(matrix[top][col])
        top += 1
        
        # 从上到下访问右边界
        for row in range(top, bottom + 1):
            spiral_order.append(matrix[row][right])
        right -= 1
        
        # 从右到左访问下边界（如果还有行）
        if top <= bottom:
            for col in range(right, left - 1, -1):
                spiral_order.append(matrix[bottom][col])
            bottom -= 1
        
        # 从下到上访问左边界（如果还有列）
        if left <= right:
            for row in range(bottom, top - 1, -1):
                spiral_order.append(matrix[row][left])
            left += 1
    # print(spiral_order)
    # spiral_order.reverse()
    
    return spiral_order


def generate_spiral_index_tensor(m, n):
    matrix = [[0 for _ in range(n)] for _ in range(m)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    current_direction = 0
    current_row, current_col = 0, 0
    count = 0

    for _ in range(m * n):
        matrix[current_row][current_col] = count
        count += 1
        next_row = current_row + directions[current_direction][0]
        next_col = current_col + directions[current_direction][1]

        if 0 <= next_row < m and 0 <= next_col < n and matrix[next_row][next_col] == 0:
            current_row, current_col = next_row, next_col
        else:
            current_direction = (current_direction + 1) % 4
            # 问题在这里：直接移动，没有检查新位置是否有效
            current_row += directions[current_direction][0]
            current_col += directions[current_direction][1]

    # Flatten the matrix into a list
    spiral_order = []
    for row in matrix:
        spiral_order.extend(row)
    # spiral_order.reverse()
    return spiral_order