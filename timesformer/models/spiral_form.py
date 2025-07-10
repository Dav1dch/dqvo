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
            current_row += directions[current_direction][0]
            current_col += directions[current_direction][1]

    # Flatten the matrix into a list
    spiral_order = []
    for row in matrix:
        spiral_order.extend(row)
    return spiral_order
