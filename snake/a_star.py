import heapq

def a_star_search(start, goal, grid):
    grid_width = len(grid)
    grid_height = len(grid[0])

    # Define movement directions
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

    # Open list implemented as a priority queue
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        current_priority, current = heapq.heappop(open_list)

        if current == goal:
            break

        for dx, dy in neighbors:
            x2, y2 = current[0] + dx, current[1] + dy
            if 0 <= x2 < grid_width and 0 <= y2 < grid_height:
                if grid[x2][y2] == 1:
                    continue  # Obstacle
                new_cost = cost_so_far[current] + 1  # Assume cost between cells is 1
                next_node = (x2, y2)
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal, next_node)
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current

    # Reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []  # No path found
    path.reverse()
    return path

def heuristic(a, b):
    # Use Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
