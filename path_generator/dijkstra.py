import heapq


def dijkstra(graph: list[list[tuple[float, int]]], banned: list[int], start: int, end: int) -> list[int]:
    N = len(graph)
    INF = 10 ** 20
    dists = [INF] * N
    prev = [-1] * N
    vis = [False] * N
    for ban in banned:
        vis[ban] = True
    if vis[start] or vis[end]:
        return None

    dists[start] = 0
    q = [(0, start)]
    while len(q) > 0:
        dis, node = heapq.heappop(q)
        if dis > dists[node]:
            continue
        vis[node] = True
        if node == end:
            break
        for edge_weight, neighbor in graph[node]:
            if vis[neighbor]:
                continue
            alt = edge_weight + dis
            if alt >= dists[neighbor]:
                continue
            dists[neighbor] = alt
            prev[neighbor] = node
            heapq.heappush(q, (alt, neighbor))
    else:
        return None

    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path