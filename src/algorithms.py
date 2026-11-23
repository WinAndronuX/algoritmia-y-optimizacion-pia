import math


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def tour_cost(tour, points):
    n = len(tour)
    return sum(dist(points[tour[i]], points[tour[(i + 1) % n]]) for i in range(n))


def nearest_neighbor_tour(points):

    n = len(points)
    unvisited = set(range(n))

    current = 0
    unvisited.remove(current)

    tour = [current]

    while unvisited:

        nxt = min(unvisited, key=lambda j: dist(points[current], points[j]))
        tour.append(nxt)

        unvisited.remove(nxt)
        current = nxt

    return tour


def two_opt(tour, points):
    """Hill-climbing 2-opt local search: repeat until no improving move."""

    n = len(tour)
    if n < 4:
        return tour[:]

    tour = tour[:]
    improved = True

    while improved:

        improved = False

        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue

                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]

                delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
                    dist(points[a], points[b]) + dist(points[c], points[d])
                )

                if delta < -1e-12:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True

    return tour


def two_opt_hill_climbing(points, rng=None):
    initial = nearest_neighbor_tour(points)
    return two_opt(initial, points)


def simulated_annealing_2opt(points, rng, t0=15.0, alpha=0.995, iters=None):
    n = len(points)

    if iters is None:
        iters = max(500, 150 * n)

    tour = nearest_neighbor_tour(points)
    cost = tour_cost(tour, points)

    best_tour, best_cost = tour[:], cost
    t = t0

    for _ in range(iters):
        if n < 4:
            break

        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)

        if i == 0 and j == n - 1:
            continue

        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]

        delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
            dist(points[a], points[b]) + dist(points[c], points[d])
        )

        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-9)):

            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            cost += delta

            if cost < best_cost:
                best_tour, best_cost = tour[:], cost

        t *= alpha

    return best_tour


def mst_tour(points):

    n = len(points)
    in_tree = [False] * n

    min_edge = [math.inf] * n
    parent = [-1] * n

    min_edge[0] = 0.0
    adj = [[] for _ in range(n)]

    for _ in range(n):

        u = -1
        for v in range(n):
            if not in_tree[v] and (u == -1 or min_edge[v] < min_edge[u]):
                u = v
        in_tree[u] = True

        if parent[u] != -1:
            adj[u].append(parent[u])
            adj[parent[u]].append(u)

        for v in range(n):
            if not in_tree[v]:

                d = dist(points[u], points[v])
                if d < min_edge[v]:
                    min_edge[v] = d
                    parent[v] = u

    tour, visited = [], [False] * n

    def dfs(u):
        visited[u] = True
        tour.append(u)

        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(0)

    return tour


def mst_two_opt(points, rng=None):
    base = mst_tour(points)
    return two_opt(base, points)


ALGORITHMS = {
    "2opt_hill_climbing": two_opt_hill_climbing,
    "sa_2opt": simulated_annealing_2opt,
    "mst_2opt": mst_two_opt,
}
