import numpy as np
import scipy
import matplotlib.pyplot as plt
from construct_graph.graph import Graph, Flat

class Random_Geometric_Graph(Graph, Flat):

    def __init__(self, num_Vs, edge_scaling=1, auto_plot=True, **kwargs):

        self.num_Vs = num_Vs
        self.dim = 2
        self.edge_scaling = edge_scaling
        self.construct_V_coords_and_interior_V_num()
        self.construct_E_lengths_by_v_num()
        self.wadjacency_matrix = self.construct_wadjacency_matrix()
        self.g_coords = self.construct_g_coords()
        if auto_plot:
            print(f"\n|V| = {self.num_Vs}")
            self.plot_graph(**kwargs)

    def construct_V_coords_and_interior_V_num(self):

        r = np.sqrt(np.random.uniform(0, 1, size=(self.num_Vs, 1)))
        theta = np.random.uniform(0, 2 * np.pi, size=(self.num_Vs, 1))
        V_coords = np.hstack((r * np.cos(theta), r * np.sin(theta)))

        self.V_coords = V_coords
        delaunay_triangulation = scipy.spatial.Delaunay(self.V_coords)
        boundary_V_num = delaunay_triangulation.convex_hull
        self.interior_V_num = np.setdiff1d(np.arange(self.num_Vs), boundary_V_num)

    def construct_E_lengths_by_v_num(self):

        E_length = self.edge_scaling * 1.2 * np.pi / (np.sqrt(self.num_Vs) - 1)
        kdtree = scipy.spatial.cKDTree(self.V_coords)

        vertices_within_distance = [kdtree.query_ball_point(vertex, E_length, return_sorted=True)[1:] for vertex in self.V_coords]
        
        E_lengths_by_v_num = {}

        for v_num, w_nums in enumerate(vertices_within_distance):
            for w_num in w_nums:
                if v_num < w_num:
                    E_lengths_by_v_num[v_num, w_num] = np.linalg.norm(self.V_coords[v_num] - self.V_coords[w_num])
        
        self.E_lengths_by_v_num = E_lengths_by_v_num

    def construct_wadjacency_matrix(self):

        wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs), dtype=np.float64)

        for v0_num, v1_num in self.E_lengths_by_v_num:

            wadjacency_matrix[v0_num, v1_num] = self.E_lengths_by_v_num[v0_num, v1_num]
            wadjacency_matrix[v1_num, v0_num] = self.E_lengths_by_v_num[v0_num, v1_num]

        return wadjacency_matrix.tocsc()

    def construct_edge_R_system(self):

        system = {} 
        system[0, 0] = scipy.sparse.lil_matrix((self.num_Vs, len(self.E_lengths_by_v_num)))
        system[0, 1] = scipy.sparse.lil_matrix((self.num_Vs, len(self.E_lengths_by_v_num)))
        system[1, 0] = scipy.sparse.lil_matrix((self.num_Vs, len(self.E_lengths_by_v_num)))
        system[1, 1] = scipy.sparse.lil_matrix((self.num_Vs, len(self.E_lengths_by_v_num)))

        for e_num, ((v_num, w_num), l_vw) in enumerate(self.E_lengths_by_v_num.items()):

            r_vw = self.calculate_r_vw(self.V_coords[v_num], self.V_coords[w_num])
            R_vw = self.update_R(r_vw, l_vw)

            system[0, 0][v_num, e_num] = R_vw[0, 0]
            system[0, 1][v_num, e_num] = R_vw[0, 1]
            system[1, 0][v_num, e_num] = R_vw[1, 0]
            system[1, 1][v_num, e_num] = R_vw[1, 1]

            system[0, 0][w_num, e_num] = R_vw[0, 0]
            system[0, 1][w_num, e_num] = R_vw[0, 1]
            system[1, 0][w_num, e_num] = R_vw[1, 0]
            system[1, 1][w_num, e_num] = R_vw[1, 1]

        return system
    
    def dfs(self, graph, node, visited):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                self.dfs(graph, neighbor, visited)

    def count_connected_components(self):

        edges = np.array(list(self.E_lengths_by_v_num.keys()))

        # Convert edges to an adjacency list representation
        num_vertices = np.max(edges) + 1
        graph = [[] for _ in range(num_vertices)]

        for edge in edges:
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        # Initialize visited array
        visited = np.zeros(num_vertices, dtype=bool)

        # Count connected components using DFS
        components_count = 0
        for vertex in range(num_vertices):
            if not visited[vertex]:
                self.dfs(graph, vertex, visited)
                components_count += 1

        return components_count