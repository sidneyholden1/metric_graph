"""
Patch takes a "total" graph. Takes out a 1x1 patch. Periodizes the patch. And solves the corresponding corrector problem.
The types of total_graph's are:
    - Truncated_Square
    - Random_Delaunay
These total_graph's need to have:
    - total_num_Vs    : total number of vertices (total = in and out of 1x1 square).
    - num_Vs          : number of vertices in 1x1 square
    - total_V_coords  : coordinates of total vertices. First num_Vs are those in the 1x1square
    - total_edges     : total edges between total vertices where edge (v_index, w_index) satisfies v_index < w_index

e.g.:
num_Vs = 100
total_graph = Random_Delaunay(num_Vs)
g = Patch(total_graph)
eq = Cell_Problem(g)
xi = eq.solve_corrector_equation()
coefficient = xi.construct_homogenized_tensor(xi)
print(coefficient)
"""

import numpy as np
import scipy
from construct_graph.graph import Graph, Flat
import sparseqr

class Patch(Graph, Flat):

    """Parent class of more specific graph classes (spiderweb, delaunay triangulations etc.)
    These child classes need the attributes:
        - num_Vs
        - V_coords
        - E_lengths_by_v_num
        - interior_V_num
        - wadjacency_matrix
    """

    def __init__(self, total_graph):

        self.total_graph = total_graph
        self.box = np.array([1, 1])

        self.num_Vs = self.total_graph.num_Vs
        self.V_coords = self.total_graph.total_V_coords[:self.num_Vs]
        self.edges, self.bulk_edges, self.periodic_edges = self.construct_edges()
        self.interior_V_num = np.arange(self.num_Vs)

        self.E_lengths_by_v_num, self.wadjacency_matrices = self.construct_NEP_data()


    def construct_NEP_data(self):

        E_lengths_by_v_num = {}
        bulk_wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs))
        for e in self.bulk_edges:
            v, w = e["vw"]
            E_lengths_by_v_num[v, w] = e["l_vw"]
            bulk_wadjacency_matrix[v, w] = e["l_vw"]
            bulk_wadjacency_matrix[w, v] = e["l_vw"]

        bulk_wadjacency_matrix = bulk_wadjacency_matrix.tocsc()

        periodic_wadjacency_matrices = []
        for e in self.periodic_edges:
            periodic_wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs))
            v, w = e["vw"]
            E_lengths_by_v_num[v, w] = e["l_vw"]
            periodic_wadjacency_matrix[v, w] = e["l_vw"]
            periodic_wadjacency_matrix[w, v] = e["l_vw"]
            periodic_wadjacency_matrix = periodic_wadjacency_matrix.tocsc()
            periodic_wadjacency_matrices.append(periodic_wadjacency_matrix)

        wadjacency_matrices = [bulk_wadjacency_matrix] + periodic_wadjacency_matrices

        return E_lengths_by_v_num, wadjacency_matrices

    def construct_L(self, k, deriv=False):

        if not deriv:
            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
        else:
            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot

        L = scipy.sparse.csr_matrix((self.num_Vs, self.num_Vs))
        for wadjacency_matrix in self.wadjacency_matrices:
            matrix_csc = wadjacency_matrix.copy()
            matrix_csc.data = calculate_csc(k, matrix_csc.data)

            matrix_cot = wadjacency_matrix.copy()
            matrix_cot.data = calculate_cot(k, matrix_cot.data)
            matrix_cot = scipy.sparse.diags(matrix_cot.sum(axis=0).flat)

            L += (matrix_cot - matrix_csc).tocsr()[self.interior_V_num[:, None], self.interior_V_num]

        L = L.tocsc()

        return L

    def construct_edges(self):

        bulk_edges = []
        periodic_edges = []
        for v, w in self.total_graph.total_edges:
            v_isin_bulk = v < self.total_graph.num_Vs
            w_isin_bulk = w < self.total_graph.num_Vs

            if v_isin_bulk and w_isin_bulk:
                v_coords = self.total_graph.total_V_coords[v]
                w_coords = self.total_graph.total_V_coords[w]
                r_xy = np.array([0, 0])
                l_vw = np.linalg.norm(v_coords - w_coords)

                edge = {"vw": (v, w),
                        "l_vw": l_vw, 
                        "r_xy": r_xy,
                        "vw_coords": (v_coords, w_coords)}
                bulk_edges.append(edge)
                
            elif v_isin_bulk and not w_isin_bulk:
                v_coords = self.total_graph.total_V_coords[v]
                w_coords = self.total_graph.total_V_coords[w]

                # Calculate r_xy. v_patch_coords is assumed np.array([0, 0]).
                wrapped_w_coords = w_coords % self.box
                w_patch_coords = w_coords - wrapped_w_coords
                r_xy = w_patch_coords 
                distances_to_bulk_Vs = np.linalg.norm(self.V_coords - wrapped_w_coords, axis=1)
                w = np.argmin(distances_to_bulk_Vs)
                w_coords = self.V_coords[w] + r_xy
                l_vw = np.linalg.norm(v_coords - w_coords)

                edge = {"vw": (v, w),
                        "l_vw": l_vw, 
                        "r_xy": r_xy,
                        "vw_coords": (v_coords, w_coords)}
                periodic_edges.append(edge)

        periodic_edges = self.prune_edges(periodic_edges)

        edges = bulk_edges + periodic_edges

        return edges, bulk_edges, periodic_edges
    
    def prune_edges(self, edges_to_be_pruned):

        if self.total_graph.totally_periodic:
            edges = []
            for e in edges_to_be_pruned:
                v, w = e["vw"]
                if v < w:
                    edges.append(e.copy())
        else:
            edges = []
            for e in edges_to_be_pruned:
                if np.random.choice(2):
                    edges.append(e.copy())

        return edges
    
class Cell_Problem:

    def __init__(self, patch):

        self.patch = patch

    def construct_homogenized_tensor(self, xi):

        Q = np.zeros((2, 2))
        T = 0

        for e in self.patch.edges:
            v, w = e["vw"]
            l_vw = e["l_vw"]
            r_xy = e["r_xy"]

            a = r_xy + xi[w] - xi[v]

            Q += np.tensordot(a, a, axes=0) / l_vw
            T += l_vw

        c = np.trace(Q / T)

        return c

    def solve_corrector_equation(self):

        LHS, RHS = self.construct_equation()
        xi = sparseqr.solve(LHS, RHS)
        xi = xi.toarray()

        return xi

    def construct_equation(self):

        LHS = scipy.sparse.lil_matrix((self.patch.num_Vs + 1, self.patch.num_Vs))
        LHS[-1] = np.ones(self.patch.num_Vs)
        RHS = scipy.sparse.lil_matrix((self.patch.num_Vs + 1, 2))

        for e in self.patch.edges:
            v, w = e["vw"]
            l_vw = e["l_vw"] 
            r_xy = e["r_xy"]
            weight = 1 / l_vw

            LHS[v, v] += weight
            LHS[w, w] += weight

            LHS[v, w] -= weight
            LHS[w, v] -= weight

            RHS[v] += weight * r_xy
            RHS[w] -= weight * r_xy

        LHS = LHS.tocsc()
        RHS = RHS.tocsc()

        return LHS, RHS

class Truncated_Square:

    def __init__(self):

        self.total_num_Vs = 8
        self.total_V_coords, self.num_Vs = self.construct_total_V_coords()
        self.total_edges = self.construct_total_edges()
        self.totally_periodic = True

    def construct_total_V_coords(self):

        bulk_V_coords = np.array([[1 / 2 + np.sqrt(2) / 2, 1 / 2 + np.sqrt(2)],
                                  [1 / 2 + np.sqrt(2) / 2, 1 / 2],
                                  [1 / 2 + np.sqrt(2), 1 / 2 + np.sqrt(2) / 2],
                                  [1 / 2, 1 / 2 + np.sqrt(2) / 2]]) / (1 + np.sqrt(2))
        
        boundary_V_coords = np.array((bulk_V_coords[1] + np.array([0, 1]),
                                      bulk_V_coords[0] - np.array([0, 1]),
                                      bulk_V_coords[3] + np.array([1, 0]),
                                      bulk_V_coords[2] - np.array([1, 0])))
        
        num_Vs = 4
        total_V_coords = np.vstack((bulk_V_coords, boundary_V_coords))

        return total_V_coords, num_Vs

    def construct_total_edges(self):

        total_edges = np.array(([0, 2],
                                [0, 3],
                                [1, 2],
                                [1, 3],
                                [0, 4],
                                [1, 5],
                                [2, 6],
                                [3, 7]))
        
        boundary_boundary_edge_for_testing = np.array([6, 7])

        total_edges = np.vstack((total_edges, boundary_boundary_edge_for_testing))
        
        return total_edges
    
class Random_Delaunay:

    def __init__(self, num_Vs):

        self.num_Vs = num_Vs
        self.total_V_coords, self.total_num_Vs = self.construct_total_V_coords()
        self.total_edges = self.construct_total_edges()

        self.totally_periodic = False

    def construct_total_V_coords(self):

        num_Vs = 0
        bulk_V_coords = []
        boundary_V_coords = []
        while num_Vs < self.num_Vs:
            v_coords_x, v_coords_y = np.random.uniform(-1, 2, size=(2))
            v_coords = np.array([v_coords_x, v_coords_y])
            if (0 < v_coords_x < 1) and (0 < v_coords_y < 1):
                num_Vs += 1
                bulk_V_coords.append(v_coords)
            else:
                boundary_V_coords.append(v_coords)
        
        total_V_coords = np.vstack((bulk_V_coords, boundary_V_coords))
        total_num_Vs = total_V_coords.shape[0]

        return total_V_coords, total_num_Vs
    
    def construct_total_edges(self):

        triangulation = scipy.spatial.Delaunay(self.total_V_coords)
        V, W = triangulation.vertex_neighbor_vertices
        total_edges = []
        for v in range(self.total_num_Vs):
            w_inds = W[V[v]:V[v + 1]]
            for w in w_inds:
                if v < w:
                    total_edges.append([v, w])
        total_edges = np.array(total_edges)
        total_edges = np.sort(total_edges, axis=1)

        return total_edges