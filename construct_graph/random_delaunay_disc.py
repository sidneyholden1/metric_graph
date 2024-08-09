import numpy as np
import scipy
from construct_graph.graph import Graph, Flat

class Random_Delaunay_Disc(Graph, Flat):

    def __init__(self, prelim_num_Vs, auto_plot=True, fix_boundary=False, **kwargs):

        (self.num_Vs, 
         self.V_coords, 
         self.wadjacency_matrix, 
         self.interior_V_num) = self.construct_graph_data(prelim_num_Vs)
        
        if fix_boundary == True:
            self.fix_boundary()
        
        self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
        
        self.g_coords = self.construct_g_coords()
        
        if auto_plot:
            print(f"\n|V| = {self.num_Vs}")

            self.plot_graph(**kwargs)

    def fix_boundary(self):

        boundary_indices = np.setdiff1d(np.arange(self.num_Vs), self.interior_V_num)
        self.V_coords[boundary_indices] /= np.linalg.norm(self.V_coords[boundary_indices], axis=1, keepdims=True)
        row_indices, col_indices = self.wadjacency_matrix.nonzero()
        for row, col in zip(row_indices, col_indices):
            self.wadjacency_matrix[row, col] = np.linalg.norm(self.V_coords[row] - self.V_coords[col])
        
    def construct_dummy_data(self, temp_num_Vs):

        if temp_num_Vs < 1000:
            bounding_dummy_square_side = 10
        elif temp_num_Vs < 10000:
            bounding_dummy_square_side = 5
        elif temp_num_Vs >= 10000:
            bounding_dummy_square_side = 3

        vol_bounding_dummy_square = bounding_dummy_square_side**2
        vol_disc = np.pi
        sample_num_Vs = int(temp_num_Vs * vol_bounding_dummy_square / vol_disc)

        half_bounding_dummy_square_side = bounding_dummy_square_side / 2
        x = np.random.uniform(-half_bounding_dummy_square_side, 
                                half_bounding_dummy_square_side, size=(sample_num_Vs, 1))
        y = np.random.uniform(-half_bounding_dummy_square_side, 
                                half_bounding_dummy_square_side, size=(sample_num_Vs, 1))
        
        V_coords = np.hstack((x, y))

        return V_coords, sample_num_Vs
    
    def construct_graph_data(self, prelim_num_Vs):

        V_coords, sample_num_Vs = self.construct_dummy_data(prelim_num_Vs)

        mask_V_outside_circle = np.linalg.norm(V_coords, axis=1) > 1
        mask_V_inside_circle = ~mask_V_outside_circle
        
        delaunay_triangulation = scipy.spatial.Delaunay(V_coords)

        v, w = delaunay_triangulation.vertex_neighbor_vertices 

        wadjacency_matrix = scipy.sparse.lil_matrix((sample_num_Vs, sample_num_Vs), dtype=np.float64)

        mask_boundary_V_num = np.zeros(sample_num_Vs, dtype=bool)

        for v_num in range(sample_num_Vs):
            v_neighbours = w[v[v_num]:v[v_num + 1]]
            if not np.all(mask_V_outside_circle[v_neighbours]):
                if np.any(mask_V_outside_circle[v_neighbours]) & np.any(mask_V_inside_circle[v_neighbours]):
                    mask_boundary_V_num[v_num] = True
                for w_num in v_neighbours:
                    e_length = np.linalg.norm(V_coords[v_num] - V_coords[w_num])
                    wadjacency_matrix[v_num, w_num] = e_length

        mask_interior_V_num = mask_V_inside_circle & ~ mask_boundary_V_num
        mask_kept_V_num = mask_interior_V_num | mask_boundary_V_num

        V_coords = V_coords[mask_kept_V_num]
        num_Vs = V_coords.shape[0]
        # wadjacency_matrix = scipy.sparse.triu(wadjacency_matrix)
        wadjacency_matrix = wadjacency_matrix.tocsr()[mask_kept_V_num][:, mask_kept_V_num]

        interior_V_num = np.argwhere(mask_interior_V_num).flatten()
        boundary_V_num = np.argwhere(mask_boundary_V_num).flatten()

        all_V_num = np.sort(np.concatenate([interior_V_num, boundary_V_num]))
        interior_V_num = np.searchsorted(all_V_num, interior_V_num)

        return num_Vs, V_coords, wadjacency_matrix, interior_V_num
    
    def construct_E_lengths_by_v_num(self):

        v, w  = self.wadjacency_matrix.nonzero()
        E_lengths = self.wadjacency_matrix.data

        E_lengths_by_v_num = {}

        for v_num, w_num, e_length in zip(v, w, E_lengths):
            E_lengths_by_v_num[v_num, w_num] = e_length  

        return E_lengths_by_v_num     


# class Random_Delaunay_Disc(Graph, Flat):

#     """Quite a bit to do:
#         - get rid of delaunay triangulation calculation multiple times
#         - add functionality -- plotting mainly
#     """

#     def __init__(self, num_Vs, auto_plot=True, **kwargs):

#         self.num_Vs = num_Vs
#         self.V_coords = self.construct_V_coords()
#         self.E_lengths_by_v_num, self.interior_V_num = self.construct_delaunay_triangulation_data()
#         self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
#         self.g_coords = self.construct_g_coords()
        
#         if auto_plot:
#             print(f"\n|V| = {self.num_Vs}")

#             self.plot_graph(**kwargs)
    
#     def construct_V_coords(self):

#         r = np.sqrt(np.random.uniform(0, 1, size=(self.num_Vs, 1)))
#         theta = np.random.uniform(0, 2 * np.pi, size=(self.num_Vs, 1))

#         V_coords = np.hstack((r * np.cos(theta), r * np.sin(theta)))

#         return V_coords

#     def construct_delaunay_triangulation_data(self):

#         delaunay_triangulation = scipy.spatial.Delaunay(self.V_coords)
        
#         v, w = delaunay_triangulation.vertex_neighbor_vertices 

#         E_lengths_by_v_num = {}
        
#         for v_num in range(self.num_Vs):
#             for w_num in w[v[v_num]:v[v_num + 1]]:
#                 E_lengths_by_v_num[tuple(np.sort([v_num, w_num]))] = None

#         boundary_V_num = delaunay_triangulation.convex_hull
#         interior_V_num = np.setdiff1d(np.arange(self.num_Vs), boundary_V_num)

#         rescaling = np.sqrt(np.mean(np.linalg.norm(self.V_coords[boundary_V_num], axis=1)**2))

#         self.V_coords /= rescaling

#         return E_lengths_by_v_num, interior_V_num
    
#     def plot_edge_locs_vs_edge_lengths(self):
#         """
#         Checks whether there is a bias in the spatial distribution of edge lengths.
#         There should be because things are weird along the boundary.
#         Answer: there are a small number of outliers (large edges) near the boundary. This
#         number goes to 0 as the graph density increases.
#         """
#         delaunay_triangulation = scipy.spatial.Delaunay(self.V_coords)
#         boundary_V_num = delaunay_triangulation.convex_hull

#         midpoints = []
#         lengths = []

#         font = {'weight': 'bold',
#                 'size': 16}

#         for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():
#             if (v_num not in boundary_V_num) and (w_num not in boundary_V_num):
#                 v_coords = self.V_coords[v_num]
#                 w_coords = self.V_coords[w_num]
#                 midpoints.append(np.linalg.norm(np.mean([v_coords, w_coords], axis=0))**2)
#                 lengths.append(l_vw)

#         fig, ax = plt.subplots(figsize=(10, 8))
#         ax.scatter(midpoints, lengths, s=10)
#         ax.set_xlabel('Radial location', fontdict=font)
#         ax.set_ylabel('Edge length', fontdict=font)
#         ax.tick_params(axis='x', labelsize=12)
#         ax.tick_params(axis='y', labelsize=12)

#         return fig, ax

#     def return_pde_data(self, m, n):
        
#         return np.array([scipy.special.jn_zeros(i, n) for i in range(m)]) / np.sqrt(2)
    
#     def return_pde_eigenfunction(self, m, n, trig="cos"):

#         x = self.V_coords[:, 0]
#         y = self.V_coords[:, 1]

#         r = np.sqrt(x**2 + y**2)
#         theta = np.arctan2(y, x)

#         radial_function = scipy.special.jn(m, r * scipy.special.jn_zeros(m, n)[-1])

#         if trig == "cos":
#             angular_function = np.cos(m * theta) 
#         elif trig == "sin":
#             angular_function = np.sin(m * theta) 
        
#         function = self.normalize_function(radial_function * angular_function)

#         return function

#     def normalize_function(self, f):

#         return f / self.inner_product(f, f)
    
#     def inner_product(self, f, g):

#         x = self.V_coords[:, 0]
#         y = self.V_coords[:, 1]

#         r = np.sqrt(x**2 + y**2)
        
#         return np.sqrt(np.sum(f * g * r))
    
#     def project_function(self, f, m, n):

#         pde1 = self.return_pde_eigenfunction(m, n, trig="cos")
#         pde2 = self.return_pde_eigenfunction(m, n, trig="sin")

#         a = self.inner_product(f, pde1)
#         b = self.inner_product(f, pde2)

#         f = a * pde1 + b * pde2

#         return self.normalize_function(f)