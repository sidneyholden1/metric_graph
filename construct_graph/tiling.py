import numpy as np
import scipy
from construct_graph.graph import Graph, Flat

class Tiling(Graph, Flat):

    tile_types = ("truncated_square", "square", "rectangular", "hexagonal", 
                  "truncated_trihexagonal", "uniformly_random_Delaunay_triangulated",
                  "criss_cross", "custom")

    def __init__(self, tile_type, periodic=True, rescale=True, 
                 autoplot=True, figsize=10, embedding_space=None, **init_kwargs):
        
        if tile_type not in Tiling.tile_types:
            raise ValueError(f"tile_type not recognised. Must be one of: {Tiling.tile_types}.")
        
        self.tile_type = tile_type
        self.periodic = periodic
        self.rescale = rescale
        self.init_kwargs = init_kwargs
        self.num_horizontal_displacements = self.init_kwargs.get('num_horizontal_displacements', 1)
        self.num_vertical_displacements = self.init_kwargs.get('num_vertical_displacements', 1)
        self.directions = {'right': (1, 0), 'upper_right': (1, 1), 'up': (0, 1), 'upper_left': (-1, 1), 
                           'left': (-1, 0), 'lower_left': (-1, -1), 'down': (0, -1), 'lower_right': (1, -1)}
        
        (tile_V_coords, 
         tile_wadjacency_matrix, 
         connectivity_map) = eval(f"self.construct_{self.tile_type}_tile_data()")
            
        self.tile_V_coords = tile_V_coords
        self.num_Vs_for_tile = self.tile_V_coords.shape[0]
        self.tile_wadjacency_matrix = tile_wadjacency_matrix
        self.connectivity_map = connectivity_map
        self.construct_tiling_data(tile_V_coords, tile_wadjacency_matrix, connectivity_map)

        if self.periodic:
            self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
            self.interior_V_num = np.arange(self.num_Vs)
            self.g_coords = self.construct_g_coords(periodic_edges=self.periodic_edges)
        else:
            self.V_coords[:, 0] -= np.min(self.V_coords[:, 0])
            self.V_coords[:, 1] -= np.min(self.V_coords[:, 1])
            factor = np.max(self.V_coords)
            self.V_coords /= factor
            self.wadjacency_matrix /= factor
            min_x_mask = self.V_coords[:, 0] < 1e-14
            min_y_mask = self.V_coords[:, 1] < 1e-14
            max_x_mask = np.abs(self.V_coords[:, 0] - 1) < 1e-14
            max_y_mask = np.abs(self.V_coords[:, 1] - 1) < 1e-14
            total_mask = ~(min_x_mask | min_y_mask | max_x_mask | max_y_mask)
            self.interior_V_num = np.arange(self.num_Vs)[total_mask]
            self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
            self.g_coords = self.construct_g_coords()

        if autoplot:
            print(f"\n|V| = {self.num_Vs}")

            self.plot_graph(figsize=figsize, embedding_space=embedding_space)

    def construct_E_lengths_by_v_num(self):

        v_nums, w_nums = self.wadjacency_matrix.nonzero()

        E_lengths_by_v_num = {}
        for v_num, w_num in zip(v_nums, w_nums):
            E_lengths_by_v_num[tuple(np.sort((v_num, w_num)))] = self.wadjacency_matrix[v_num, w_num]

        return E_lengths_by_v_num

    def calculate_global_index(self, tile_x, tile_y, local_index, num_Vs_for_tile):
        # Requires V_coords construction by displacement tile in rows, and *then* columns
        return (tile_y * self.num_horizontal_displacements + tile_x) * num_Vs_for_tile + local_index
    
    def construct_tiling_data(self, tile_V_coords, tile_wadjacency_matrix, connectivity_map):

        # Create vertex coords for full tiling. The tile is translated along rows, and then columns. This is crucial
        # for how self.calculate_global_index(...) is defined
        x = np.tile(np.arange(self.num_horizontal_displacements) * self.horizontal_displacement, 
                    self.num_vertical_displacements)[:, None]
        y = np.repeat(np.arange(self.num_vertical_displacements) * self.vertical_displacement, 
                      self.num_horizontal_displacements)[:, None]
        all_displacements = np.hstack((x, y))
        all_displacements = np.repeat(all_displacements, self.num_Vs_for_tile, axis=0)
        repeated_V_coords = np.tile(tile_V_coords, (self.num_horizontal_displacements * self.num_vertical_displacements, 1))
        V_coords = repeated_V_coords + all_displacements

        # Create wadjacency matrix for full tiling
        # fill in intra-tile connections
        matrices = [tile_wadjacency_matrix.copy() for _ in range(self.num_horizontal_displacements * self.num_vertical_displacements)]
        wadjacency_matrix = scipy.sparse.block_diag(matrices, format='lil', dtype=np.float64) 

        # fill in inter-tile connections
        periodic_edges = {}
        # Iterate through displacements of tile
        for num_horizontal_displacement in range(self.num_horizontal_displacements):
            for num_vertical_displacement in range(self.num_vertical_displacements):
                # Iterate through connections between vertices in current tile and adjacent tiles 
                for (local_v_num, local_w_num), edges_params in connectivity_map.items():
                    for edge_params in edges_params:
                        # Retrieve relative adjacent tile displacements
                        adjacent_tile_dx, adjacent_tile_dy = self.directions[edge_params['direction']]
                        if not self.periodic:
                            adjacent_tile_x = num_horizontal_displacement + adjacent_tile_dx
                            adjacent_tile_y = num_vertical_displacement + adjacent_tile_dy
                            
                            if ((-1 < adjacent_tile_x < self.num_horizontal_displacements) 
                                and (-1 < adjacent_tile_y < self.num_vertical_displacements)):
                                global_v_num_in_current_tile = self.calculate_global_index(num_horizontal_displacement, 
                                                                                           num_vertical_displacement,
                                                                                           local_v_num,
                                                                                           self.num_Vs_for_tile)
                                global_w_num_in_adjacent_tile = self.calculate_global_index(adjacent_tile_x, 
                                                                                            adjacent_tile_y,
                                                                                            local_w_num,
                                                                                            self.num_Vs_for_tile)                   
                                # Set wadjacency wmatrix entries with global indices
                                e_length = edge_params["e_length"]
                                wadjacency_matrix[global_v_num_in_current_tile, global_w_num_in_adjacent_tile] = e_length
                                wadjacency_matrix[global_w_num_in_adjacent_tile, global_v_num_in_current_tile] = e_length

                        else:
                            # Find adjacent tile indices
                            adjacent_tile_x = (num_horizontal_displacement + adjacent_tile_dx) % self.num_horizontal_displacements
                            adjacent_tile_y = (num_vertical_displacement + adjacent_tile_dy) % self.num_vertical_displacements
                            # Retrieve local v_nums in a tile, and edge length
                            e_length = edge_params["e_length"]
                            # Calculate global v_nums
                            global_v_num_in_current_tile = self.calculate_global_index(num_horizontal_displacement, 
                                                                                    num_vertical_displacement,
                                                                                    local_v_num,
                                                                                    self.num_Vs_for_tile)
                            global_w_num_in_adjacent_tile = self.calculate_global_index(adjacent_tile_x, 
                                                                                        adjacent_tile_y,
                                                                                        local_w_num,
                                                                                        self.num_Vs_for_tile)
                            # Set wadjacency matrix entries with global indices
                            wadjacency_matrix[global_v_num_in_current_tile, global_w_num_in_adjacent_tile] = e_length
                            wadjacency_matrix[global_w_num_in_adjacent_tile, global_v_num_in_current_tile] = e_length
                            # Determine edge segments corresponding to periodic edges, without wrapping "around", and
                            # in 'upper and right' half of graph: ['right', 'upper_right', 'up', 'upper_left']
                            # Check first that we have a periodic edge
                            full_dx = num_horizontal_displacement + adjacent_tile_dx
                            full_dy = num_vertical_displacement + adjacent_tile_dy
                            if not ((-1 < full_dx < self.num_horizontal_displacements) 
                                    and (-1 < full_dy < self.num_vertical_displacements)):
                                # If current tile is correct, then translate w up to free end
                                if edge_params['direction'] in ['right', 'upper_right', 'up', 'upper_left']:
                                    v_wrap_num = global_v_num_in_current_tile
                                    w_wrap_num = self.calculate_global_index(num_horizontal_displacement, 
                                                                            num_vertical_displacement,
                                                                            local_w_num,
                                                                            self.num_Vs_for_tile)
                                    displacement_v = np.array([0, 0])
                                    displacement_w_x = adjacent_tile_dx * self.horizontal_displacement
                                    displacement_w_y = adjacent_tile_dy * self.vertical_displacement
                                    displacement_w = np.array([displacement_w_x, displacement_w_y])
                                # Else, adjacent tile is correct, and translate v up to free end
                                else:
                                    v_wrap_num = self.calculate_global_index(adjacent_tile_x, 
                                                                            adjacent_tile_y,
                                                                            local_v_num,
                                                                            self.num_Vs_for_tile)
                                    w_wrap_num = global_w_num_in_adjacent_tile
                                    displacement_v_x = -adjacent_tile_dx * self.horizontal_displacement
                                    displacement_v_y = -adjacent_tile_dy * self.vertical_displacement
                                    displacement_v = np.array([displacement_v_x, displacement_v_y])
                                    displacement_w = np.array([0, 0])

                                e_by_v_num = np.array([global_v_num_in_current_tile, global_w_num_in_adjacent_tile])
                                sorted_vs = np.argsort(e_by_v_num)
                                e_by_v_coords = np.array((V_coords[v_wrap_num] + displacement_v, 
                                                        V_coords[w_wrap_num] + displacement_w))[sorted_vs]
                                e_by_v_num = tuple(e_by_v_num[sorted_vs])
                                
                                periodic_edges[e_by_v_num] = e_by_v_coords

        if self.rescale:
            # Calculate lengths of graph in x and y directions then factor = max
            factor = np.max((self.num_horizontal_displacements * self.horizontal_displacement, 
                             self.num_vertical_displacements * self.vertical_displacement))
            V_coords /= factor
            wadjacency_matrix /= factor
            if self.periodic:
                for key in periodic_edges:
                    n = len(periodic_edges[key])
                    for i in range(n):
                        periodic_edges[key][i] /= factor
                
        # Set graph attributes:
        self.V_coords = V_coords
        self.num_Vs = self.V_coords.shape[0]
        self.wadjacency_matrix = wadjacency_matrix.tocsc()
        if self.periodic:
            self.periodic_edges = periodic_edges

    def construct_custom_tile_data(self):

        if "tile_V_coords" not in self.init_kwargs:
            raise ValueError("Must provide num_Vs_for_tile")
        if "tile_wadjacency_matrix" not in self.init_kwargs:
            raise ValueError("Must provide tile_wadjacency_matrix")
        if "conn_map" not in self.init_kwargs:
            raise ValueError("Must provide conn_map")
        
        return (self.init_kwargs["tile_V_coords"], 
                self.init_kwargs["tile_wadjacency_matrix"], 
                self.init_kwargs["conn_map"])
    
    def construct_square_tile_data(self):

        tile_V_coords = np.array([[1 / 2, 1 / 2], 
                                  [3 / 2, 1 / 2],
                                  [3 / 2, 3 / 2],
                                  [1 / 2, 3 / 2]])
        tile_wadjacency_matrix = scipy.sparse.csc_matrix([[0, 1, 0, 1], 
                                                          [1, 0, 1, 0], 
                                                          [0, 1, 0, 1], 
                                                          [1, 0, 1, 0]], dtype=np.float64)

        self.horizontal_displacement = 2
        self.vertical_displacement = 2
        connectivity_map = {(0, 1): ({'direction': 'left', 'e_length': 1},),
                            (0, 3): ({'direction': 'down', 'e_length': 1},),
                            (2, 3): ({'direction': 'right', 'e_length': 1},),
                            (1, 2): ({'direction': 'down', 'e_length': 1},)}
        
        return tile_V_coords, tile_wadjacency_matrix, connectivity_map
    
    def construct_criss_cross_tile_data(self):

        tile_V_coords = np.array([[1 / 2, 1 / 2]])
        tile_wadjacency_matrix = scipy.sparse.csc_matrix([[0]], dtype=np.float64)

        self.horizontal_displacement = 1
        self.vertical_displacement = 1
        connectivity_map = {(0, 0): ({'direction': 'up', 'e_length': 1}, 
                                     {'direction': 'right', 'e_length': 1},
                                     {'direction': 'upper_right', 'e_length': np.sqrt(2)},
                                     {'direction': 'lower_right', 'e_length': np.sqrt(2)})}
        
        return tile_V_coords, tile_wadjacency_matrix, connectivity_map

    def construct_rectangular_tile_data(self):

        tile_V_coords = np.array([[1 / 4, 1 / 2]])
        tile_wadjacency_matrix = scipy.sparse.csc_matrix([[0]], dtype=np.float64)

        self.horizontal_displacement = 1 / 2
        self.vertical_displacement = 1
        connectivity_map = {(0, 0): ({'direction': 'up', 'e_length': 1}, 
                                     {'direction': 'right', 'e_length': 1 / 2})}
        
        return tile_V_coords, tile_wadjacency_matrix, connectivity_map
    
    def construct_hexagonal_tile_data(self):

        offset = np.array(([1 / 2, np.sqrt(3) / 2]))
        tile_V_coords = np.array(([0, 0],
                                  [np.cos(np.pi / 3), np.sin(np.pi / 3)],
                                  [np.cos(np.pi / 3) + 1, np.sin(np.pi / 3)],
                                  [2, 0])) + offset
        tile_wadjacency_matrix = scipy.sparse.csc_matrix([[0, 1, 0, 0], 
                                                          [1, 0, 1, 0], 
                                                          [0, 1, 0, 1], 
                                                          [0, 0, 1, 0]], dtype=np.float64)

        self.horizontal_displacement = 3
        self.vertical_displacement = np.sqrt(3)
        connectivity_map = {(3, 0): ({'direction': 'right', 'e_length': 1},),
                            (0, 1): ({'direction': 'down', 'e_length': 1},),
                            (3, 2): ({'direction': 'down', 'e_length': 1},)}

        return tile_V_coords, tile_wadjacency_matrix, connectivity_map

    def construct_truncated_square_tile_data(self):

        tile_V_coords = np.array([[1 / 2, 1 / 2 + np.sqrt(2) / 2], 
                                  [1 / 2 + np.sqrt(2) / 2, 1 / 2 + np.sqrt(2)],
                                  [1 / 2 + np.sqrt(2), 1 / 2 + np.sqrt(2) / 2],
                                  [1 / 2 + np.sqrt(2) / 2, 1 / 2]])
        tile_wadjacency_matrix = scipy.sparse.csc_matrix([[0, 1, 0, 1], 
                                                          [1, 0, 1, 0], 
                                                          [0, 1, 0, 1], 
                                                          [1, 0, 1, 0]], dtype=np.float64)

        self.horizontal_displacement = 1 + np.sqrt(2)
        self.vertical_displacement = 1 + np.sqrt(2)
        connectivity_map = {(0, 2): ({'direction': 'left', 'e_length': 1},),
                            (1, 3): ({'direction': 'up', 'e_length': 1},)}
        
        return tile_V_coords, tile_wadjacency_matrix, connectivity_map
    
    def construct_truncated_trihexagonal_tile_data(self):

        dodecagon_radius = 1 + np.sqrt(3) / 2
        hexagon_radius = np.sqrt(3) / 2
        square_radius = 1 / 2
        displacement = np.array([dodecagon_radius + 2 * hexagon_radius + square_radius,
                                 square_radius + dodecagon_radius])
        N = 12
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False) + 2 * np.pi / N / 2
        r = (np.sqrt(6) + np.sqrt(2)) / 2
        tile_V_coords = r * np.array([np.cos(thetas), np.sin(thetas)]).T
        tile_V_coords += np.ones(2) * (dodecagon_radius + square_radius)
        tile_V_coords = np.vstack((tile_V_coords, tile_V_coords + displacement))

        self.horizontal_displacement = 2 * (2 * hexagon_radius + square_radius + dodecagon_radius)
        self.vertical_displacement = 2 * (square_radius + dodecagon_radius)
        
        tile_wadjacency_matrix = scipy.sparse.lil_matrix((24, 24), dtype=np.float64)
        for i in range(12):
            tile_wadjacency_matrix[i % 12, (i + 1) % 12] = 1
            tile_wadjacency_matrix[(i + 1) % 12, i % 12] = 1
        for i in range(12):
            tile_wadjacency_matrix[(i % 12) + 12, ((i + 1) % 12) + 12] = 1
            tile_wadjacency_matrix[((i + 1) % 12) + 12, (i % 12) + 12] = 1
        tile_wadjacency_matrix[0, 19] = 1
        tile_wadjacency_matrix[19, 0] = 1
        tile_wadjacency_matrix[1, 18] = 1
        tile_wadjacency_matrix[18, 1] = 1

        connectivity_map = {(22, 5): ({'direction': 'right', 'e_length': 1},),
                            (23, 4): ({'direction': 'right', 'e_length': 1},),
                            (12, 7): ({'direction': 'upper_right', 'e_length': 1},),
                            (13, 6): ({'direction': 'upper_right', 'e_length': 1},),
                            (2, 9): ({'direction': 'up', 'e_length': 1},),
                            (3, 8): ({'direction': 'up', 'e_length': 1},),
                            (16, 11): ({'direction': 'up', 'e_length': 1},),
                            (17, 10): ({'direction': 'up', 'e_length': 1},),
                            (14, 21): ({'direction': 'up', 'e_length': 1},),
                            (15, 20): ({'direction': 'up', 'e_length': 1},)}
        
        return tile_V_coords, tile_wadjacency_matrix, connectivity_map

    def construct_uniformly_random_Delaunay_triangulated_tile_data(self):

        if "tile_V_coords" in self.init_kwargs:
            tile_V_coords = self.init_kwargs["tile_V_coords"]
            num_Vs_for_tile = np.shape(tile_V_coords)[0]
        elif "num_Vs_for_tile" not in self.init_kwargs:
            raise ValueError("Must provide num_Vs_for_tile")
        else:
            num_Vs_for_tile = self.init_kwargs["num_Vs_for_tile"]
            tile_V_coords = np.random.uniform(0, 1, size=(num_Vs_for_tile, 2))
            
        self.horizontal_displacement = 1
        self.vertical_displacement = 1

        # Construct Delaunay triangulation with translated V_coords in every direction
        extended_tile_V_coords = np.vstack(([tile_V_coords] + [tile_V_coords + np.array(i) for i in self.directions.values()])) 
        delaunay_triangulation = scipy.spatial.Delaunay(extended_tile_V_coords)
        tile_wadjacency_matrix = scipy.sparse.lil_matrix((num_Vs_for_tile, num_Vs_for_tile), dtype=np.float64)    
        v, w = delaunay_triangulation.vertex_neighbor_vertices 
        num_extended_Vs = extended_tile_V_coords.shape[0]

        # Construct wadjacency matrix for internal connections, and connectivity map for periodic connections
        connectivity_map = {}
        for v_num in range(num_extended_Vs):
            v_neighbours = w[v[v_num]:v[v_num + 1]]
            for w_num in v_neighbours:
                if v_num < w_num:
                    if not (v_num >= num_Vs_for_tile and w_num >= num_Vs_for_tile):
                        v_direction_index, tile_v_num = divmod(v_num, num_Vs_for_tile)
                        w_direction_index, tile_w_num = divmod(w_num, num_Vs_for_tile)
                        e_length = np.linalg.norm(extended_tile_V_coords[v_num] - extended_tile_V_coords[w_num])
                        if v_direction_index > 0:
                            v_direction =  list(self.directions.keys())[v_direction_index - 1]
                            if v_direction in ['right', 'upper_right', 'up', 'upper_left']:
                                connectivity_map[(tile_w_num, tile_v_num)] = {'direction': v_direction, 
                                                                              'e_length': e_length}
                        elif w_direction_index > 0:
                            w_direction =  list(self.directions.keys())[w_direction_index - 1]
                            if w_direction in ['right', 'upper_right', 'up', 'upper_left']:
                                connectivity_map[(tile_v_num, tile_w_num)] = {'direction': w_direction, 
                                                                              'e_length': e_length}
                        else:
                            tile_wadjacency_matrix[tile_v_num, tile_w_num] = e_length 
                            tile_wadjacency_matrix[tile_w_num, tile_v_num] = e_length 

        conn_map = {}
        for v_num, w_num in connectivity_map:
            if (w_num, v_num) not in conn_map:
                conn_map[(v_num, w_num)] = (connectivity_map[(v_num, w_num)],)

        self.extended_tile_V_coords = extended_tile_V_coords
        return tile_V_coords, tile_wadjacency_matrix, conn_map