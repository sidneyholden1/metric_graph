import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from construct_graph.graph import Graph, Spherical


class Icosphere(Graph, Spherical):
    
    """This class produces attributes of Icosahedron (seed) after successive Conway 
        polyhedron operations: truncate (t) and dual (d): tdtdtdt...

        mesh_type   : type of graph we end up with--dual (triangular mesh) or truncate 
        (hexagonal mesh with 12 pentagons (one for each of the 12 vertices of the Icosahedron))

        num_subdivs : the number of subdivisions. E.g. ("truncate", 2) will perform tdt 
        to the icosahedron; ("dual", 3) will perform tdtdtd.

        The main attributes we end up with are: 
            - the edge length-weighted adjacency matrix
            - a plotting function
    """
    
    def __init__(self, mesh_type_and_num_subdivs = ("dual", 0), auto_plot=True, **kwargs):
        
        self.update_attributes_by_seed()
        
        self.mesh_type, self.num_subdivs = mesh_type_and_num_subdivs
        
        if self.mesh_type == "truncate" and self.num_subdivs == 0:  
            raise ValueError("Seed graph cannot be of type truncate")
        
        self.initial_build = True
        
        if self.num_subdivs > 0:
            for _ in range(self.num_subdivs - 1):
                self.update_attributes_by_truncate()
                self.update_attributes_by_dual()

            self.update_attributes_by_truncate()

            if self.mesh_type == "dual": 
                self.update_attributes_by_dual()
                
        self.spher_V_coords = self.convert_V_coords_to_spher()
        self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
        self.interior_V_num = np.arange(self.num_Vs)
        self.g_coords = self.construct_g_coords()
                
        self.initial_build = False
        
        if auto_plot:
            print(f"\n|V| = {self.num_Vs}")

            try: figsize = kwargs["figsize"]
            except: figsize = 5
            try: view = kwargs["view"]
            except: view = [-58.4,0]

            self.plot_graph(figsize=figsize, view=view, return_figax=False)
            
    def calculate_cell_area(self):
        
        if self.mesh_type == "truncate":
            V_by_f_num = self.construct_V_by_f_num()
            
        F_area = self.calculate_F_area()
        num_Vs_of_face = [len(f) for f in self.F_by_v_num]
        rescaled_F_area = np.array(F_area) / np.array(num_Vs_of_face)

        cell_area = []

        for v_by_f_num in V_by_f_num:
            cell_area.append(sum([rescaled_F_area[f_num] for f_num in v_by_f_num]))

        return cell_area
    
    def calculate_F_area(self):
        
        F_area = []
        F_by_v_coord = self.construct_F_by_v_coord()
        
        for f_by_v_coord in F_by_v_coord:
            f_vertex_pairs = list(zip(f_by_v_coord, np.roll(f_by_v_coord, -1, axis=0)))
            
            F_area.append(1/2 * np.linalg.norm(sum([np.cross(v_coord, w_coord) 
                                                    for v_coord, w_coord in f_vertex_pairs])))

        return F_area
    
    def construct_curve(self, v, w):

        t = np.linspace(0, 1, 100).reshape(1, 100)
        l = v.reshape(3, 1) @ t + w.reshape(3, 1) @ (1 - t)
        norms = np.linalg.norm(l, axis=0, keepdims=True)

        l /= norms

        return l
        
    def construct_E_lengths_by_v_num(self):

        E_lengths_by_v_num = {}

        for f_by_v_num in self.F_by_v_num:
            edge = list(zip(f_by_v_num, np.roll(f_by_v_num, -1)))
            E_lengths_by_v_num |= {(min(e), max(e)): None for e in edge}

        return E_lengths_by_v_num 

    def construct_F_by_v_coord(self, purpose=""):
        
        F_by_v_coord = [self.V_coords[f_by_v_num] for f_by_v_num in self.F_by_v_num]
        
        if purpose=="plot" and self.mesh_type=="truncate":
            F_by_v_coord.sort(key=len)

            return F_by_v_coord[::-1] # put pentagons at start to colour them red

        return F_by_v_coord
    
    def construct_next_graph(self):
        
        next_graph = copy.deepcopy(self)
        
        if next_graph.mesh_type == "dual":
            next_graph.update_attributes_by_truncate()
        else:
            next_graph.update_attributes_by_dual()
        
        return next_graph
    
    def construct_V_by_v_neighbours_num(self):

        V_by_v_neighbours_num = [[] for _ in range(self.num_Vs)]

        for v_a, v_b in self.E_lengths_by_v_num:
            V_by_v_neighbours_num[v_a].append(v_b)
            V_by_v_neighbours_num[v_b].append(v_a)

        return V_by_v_neighbours_num
    
    def order_F_by_v_num_and_construct_V_by_f_num(self, new_F_by_new_v_num):
        
        V_by_f_num = [[] for _ in range(self.num_Vs)]
        
        for f_num, f_by_v_num in enumerate(new_F_by_new_v_num):
            for v in f_by_v_num:
                V_by_f_num[v].append(f_num)
            # Calculates ordering of vertices of polygon using
            # angles of each w away from v using centroid:
            
            centroid = np.mean(self.V_coords[f_by_v_num], axis=0)
            disps = self.V_coords[f_by_v_num] - centroid

            costheta = np.dot(disps[1:], disps[0])
            sintheta = np.cross(disps[1:], disps[0])
            back = np.where(np.dot(sintheta, centroid) < 0)[0]
            sintheta = np.linalg.norm(np.cross(disps[1:], disps[0]), axis=1)
            sintheta[back] *= -1
            atan2 = np.arctan2(sintheta, costheta)
            back = np.where(atan2 < 0)[0]
            atan2[back] = 2*np.pi + atan2[back]
            sort = np.argsort(atan2)
            new_F_by_new_v_num[f_num] = ([new_F_by_new_v_num[f_num][0]] 
                                         + list(np.array(new_F_by_new_v_num[f_num][1:])[(sort)]))
                
        return new_F_by_new_v_num, V_by_f_num
    
    def construct_V_by_f_num(self):

        V_by_f_num = [[] for _ in range(self.num_Vs)]

        for f_num, f_by_v_num in enumerate(self.F_by_v_num):
            for v in f_by_v_num:
                V_by_f_num[v].append(f_num)

        return(V_by_f_num)
    
    def return_flipped_mesh_type(self):
        
        if self.mesh_type == "dual": 
            return "truncate"
        
        return "dual"
    
    def plot_graph(self, figsize=7, markersize=10, lim=0.7, view=[-58.4,0], alpha=0.94, 
                   return_figax=False, eigenmode=None):

        F_by_v_coord = self.construct_F_by_v_coord(purpose="plot")

        fig = plt.figure(figsize=(figsize, figsize))
        ax = fig.add_subplot(111, projection='3d')

        if eigenmode is not None:
            x, y, z = np.hstack((self.g_coords))
            ax.scatter3D(x, y, z, s=markersize, c=eigenmode, cmap="plasma")

        elif self.mesh_type=="truncate":
            facecolors = [['white'], ['black']]
            num_F_by_v_coord = len(F_by_v_coord)
            ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor='black', 
                                                facecolors=((num_F_by_v_coord - 12) * facecolors[0] + 12 * facecolors[1]), 
                                                linewidths=3, alpha=alpha))
                
        else:
            ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor='black', 
                                                facecolors='white', 
                                                linewidths=3, alpha=alpha))

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        ax.view_init(*view)
        fig.tight_layout()

        if return_figax:
            return fig, ax
        else:
            plt.show()
    
    # def plot_graph(self, scatter=False, figsize=7, markersize=40, 
    #                lim=0.7, view=[-58.4,0], alpha=0.94, return_figax=False, 
    #                show_pentagons=True, soccer_ball_colors=True, eigenmode=False, mode_color=None):
        
    #     F_by_v_coord = self.construct_F_by_v_coord(purpose="plot")
        
    #     fig = plt.figure(figsize=(figsize, figsize))

    #     ax = fig.add_subplot(111, projection='3d')
        
    #     if self.mesh_type=="truncate":
    #         if show_pentagons:
    #             if soccer_ball_colors:
    #                 edgecolor = "black"
    #                 facecolors = [['white'], ['black']]
    #                 alpha = 1

    #             else:
    #                 edgecolor = 'blue'
    #                 facecolors = [['xkcd:light periwinkle'], ['red']]

    #             num_F_by_v_coord = len(F_by_v_coord)
    #             ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor=edgecolor, 
    #                                                  facecolors=((num_F_by_v_coord-12)*facecolors[0] + 12*facecolors[1]), 
    #                                                  linewidths=3, alpha=alpha))
                
    #         elif eigenmode:
    #             domain = np.hstack((self.construct_g_coords(points=10)))
    #             ax.scatter3D(domain[0], domain[1], domain[2], s=10, c=mode_color)

    #         else:  
    #             fac=0.9842
    #             u = np.linspace(0, 2 * np.pi, 100)
    #             v = np.linspace(0, np.pi, 100)
    #             x = fac * 1 * np.outer(np.cos(u), np.sin(v))
    #             y = fac * 1 * np.outer(np.sin(u), np.sin(v))
    #             z = fac * 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    #             ax.plot_surface(x, y, z, 
    #                             color='xkcd:light periwinkle', alpha=0.4,
    #                             shade=False)
                
    #             domain = np.hstack((self.construct_g_coords(points=100)))
    #             ax.scatter3D(domain[0], domain[1], domain[2], s=10, c='b')
            
    #     else:
            
    #         ax.add_collection3d(Poly3DCollection(F_by_v_coord, edgecolor='black', 
    #                                              facecolors='white', 
    #                                              linewidths=3, alpha=alpha))

    #     if scatter: ax.scatter3D(self.V_coords[:,0], self.V_coords[:,1], self.V_coords[:,2], 
    #                              s=markersize)

    #     ax.set_xlim([-lim, lim])
    #     ax.set_ylim([-lim, lim])
    #     ax.set_zlim([-lim, lim])
    #     ax.set_box_aspect([1, 1, 1])
    #     ax.axis('off')
    #     ax.view_init(*view)
    #     fig.tight_layout()

    #     if return_figax:
    #         return fig, ax
    #     else:
    #         plt.show()        
    
    def truncate_V_by_v_neighbours_num(self):

        new_V_coord = []
        new_F_by_new_v_num = []
        old_E_by_new_v_num = {f"{v_a},{v_b}": [] for v_a, v_b in self.E_lengths_by_v_num}

        for v_num, v_neighbours_num in enumerate(self.V_by_v_neighbours_num):

            v_coord = self.V_coords[v_num]

            new_Vs = v_coord + 1/3 * (self.V_coords[v_neighbours_num] - v_coord)
            new_Vs /= np.linalg.norm(new_Vs, axis=1)[:, np.newaxis]
            new_V_coord += list(new_Vs)

            current_new_num_Vs = len(new_V_coord)

            new_F_by_new_v_num.append(np.arange(current_new_num_Vs - len(v_neighbours_num), current_new_num_Vs))

            edges = [[min([v_num, v_neighbour_num]), max([v_num, v_neighbour_num])] 
                        for v_neighbour_num in v_neighbours_num]

            self.update_old_E_by_new_v_num(edges, new_F_by_new_v_num[-1], old_E_by_new_v_num)

        return new_V_coord, new_F_by_new_v_num, old_E_by_new_v_num
        
    def update_attributes_by_dual(self):
        
        new_V_coord = np.array([np.mean(self.V_coords[face], axis=0) for face in self.F_by_v_num])
        self.V_coords = new_V_coord / np.linalg.norm(new_V_coord, axis=1)[:, np.newaxis]
        self.num_Vs = len(self.V_coords)

        self.F_by_v_num = self.V_by_f_num.copy()
        del self.V_by_f_num

        self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
        self.V_by_v_neighbours_num = self.construct_V_by_v_neighbours_num()
        
        if not self.initial_build:      
            self.update_shared_attributes()
            
    def update_attributes_by_seed(self):

        """
        Update graph to seed--initial graph--the icosahedron (the largest 
        Platonic solid and smallest geodesic polyhedron)
        """

        phi = (1 + np.sqrt(5)) / 2
        self.V_coords = np.array(([-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0],
                                  [1, -phi, 0], [0, -1, phi], [0,  1, phi],
                                  [0, -1, -phi], [0,  1, -phi], [phi, 0, -1],
                                  [phi, 0, 1], [-phi, 0, -1], [-phi, 0,  1]))
        self.V_coords /= np.linalg.norm(self.V_coords, axis=1)[:, np.newaxis]
        self.spher_V_coords = self.convert_V_coords_to_spher()
        self.num_Vs = len(self.V_coords)
        self.F_by_v_num = np.array(([0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                                    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                                    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                                    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]))
        self.F_by_v_num = np.sort(self.F_by_v_num, axis=1)
        self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
        self.V_by_v_neighbours_num = self.construct_V_by_v_neighbours_num()
        self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
        
    def update_attributes_by_truncate(self):

        new_V_coord, new_F_by_new_v_num, old_E_by_new_v_num = self.truncate_V_by_v_neighbours_num()
        del self.V_by_v_neighbours_num

        self.V_coords = np.array(new_V_coord)
        self.num_Vs = len(self.V_coords)

        new_F_by_new_v_num += [np.array(old_E_by_new_v_num[f"{v_a},{v_b}"]
                                        + old_E_by_new_v_num[f"{v_a},{v_c}"]
                                        + old_E_by_new_v_num[f"{v_b},{v_c}"]) 
                               for v_a, v_b, v_c in self.F_by_v_num]

        self.F_by_v_num, self.V_by_f_num = self.order_F_by_v_num_and_construct_V_by_f_num(new_F_by_new_v_num)
        self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
        
        if not self.initial_build:
            
            self.update_shared_attributes()
            
    def update_shared_attributes(self):
        
        self.spher_V_coords = self.convert_V_coords_to_spher()
        self.mesh_type = self.return_flipped_mesh_type()
        self.num_subdivs += 1
        self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
        self.interior_V_num = np.arange(self.num_Vs)
        self.g_coords = self.construct_g_coords()

    def update_old_E_by_new_v_num(self, edges, new_f_by_new_v_num, old_E_by_new_v_num):

        for (v_a, v_b), new_v in zip(edges, new_f_by_new_v_num):
            old_E_by_new_v_num[f"{v_a},{v_b}"].append(new_v)