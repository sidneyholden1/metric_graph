import numpy as np
import scipy
import osmnx as ox
import networkx as nx
from construct_graph.graph import Graph, Flat

class Transport_Network(Graph, Flat):

    """Accepts any place_names which the Geopy geocoding service can interpret and resolve.
    Examples include:
        - 'San Francisco, California, USA'
        - 'Paris, France'
        - 'Santa Clara County, California'
        - 'California, USA'
        - 'USA'
        - 'Manhattan, New York City, New York, USA'
    """

    def __init__(self, place_name, build_type="load", auto_plot=True, **kwargs):

        if build_type == "load":
            graph = ox.io.load_graphml(filepath='nyc_road_data.graphml')
        elif build_type == "graph_from_place":
            graph = ox.graph_from_place(place_name, network_type='all')
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(graph))
        graph.remove_nodes_from(isolated_nodes)
        
        # Build attributes
        self.V_coords = self.calculate_V_coords(graph)
        self.num_Vs = len(self.V_coords)
        self.E_lengths_by_v_num = self.calculate_E_lengths_by_v_num(graph)
        self.interior_V_num = self.calculate_interior_V_num(graph)
        self.wadjacency_matrix = self.calculate_wadjacency_matrix(graph)
        self.g_coords = self.calculate_g_coords(graph)

        if auto_plot:
            print(f"\n|V| = {self.num_Vs}")

            self.plot_graph(**kwargs)

    def calculate_V_coords(self, graph):
        return np.array([[data['x'], data['y']] for node, data in graph.nodes(data=True)])

    def calculate_E_lengths_by_v_num(self, graph):
        lengths = {}
        node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        for u, v, data in graph.edges(data=True):
            u_idx, v_idx = node_to_index[u], node_to_index[v]
            # Ensure u_idx < v_idx
            u_idx, v_idx = min(u_idx, v_idx), max(u_idx, v_idx)
            lengths[(u_idx, v_idx)] = data['length']
        return lengths
    
    def calculate_interior_V_num(self, graph):
        degree_array = np.array([d for n, d in graph.degree()])
        return np.where(degree_array > 1)[0]
    
    def calculate_wadjacency_matrix(self, graph):
        node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        row, col, data = [], [], []
        for (u, v), length in self.E_lengths_by_v_num.items():
            row.extend([u, v])
            col.extend([v, u])
            data.extend([length, length])
        return scipy.sparse.csr_matrix((data, (row, col)), shape=(self.num_Vs, self.num_Vs))
        
    def calculate_g_coords(self, graph):
        g_coords = []
        node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        
        for (u, v), _ in self.E_lengths_by_v_num.items():
            u_orig, v_orig = list(node_to_index.keys())[u], list(node_to_index.keys())[v]
            
            # Check for the existence of the edge in the original graph.
            if graph.has_edge(u_orig, v_orig):
                if 'geometry' in graph[u_orig][v_orig][0]:
                    coords = np.array(graph[u_orig][v_orig][0]['geometry'].xy).T
                else:
                    coords = self.V_coords[[u, v]]
            elif graph.has_edge(v_orig, u_orig):  # check the reverse edge
                if 'geometry' in graph[v_orig][u_orig][0]:
                    coords = np.array(graph[v_orig][u_orig][0]['geometry'].xy).T
                else:
                    coords = self.V_coords[[v, u]]
            else:
                raise ValueError(f"Edge ({u_orig}, {v_orig}) does not exist in the graph.")
                
            g_coords.append(coords.T)
        return g_coords