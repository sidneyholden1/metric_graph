import numpy as np
import scipy
from construct_graph.graph import Graph, Flat
import geopandas as gpd
import shapely

class Random_Delaunay_Country(Graph, Flat):

    """Broken for now: but just need to switch to constrained Delaunay triangulation 
    to restrict triangles/edges to landmass interiors: https://rufat.be/triangle/API.html.
    """

    def __init__(self, country_name, num_Vs, auto_plot=True, seed=0, **kwargs):

        self.country_name = country_name
        self.num_Vs = num_Vs
        self.seed = seed

        np.random.seed(self.seed)

        self.V_coords, self.interior_V_num = self.construct_V_coords_and_interior_V_num()
        self.E_lengths_by_v_num = self.construct_delaunay_triangulation_data()

        self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
        self.g_coords = self.construct_g_coords()

        if auto_plot:
            print(f"\n|V| = {self.num_Vs}")

            self.plot_graph(**kwargs)

    def construct_delaunay_triangulation_data(self):

        delaunay_triangulation = scipy.spatial.Delaunay(self.V_coords)
        
        v, w = delaunay_triangulation.vertex_neighbor_vertices 

        E_lengths_by_v_num = {}
        
        for v_num in range(self.num_Vs):
            for w_num in w[v[v_num]:v[v_num + 1]]:
                if not (v_num > self.num_interior_Vs 
                        and w_num > self.num_interior_Vs 
                        and np.abs(v_num - w_num) > 1):
                    E_lengths_by_v_num[tuple(np.sort([v_num, w_num]))] = None

        return E_lengths_by_v_num

    def construct_V_coords_and_interior_V_num(self):

        world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        country_geometry = world_data[world_data['name'] == self.country_name].geometry.iloc[0]

        if country_geometry.geom_type == 'Polygon':
            geometries = [country_geometry]
        else:
            geometries = country_geometry.geoms

        interior_V_coords = []
        boundary_V_coords = []
        total_area = sum(p.area for p in geometries)

        for polygon in geometries:
            x, y = polygon.exterior.coords.xy
            coords = np.dstack((x, y)).squeeze()
            boundary_V_coords.extend(coords)

        boundary_V_coords = np.array(boundary_V_coords)
        self.num_interior_Vs = self.num_Vs - boundary_V_coords.shape[0]

        points_for_polygons = []
        for polygon in geometries:
            proportion = polygon.area / total_area
            points_for_polygons.append(int(self.num_interior_Vs * proportion))

        dif_samp = self.num_interior_Vs - sum(points_for_polygons)
        if dif_samp > 0: points_for_polygons[0] += dif_samp

        # For each polygon, calculate its area proportion and allocate points accordingly
        for polygon, points_for_polygon in zip(geometries, points_for_polygons):
            interior_V_coords.extend(self.sample_inside_polygon(polygon, points_for_polygon))

        interior_V_coords = np.array(interior_V_coords)

        V_coords = np.vstack((interior_V_coords, boundary_V_coords))
        interior_V_num = np.arange(interior_V_coords.shape[0])

        V_coords -= np.min(V_coords, axis=0)
        V_coords /= np.max(V_coords, axis=0)
        
        return V_coords, interior_V_num
        
    def sample_inside_polygon(self, polygon, n_points):

        minx, miny, maxx, maxy = polygon.bounds
        inside_points = []

        while len(inside_points) < n_points:
            sample_x = np.random.uniform(minx, maxx)
            sample_y = np.random.uniform(miny, maxy)
            point = shapely.geometry.Point(sample_x, sample_y)
            if polygon.contains(point):
                inside_points.append((sample_x, sample_y))

        return np.array(inside_points)