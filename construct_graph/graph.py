import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import matplotlib.patheffects as path_effects


class Graph:

    """Parent class of more specific graph classes (spiderweb, delaunay triangulations etc.)
    These child classes need to produce the attributes:
        - num_Vs
        - V_coords
        - E_lengths_by_v_num
        - interior_V_num
    """

    def __init__(self):
        """
        Required attributes:
            - num_Vs
            - V_coords
            - E_lengths_by_v_num
            - interior_V_num
            - wadjacency_matrix
        """

        if not hasattr(self, "V_coords"):
            raise AttributeError(f"{self.__class__.__name__} object doesn't have V_coords")
        
        if not hasattr(self, "num_Vs"):
            self.num_Vs = self.V_coords.shape[0]
        
        if not hasattr(self, "wadjacency_matrix"):
            raise AttributeError(f"{self.__class__.__name__} object doesn't have wadjacency_matrix")

        if not hasattr(self, "interior_V_num"):
            raise AttributeError(f"{self.__class__.__name__} object doesn't have interior_V_num")
        
        if not hasattr(self, "g_coords"):
            self.g_coords = self.construct_g_coords()

    def calculate_csc(self, k, l):

        return 1 / np.sin(k * l)
    
    def calculate_sec(self, k, l):

        return 1 / np.cos(k * l)
    
    def calculate_cot(self, k, l):

        return 1 / np.tan(k * l)

    def calculate_dcsc(self, k, l):

        return -l * self.calculate_csc(k, l) * self.calculate_cot(k, l)

    def calculate_dcot(self, k, l):

        return -l * self.calculate_csc(k, l)**2

    def calculate_dsec(self, k, l):

        return l * self.calculate_sec(k, l) * np.tan(k * l)
    
    def construct_wadjacency_matrix(self):

        wadjacency_matrix = scipy.sparse.lil_matrix((self.num_Vs, self.num_Vs), dtype=np.float64)

        for v0_num, v1_num in self.E_lengths_by_v_num:

            wadjacency_matrix[v0_num, v1_num] = self.E_lengths_by_v_num[v0_num, v1_num]
            wadjacency_matrix[v1_num, v0_num] = self.E_lengths_by_v_num[v0_num, v1_num]

        return wadjacency_matrix.tocsc()
    
    def construct_L(self, k, deriv=False):

        if not deriv:
            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
        else:
            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot

        matrix_csc = self.wadjacency_matrix.copy()
        matrix_csc.data = calculate_csc(k, matrix_csc.data)

        matrix_cot = self.wadjacency_matrix.copy()
        matrix_cot.data = calculate_cot(k, matrix_cot.data)
        matrix_cot = scipy.sparse.diags(matrix_cot.sum(axis=0).flat)

        return (matrix_cot - matrix_csc).tocsc()[self.interior_V_num[:, None], self.interior_V_num]
    
    def normalize_array(self, x, return_norms=False, ndim=3):

        """x.shape == (number of points, ndim)
        x rows are vertex coords.
        Normalizes x so that np.linalg.norm(x, axis=1) = flat array of ones
        """

        shape = x.shape

        if shape[-1] == ndim:
            norms = np.linalg.norm(x, axis=len(shape) - 1, keepdims=True)

            if return_norms:
                return x / norms, norms

            return x / norms

        raise ValueError("Make sure x.shape == (number of points, ndim)")
    
    def convert_to_column_vector(self, v):

        try:
            shape = v.shape
            dims = len(shape)

            if (dims == 2 and np.any(shape == 1)) or dims == 1:
                return v.reshape(max(shape), 1)

        except:
            v = np.array([v])

            shape = v.shape
            dims = len(shape)

            if dims == 1 or min(shape) == 1:
                return v.reshape(max(shape), 1)

            raise ValueError("Cannot be converted to column vector: v has bad type or shape")

    def generate_graph_linewidth(self):
        # Just a heuristic
        return np.max([0.5, -0.086 * np.sqrt(self.num_Vs) + 7.66])

    def generate_plot_kwargs(self, **kwargs):

        stacked_g_coords = np.hstack(self.g_coords)
        min_coord, max_coord = np.min(stacked_g_coords), np.max(stacked_g_coords)
        min_coord -= 0.01 * np.abs(max_coord)
        max_coord += 0.01 * np.abs(max_coord)

        defaults = {
            "fig": None,
            "ax": None,
            "figsize": 10,
            "linewidth": self.generate_graph_linewidth(),
            "return_figax": False,
            "eigenmode": None,
            "dim": 2,
            "view": (15, 15),
            "colors": "b",
            "capstyle": "round",
            "background_color": None,
            "cmap": "plasma",
            "embedding_space": None,
        }

        for key in kwargs:
            if key not in defaults:
                raise ValueError(f"No kwarg '{key}'.\n\n" + 
                                 f"Only kwargs allowed are: {list(defaults.keys())}")

        return {key: kwargs.get(key, default) for key, default in defaults.items()}

class Flat:

    def calculate_e_length(self, u_coord, v_coord):

        return np.linalg.norm(u_coord - v_coord)
    
    def calculate_r_vw(self, u_coord, v_coord):

        return u_coord - v_coord
    
    def update_R(self, r, l):

        return np.tensordot(r, r, axes=0) / l
    
    def construct_g_coords(self, fixed_num_points=False, periodic_edges=None, **kwargs):

        if fixed_num_points: 
            try: num_points = kwargs["num_points"]
            except: num_points = 33
            calculate_points_per_edge = lambda _: num_points
        else: 
            # Make sure the discretization size is roughly equal on each edge.
            # Also make sure that there is a minimum of one discretization point
            # on each edge
            min_l_vw = np.min(np.array(list(self.E_lengths_by_v_num.values())))
            num_points = 9 / min_l_vw
            calculate_points_per_edge = lambda l: min(max(9, int(l * num_points)), 33)

        g_coords = []

        if periodic_edges:
            for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():
                if (v_num, w_num) in periodic_edges:
                    v_coord, w_coord = periodic_edges[v_num, w_num]
                else:
                    v_coord = self.V_coords[v_num]
                    w_coord = self.V_coords[w_num]

                points_per_edge = calculate_points_per_edge(l_vw)
                x = np.linspace(0, 1, points_per_edge)
                l = (x * w_coord[:, np.newaxis] + (1 - x) * v_coord[:, np.newaxis])
                g_coords.append(l)

        else:
            for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():
                v_coord = self.V_coords[v_num]
                w_coord = self.V_coords[w_num]

                points_per_edge = calculate_points_per_edge(l_vw)
                x = np.linspace(0, 1, points_per_edge)
                l = (x * w_coord[:, np.newaxis] + (1 - x) * v_coord[:, np.newaxis])
                g_coords.append(l)

        return g_coords
    
    def construct_R(self):

        R = [0 for _ in range(self.num_Vs)]

        for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():

            r_vw = self.calculate_r_vw(self.V_coords[v_num], self.V_coords[w_num])

            R[v_num] += self.update_R(r_vw, l_vw)
            R[w_num] += self.update_R(r_vw, l_vw)

        return R

    def construct_segments(self, coords):
        return np.stack((coords[:, :-1].T, coords[:, 1:].T), axis=1)

    def construct_colors_for_segments(self, eigen):

        color = []

        if eigen.shape[-1] > 2:
            color.append(eigen[0])
            color.extend(((eigen[1:-2] + eigen[2:-1]) / 2).tolist())
            color.append(eigen[-1])
        else:
            color.append(np.mean(eigen).tolist())

        return color

    def map_colors(self, colors, cmap):

        colors = np.array(colors)
        colors[np.abs(colors) < 1e-15] = 0
        colormap = eval(f"plt.cm.{cmap}")
        norm = plt.Normalize(vmin = np.min(colors), vmax = np.max(colors))
        colors = colormap(norm(colors))

        return colors
    
    def add_embedding_space(self, ax, space):

        # Originally had color='xkcd:periwinkle blue' with alpha=0.25
        # Equivalent with rgba is (0.89, 0.89, 0.978), but this looks closer instead:
        rgba_periwinkle_blue = (0.89, 0.902, 1)
        if self.num_Vs < 4000: linewidth = 4
        else: linewidth = 1
        if space == "square":
            x_min = np.min(self.V_coords[:, 0])
            y_min = np.min(self.V_coords[:, 1])
            square = matplotlib.patches.Rectangle((x_min, y_min), 1, 1, 
                                                   facecolor=rgba_periwinkle_blue, 
                                                   edgecolor=rgba_periwinkle_blue, linewidth=linewidth)
            ax.add_patch(square)
        elif space == "disc":
            circle = plt.Circle((0, 0), 1, color='xkcd:periwinkle blue', edgecolor='black', linewidth=0)
            ax.add_patch(circle)
        elif space == "hexagon_rectangle":
            x_min = np.min(self.V_coords[:, 0])
            y_min = np.min(self.V_coords[:, 1])
            rectangle = matplotlib.patches.Rectangle((x_min, y_min), 1, 1 / np.sqrt(3), 
                                                     facecolor=rgba_periwinkle_blue, 
                                                     edgecolor=rgba_periwinkle_blue, linewidth=linewidth)
            ax.add_patch(rectangle)

    def set_plot_geometry(self, fig, ax):

        fig.tight_layout()
        ax.autoscale(tight=True)
        ax.margins(x=0.01, y=0.01)
        ax.axis('off')
        ax.set_aspect("equal")

    def plot_graph(self, **kwargs):

        params = self.generate_plot_kwargs(**kwargs)

        if params["fig"]: fig = params["fig"]
        else: fig = plt.figure(figsize=(params["figsize"], params["figsize"]))

        # Set background color
        if params["background_color"]:
            fig.set_facecolor(params["background_color"])
            if params["dim"] == 3:
                ax.set_facecolor(params["background_color"])

        # Set ax and other functions depending on 2d or 3d plot
        if params["dim"] == 2:
            if params["ax"]: ax = params["ax"]
            else: ax = fig.add_subplot(111)
            line_collection = matplotlib.collections.LineCollection
            add_collection = ax.add_collection
        elif params["dim"] == 3:
            if params["ax"]: ax = params["ax"]
            else: ax = fig.add_subplot(111, projection="3d")
            line_collection = mpl_toolkits.mplot3d.art3d.Line3DCollection
            add_collection = ax.add_collection3d
            ax.view_init(*params["view"])

        # Plot background embedding space
        if params["embedding_space"] is not None:
            self.add_embedding_space(ax, params["embedding_space"])

        # Construct segmentation of each edge for color plotting
        segments = []
        dxs = []
        if params["eigenmode"] is not None:
            colors = []

            for coords, eigen in zip(self.g_coords, params["eigenmode"]):
                coords_zipped = self.construct_segments(coords)
                dxs.append(len(coords_zipped))
                segments.extend(coords_zipped)

                color = self.construct_colors_for_segments(eigen)
                colors.extend(color)

            colors = self.map_colors(colors, params["cmap"])

        else:
            for coords in self.g_coords:
                coords_zipped = self.construct_segments(coords)
                dxs.append(len(coords_zipped))
                segments.extend(coords_zipped)

            colors = params["colors"]
        segments = np.array(segments)

        # Add line_collection based off segments and colors. Nice capstyle ("round") is expensive
        if params["capstyle"]:
            line_collection = line_collection(segments, colors=colors, linewidth=params["linewidth"],
                                              path_effects=[path_effects.Stroke(capstyle=params["capstyle"])])
        else:
            line_collection = line_collection(segments, colors=colors, linewidth=params["linewidth"])
        add_collection(line_collection)

        # Set linewidth(s)
        if isinstance(params["linewidth"], (list, np.ndarray)):
            params["linewidth"] = np.repeat(params["linewidth"], dxs)

        # Set bordering and aspect ratio
        self.set_plot_geometry(fig, ax)
        # Return plt objects of show
        if params["return_figax"]: 
            return fig, ax
        else: 
            plt.show()

    
class Spherical:

    """Requires self.spher_V_coords.
    """

    def calculate_e_length(self, u_coord, v_coord):

        dot = np.dot(u_coord, v_coord)
        cross = np.linalg.norm(np.cross(u_coord, v_coord))

        return np.arctan2(cross, dot)

    def calculate_r_vw(self, u_coord, v_coord):

        dot = np.dot(u_coord, v_coord)

        tangent_u = self.normalize_array(dot * u_coord - v_coord)
        tangent_v = self.normalize_array(dot * v_coord - u_coord)

        return tangent_u, tangent_v
    
    def update_R(self, r, l):

        return np.tensordot(r, r, axes=0) * l
    
    def construct_cart_to_spher_transformation_matrix(self, theta, phi):

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        return np.array(([cos_theta * cos_phi, -sin_phi, sin_theta * cos_phi],
                         [cos_theta * sin_phi, cos_phi, sin_theta * sin_phi],
                         [-sin_theta, 0, cos_theta]))
    
    def convert_V_coords_to_spher(self):
        
        return self.cart_to_spher(self.V_coords[:, 0], self.V_coords[:, 1], self.V_coords[:, 2])

    def cart_to_spher(self, x, y, z):

        x, y, z = (self.convert_to_column_vector(i) for i in [x, y, z])

        xy = x**2 + y**2

        return np.hstack([np.arctan2(np.sqrt(xy), z), np.arctan2(y, x), np.sqrt(xy + z**2)])
    
    def construct_g_coords(self, num_points=10, fixed_num_points=True):

        if fixed_num_points: calculate_points_per_edge = lambda _: num_points
        else: calculate_points_per_edge = lambda l: max(3, int(l * num_points))

        g_coords = []

        for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():
            v_coord = self.V_coords[v_num]
            w_coord = self.V_coords[w_num]

            points_per_edge = calculate_points_per_edge(l_vw)
            x = np.linspace(0, 1, points_per_edge)
            l = (x * w_coord[:, np.newaxis] + (1 - x) * v_coord[:, np.newaxis])
            l /= np.linalg.norm(l, axis=0, keepdims=True)

            g_coords.append(l)

        return g_coords
    
    def construct_R(self):

        R = [0 for _ in range(self.num_Vs)]

        for (v_num, w_num), l_vw in self.E_lengths_by_v_num.items():

            r_vw, r_wv = self.calculate_r_vw(self.V_coords[v_num], self.V_coords[w_num])

            R[v_num] += self.update_R(r_vw, l_vw)
            R[w_num] += self.update_R(r_wv, l_vw)

        for r_num, (theta, phi, _) in enumerate(self.spher_V_coords):
            transf = self.construct_cart_to_spher_transformation_matrix(theta, phi)
            R[r_num] = (transf.T @ R[r_num] @ transf)[:-1, :-1]

        return R