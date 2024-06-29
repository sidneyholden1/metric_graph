import numpy as np
import scipy
from construct_graph.graph import Graph, Flat
import matplotlib.pyplot as plt


class Spiderweb(Graph, Flat):

    def __init__(self, num_radial_Vs, num_angular_Vs, j=None, rtype=0, **kwargs):

        self.num_radial_Vs = num_radial_Vs
        self.num_angular_Vs = num_angular_Vs
        self.num_Vs = self.num_angular_Vs * (self.num_radial_Vs - 1) + 1
        self.j = j
        self.rtype = rtype

        self.dtheta = 2 * np.pi / self.num_angular_Vs

        r = self.construct_radial_distribution(rtype=self.rtype)
        theta = np.linspace(0, 2 * np.pi, self.num_angular_Vs, endpoint=False)
        self.radial_lengths = np.array([np.linalg.norm(r[i] - r[i - 1])
                                        for i in range(1, self.num_radial_Vs)])

        self.V_coords = np.array([[0, 0]] + [[i * np.cos(j), i * np.sin(j)]
                                             for i in r[1:] for j in theta])

        self.interior_V_num = np.arange(self.num_Vs - self.num_angular_Vs)
        self.E_lengths_by_v_num = self.construct_E_lengths_by_v_num()
        self.wadjacency_matrix = self.construct_wadjacency_matrix_and_fill_E_lengths_by_v_num()
        self.g_coords = self.construct_g_coords()

        self.plot_graph(**kwargs)
        
    def construct_radial_distribution(self, rtype=0):  # [..., 1(1-dtheta)(1-dtheta), 1(1-dtheta), 1]

        if rtype == 0:
            dtheta = 2 * np.pi / self.num_angular_Vs

            radial_distribution = [1]

            for _ in range(self.num_radial_Vs - 2):
                radial_distribution = [radial_distribution[0] * (1 - dtheta)] + radial_distribution

            return np.array([0] + radial_distribution)

        elif rtype == 1:
            dtheta = 2 * np.pi / (self.num_angular_Vs + 1)

            radial_distribution = [1]

            for _ in range(self.num_radial_Vs - 1):
                radial_distribution = [radial_distribution[0] * (1 - dtheta)] + radial_distribution

            radial_distribution = np.array(radial_distribution)
            radial_distribution -= radial_distribution[0]
            radial_distribution /= radial_distribution[-1]

            return radial_distribution

    def construct_E_lengths_by_v_num(self):

        E_lengths_by_v_num = [[0, w_num] for w_num in np.arange(1, self.num_angular_Vs + 1)]

        for i in range(1, self.num_angular_Vs + 1):
            neighbours = [np.mod(i, self.num_angular_Vs) + 1,
                          np.mod(i + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                          i + self.num_angular_Vs]

            E_lengths_by_v_num += [[i, neighbour] for neighbour in neighbours]

            a = self.num_Vs - self.num_angular_Vs + i - 1
            b = self.num_angular_Vs * (self.num_radial_Vs - 2)
            neighbours = [b + np.mod(a, self.num_angular_Vs) + 1,
                          b + np.mod(a + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                          a - self.num_angular_Vs]

            E_lengths_by_v_num += [[a, neighbour] for neighbour in neighbours]

        for i in range(1, self.num_radial_Vs - 2):

            a = i * self.num_angular_Vs + 1

            for j in range(self.num_angular_Vs):
                b = a + j
                c = self.num_angular_Vs * i
                neighbours = [c + np.mod(b, self.num_angular_Vs) + 1,
                              c + np.mod(b + self.num_angular_Vs - 2, self.num_angular_Vs) + 1,
                              b - self.num_angular_Vs,
                              b + self.num_angular_Vs]

                E_lengths_by_v_num += [[b, neighbour] for neighbour in neighbours]

        E_lengths_by_v_num = np.unique(np.sort(E_lengths_by_v_num, axis=1), axis=0)
        E_lengths_by_v_num = {(v, w): None for v,w in E_lengths_by_v_num}

        return E_lengths_by_v_num

    def construct_L(self, k, deriv=False):

        matrix = scipy.sparse.lil_matrix((self.num_radial_Vs, self.num_radial_Vs), dtype=np.float64)

        if not deriv:
            calculate_csc = self.calculate_csc
            calculate_cot = self.calculate_cot
            calculate_sec = self.calculate_sec
            matrix[0, 0] = 1
            matrix[-1, -1] = 1

        else:
            calculate_csc = self.calculate_dcsc
            calculate_cot = self.calculate_dcot
            calculate_sec = self.calculate_dsec

        if self.j == 0:
            matrix[0, 1] = -calculate_sec(k, self.radial_lengths[0])

        for i in range(1, self.num_radial_Vs - 1):
            back = self.radial_lengths[i - 1]
            forward = self.radial_lengths[i]

            rho = 2 * np.sum(self.radial_lengths[:i]) * np.sin(self.dtheta / 2)

            matrix[i, i - 1] = -calculate_csc(k, back)
            matrix[i, i] = (calculate_cot(k, back) + calculate_cot(k, forward)
                            + 2 * (calculate_cot(k, rho) - np.cos(self.j * self.dtheta) * calculate_csc(k, rho)))
            matrix[i, i + 1] = -calculate_csc(k, forward)

        return matrix.tocsc()
    
    def return_pde_data(self, rtype=0):

        """Returns m x n array of eigenvalues.
        The m==0 eigenvalues are (m + 1/2) * pi / sqrt(2).
        The m==1, 2 eigenvalues are the zeta_{m, n} / sqrt(2), where zeta_{m, n} is 
        the nth zero of the the Bessel function of order sqrt(1/4 + m^2)}.
        """

        if rtype==0:
            return np.array(([1.110720734539592, 3.332162203618775, 5.553603672697958],
                             [2.821392690363739, 5.080521931144712, 7.316750028199640],
                             [3.686578856962347, 6.011391581456534, 8.278125681120072]))