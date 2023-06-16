import time
from math import fmod


class PseudoRandom:
    def __init__(self, seed : float = None):
        self.seed(seed)
        self.sigma = 10
        self.r = 28
        self.b = 2.66
        self.n_range = (100, 1000)
        self.x_range = (2.8915, 3.2027)
        self.y_range = (1.4296, 1.7365)
        self.z_range = (15.2113, 16.1852)
        self.dt_range = (1e-20, 0.1)

    def fx(self, x, y, z):
        return self.sigma * (y - x)

    def fy(self, x, y, z):
        return x * (self.r - z) - y

    def fz(self, x, y, z):
        return x * y - self.b * z
    
    def seed(self, seed : float = None):
        self._seed = seed if seed is not None else time.time_ns()
        self.entropy = self._seed

    def get_seed(self):
        return self._seed
    
    def uniform(self, values: tuple[2], div):
        return values[0] + fmod(self.entropy+div/3, div) * (values[1] - values[0]) / div

    def generate(self):
        x_i = self.uniform(self.x_range, 12399)
        y_i = self.uniform(self.y_range, 874323)
        z_i = self.uniform(self.z_range, 56664)
        dt = self.uniform(self.dt_range, 230487)
        n = int( self.uniform(self.n_range, 1000) )

        cut = 10000
        for _ in range(n):
            x_ = x_i + dt * self.fx(x_i, y_i, z_i)
            y_ = y_i + dt * self.fy(x_i, y_i, z_i)
            z_ = z_i + dt * self.fz(x_i, y_i, z_i)
            x_i = fmod(x_, cut)
            y_i = fmod(y_, cut)
            z_i = fmod(z_, cut)

        number = abs(fmod(x_i, 1))

        self.entropy = number * 1e+10

        return number