import time
from math import fmod


class PseudoRandom:
    def __init__(self):
        self.sigma = 10
        self.r = 28
        self.b = 2.66
        self.n_min, self.n_max = 100, 1000
        self.x_range = (2.8915, 3.2027)
        self.y_range = (1.4296, 1.7365)
        self.z_range = (15.2113, 16.1852)
        self.dt_range = (1e-20, 0.1)
        self.entropy = 1

    def fx(self, x, y, z):
        return self.sigma * (y - x)

    def fy(self, x, y, z):
        return x * (self.r - z) - y

    def fz(self, x, y, z):
        return x * y - self.b * z

    @staticmethod
    def get_rand(seed, values: tuple, div):
        return values[0] + fmod(seed, div) * (values[1] - values[0]) / div

    def generate(self):
        self.entropy = time.time() * 1000

        x_i = self.get_rand(self.entropy, self.x_range, 463)
        y_i = self.get_rand(self.entropy, self.y_range, 539)
        z_i = self.get_rand(self.entropy, self.z_range, 822)
        dt = self.get_rand(self.entropy, self.dt_range, 1000)
        n = int(self.n_min + self.get_rand(self.entropy, (1, self.n_max - self.n_min), self.n_max - self.n_min))

        cut = 10000
        for _ in range(n):
            x_ = x_i + dt * self.fx(x_i, y_i, z_i)
            y_ = y_i + dt * self.fy(x_i, y_i, z_i)
            z_ = z_i + dt * self.fz(x_i, y_i, z_i)
            x_i = fmod(x_, cut)
            y_i = fmod(y_, cut)
            z_i = fmod(z_, cut)

        return abs(fmod(x_i, 1))
