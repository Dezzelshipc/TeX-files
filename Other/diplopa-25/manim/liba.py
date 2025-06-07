import numpy as np
import copy
from dataclasses import dataclass

from cachetools.func import lru_cache

from manim import *
from manim_slides import Slide

@dataclass
class DataClass:
    Q: float
    n: int
    alpha: np.ndarray
    k: np.ndarray
    m: np.ndarray
    a: np.ndarray

    def __copy__(self):
        return DataClass(
            Q = self.Q,
            n = self.n,
            alpha = self.alpha,
            k = self.k,
            m = self.m,
            a = self.a
        )
    
    def __hash__(self):
        return sum( field * 11**i for i, field in enumerate([self.n, hash(self.alpha.sum()), hash(self.k.sum()), hash(self.m.sum()), hash(self.a.sum()), hash(self.Q)]) )

    def with_Q(self, Q):
        data = copy.copy(self)
        data.Q = Q
        return data
    
    @staticmethod
    def get_example1():
        return DataClass(
            Q = 0,
            n = 4,
            alpha = np.array([20, 16, 12, 8]), # n
            k = np.append([0], [0.3, 0.2, 0.1]), # n-1, 0<=k_i<=1
            m = np.append([0], [4, 3, 2]), # n-1
            a = np.append([0], [0.9, 0.9, 0.9]), # n-1, 0<=a_i<=1
        )
    
@lru_cache(maxsize=1024)
def runge_kutta(function, y0: tuple | float, time_space_params: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    time_space = np.arange(0, *time_space_params)
    h = time_space[1] - time_space[0]
    num = len(time_space)
    x_a = time_space

    y_a = [y0] * (num)

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a, dtype='float64')


def get_right_flow(func_v, data: DataClass):
    Q = data.Q
    n = data.n
    alpha = data.alpha
    k = data.k
    m = data.m
    def right_flow(t, x):
        return np.append(
            [Q - alpha[0] * func_v(x[0]) * x[1]],
            [
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < n-1 else 0 ) * alpha[i] * func_v(x[i])
                for i in range(1, n)
            ]
        )
    return right_flow

def identity(x):
    return x


class Title2(Text):
    def __init__(
        self,
        *text_parts,
        include_underline=True,
        match_underline_width_to_text=False,
        underline_buff=MED_SMALL_BUFF,
        **kwargs,
    ):
        self.include_underline = include_underline
        self.match_underline_width_to_text = match_underline_width_to_text
        self.underline_buff = underline_buff
        kwargs["font"] = kwargs.get("font") or "Times New Roman"
        super().__init__(*text_parts, **kwargs)
        self.to_edge(UP)
        if self.include_underline:
            underline_width = config["frame_width"] - 2
            underline = Line(LEFT, RIGHT)
            underline.next_to(self, DOWN, buff=self.underline_buff)
            if self.match_underline_width_to_text:
                underline.match_width(self)
            else:
                underline.width = underline_width
            self.add(underline)
            self.underline = underline

class Tikz(MathTex):
    def __init__(
            self, *tex_strings, arg_separator="", tex_environment="tikzpicture", **kwargs
    ):
        kwargs["stroke_width"] = kwargs.get("stroke_width") or 1 
        kwargs["tex_template"] = kwargs.get("tex_template") or TexTemplate(tex_compiler='xelatex',output_format='.pdf')\
            .add_to_preamble(r"""\usepackage{tikz}
                    \usepackage{polyglossia}
                    \usepackage{xecyr}
                             
                    \setmainfont{Times New Roman} 
                    \newfontfamily{\cyrillicfont}{Times New Roman}""")\
            .add_to_document(r"\usetikzlibrary{shapes.geometric, calc, arrows.meta, shapes.multipart}")
        super().__init__(
            *tex_strings,
            arg_separator=arg_separator,
            tex_environment=tex_environment,
            **kwargs,
        )

class SWSlide(Slide):
    def sw(self, time = 0):
        shortest_time = 1/self.camera.frame_rate
        if time < shortest_time:
            time = shortest_time

        self.wait(time)
        _marker = Dot(color=YELLOW).to_corner(DR).shift(0.55*DR).set_opacity(0.2)
        self.add(_marker)
        self.wait(shortest_time)
        self.next_slide()
        self.remove(_marker)

class PageNum(Integer):
    def __init__(self, number = 0, edge = DR, **kwargs):
        super().__init__(number, 0, **kwargs)
        self.edge = edge
        self.to_corner(edge)

    def bump(self):
        self.increment_value(1)
        self.to_corner(self.edge)