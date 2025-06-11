import numpy as np
import copy
from dataclasses import dataclass
from pathlib import Path

from manim import *
from manim_slides.slide import Slide

@dataclass
class DataClass:
    Q: float
    n: int | tuple[int, int]
    alpha: np.ndarray
    k: np.ndarray
    m: np.ndarray
    a: np.ndarray

    s: int | None = None
    alpha_b: float | None = None

    def __copy__(self):
        return DataClass(
            Q = self.Q,
            n = self.n,
            s = self.s,
            alpha = self.alpha,
            alpha_b = self.alpha_b,
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
    
    @staticmethod
    def get_example_split_base():
        _q = 4
        _r = 2

        alpha = np.linspace(20, 10, _q+1)
        k = np.append([0], np.linspace(0.5, 0.2, _q))
        m = np.append([0], np.linspace(5, 2, _q))
        a = np.append([0, 0.2], [0] * (_q-1))

        alpha_b = 16
        alpha2 = np.append([alpha_b], np.linspace(16, 8, _r)) 
        k2 = np.append([0], np.linspace(0.5, 0.3, _r))
        m2 = np.append([0], np.linspace(4, 1, _r))
        a2 = np.append([0, 0.0], np.array([0] * (_r-1)))

        alpha = np.append(alpha, alpha2[1:])
        k = np.append(k, k2[1:])
        m = np.append(m, m2[1:])
        a = np.append(a, a2[1:])

        return DataClass(
            Q = 0,
            n = (_q,_r),
            alpha = alpha,
            k = k,
            m = m,
            a = a,
        )
    
    @staticmethod
    def get_example_split_s1a8():
        data = DataClass.get_example_split_base()
        data.alpha_b = 8
        data.s = 1

        return data
    
    @staticmethod
    def get_example_split_s3a8():
        data = DataClass.get_example_split_base()
        data.alpha_b = 8
        data.s = 3

        return data
    

def save_numpy_deco(func):
    def _wrapper(*args, **kwargs):
        _Q = kwargs['Q']
        _f = kwargs['function'].__name__
        _y0 = kwargs['y0']
        _t = kwargs['time_space_params']
        _add = kwargs["add"] or ""
        save_path = Path(__file__).parent / "save_rk_result"

        save_path_x = save_path / "X"
        save_path_y = save_path / f"Y{_add}" / f"{_f}_{_y0}_{_t}"

        save_path_x.mkdir(parents=True, exist_ok=True)
        save_path_y.mkdir(parents=True, exist_ok=True)

        save_path_x = save_path_x / f"X_{_t}.npy".replace(" ", "")
        save_path_y = save_path_y / f"Y_{_Q}.npy".replace(" ", "")

        if save_path_y.exists():
            # print(f"LOADING : {save_path}")
            return np.load(save_path_x), np.load(save_path_y)
        else:
            # print(f"SAVING : {save_path}")
            X, Y = func(*args, **kwargs)
            if not save_path_x.exists():
                np.save(save_path_x, X)
            np.save(save_path_y, Y)
            return X, Y
    return _wrapper

@save_numpy_deco
def runge_kutta(function, y0: tuple | float, time_space_params: tuple[float, float], **kwargs) -> tuple[np.ndarray, np.ndarray]:
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
    
    assert isinstance(n, int)

    def right_flow(t, x):
        return np.append(
            [Q - alpha[0] * func_v(x[0]) * x[1]],
            [
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < n-1 else 0 ) * alpha[i] * func_v(x[i])
                for i in range(1, n)
            ]
        )
    return right_flow

def get_right_split(func_v, data: DataClass):
    assert isinstance(data.n, tuple)

    s = data.s
    Q = data.Q
    _q, _r = data.n
    alpha = data.alpha
    alpha_b = data.alpha_b
    k = data.k
    m = data.m
    a = data.a

    assert s > 0
    assert alpha_b > 0

    cc = a*m
    def right_split(_, x):
        return np.array([
            *[Q - alpha[0] * func_v(x[0]) * x[1] + sum(cc * x)],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - x[i+1] * alpha[i] * func_v(x[i])
                for i in range(1, s)
            ],
            *[
                -m[s] * x[s] + k[s] * alpha[s-1] * func_v(x[s-1]) * x[s] - x[s+1] * alpha[s] * func_v(x[s]) - (alpha_b * func_v(x[s]) * x[_q+1] if _r > 0 else 0)
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < _q else 0 ) * alpha[i] * func_v(x[i])
                for i in range(s+1, _q+1)
            ],
            *[
                -m[_q+1] * x[_q+1] + k[_q+1] * alpha_b * func_v(x[s]) * x[_q+1] - (x[_q+2] * alpha[_q+1] * func_v(x[_q+1]) if _r > 1 else 0 )
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < _q+_r else 0 ) * alpha[i] * func_v(x[i])
                for i in range(_q+2, _q+_r+1)
            ],
        ])
    return right_split

def identity(x):
    return x

if __name__ == "__main__":
    data = DataClass.get_example1()
            
    right_flow = get_right_flow(identity, data)

    h = 0.01
    X, Y = runge_kutta(
        function=right_flow,
        time_space_params=(100, h),
        y0=(2,2,2,2),
        Q=data.Q
    )
 
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

Text.set_default(font="Times New Roman")

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


def AnimGroupFunc(mobjects, func, lag_ratio=0, **kwargs):
    return AnimationGroup(*map(lambda x: func(x, **kwargs), mobjects), lag_ratio=lag_ratio)