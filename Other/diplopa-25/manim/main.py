from manim import *
from manim_slides import Slide
from liba import *
# import matplotlib.pyplot as plt

class Tikz(MathTex):
    def __init__(
            self, *tex_strings, arg_separator="", tex_environment="tikzpicture", **kwargs
    ):
        kwargs["stroke_width"] = kwargs.get("stroke_width") or 1 
        kwargs["tex_template"] = kwargs.get("tex_template") or TexTemplate()\
            .add_to_preamble(r"\usepackage{tikz}")\
            .add_to_document(r"\usetikzlibrary{shapes.geometric, calc}")
        super().__init__(
            *tex_strings,
            arg_separator=arg_separator,
            tex_environment=tex_environment,
            **kwargs,
        )

class Intro(Slide):
    def construct(self):
        t1 = Tex("1").to_corner(DL).set_opacity(0.1)
        self.add(t1)
         
        self.wait()
        self.next_slide()
        self.remove(t1)

        logo = SVGMobject("assets/logo").scale(1.5)
        self.play(Write(logo), run_time=2)
        self.wait(0.05)

        self.play(LaggedStart( ApplyWave(logo), Circumscribe(logo, Circle), lag_ratio=0.25))

        names = Text("Держапольский Юрий Витальевич\nБ9121-01.03.02сп").scale(0.5).to_corner(DL)

        title = VGroup(
            Text("Моделирование трофических сетей").scale(1.2),
            Text("Особенности динамики видов в трофических цепях").move_to(DOWN)
        ).scale(0.7).move_to(ORIGIN)

        self.play(
            LaggedStart( 
            logo.animate.scale(0.5).to_corner(DR),
            [
            Write(names),
            Write(title)
            ],
            lag_ratio=0.1)
        )

        self.next_slide()

        self.play(
            FadeOut(logo, shift=DOWN),
            FadeOut(title, shift=DOWN),
            FadeOut(names, shift=DOWN)
        )

        self.wait()

colors = [ BLUE, ORANGE, GREEN, RED, PURPLE, YELLOW ]

class ExampleFlow1(Slide):
    def construct(self):
        def get_graph_points(data_main: DataClass, func_v, y0) -> VGroup:
            data = data_main.with_Q(Q.tracker.get_value())
            
            right_flow = get_right_flow(func_v, data)

            h = 0.01
            X, Y = runge_kutta(
                function=right_flow,
                time_space_params=(x_max.get_value(), h),
                y0=y0
            )

            _axes = Axes(
                x_range=[0, x_max.get_value()],
                y_range=[0, y_max.get_value()]
            )

            return VGroup(*[ 
                VMobject().set_points_as_corners(_axes.c2p(np.matrix([X, y_i]).T)).set_color(clr)
                for y_i, clr in zip(Y.T, colors)
            ], *[
                DecimalNumber(y_i[-1], num_decimal_places=3).move_to(_axes.c2p(X[-1], y_i[-1])).shift(UP/4).set_color(clr)
                for y_i, clr in zip(Y.T, colors)
            ])
        
        x_max = ValueTracker(20)
        y_max = ValueTracker(2)
        Q = Variable(0.5, "Q", num_decimal_places=2).to_corner(UP)
        Q.value.group_with_commas=False

        axes = always_redraw(lambda: Axes(
            x_range=[0, x_max.get_value(), x_max.get_value()//10],
            y_range=[0, y_max.get_value(), max(0.5, y_max.get_value()//4)],
            ).add_coordinates()
        )

        x_label = axes.get_x_axis_label("t", edge=RIGHT, direction=DR)
        y_label = axes.get_y_axis_label("N", edge=UP, direction=UL)
        axes_labels = VGroup(x_label, y_label)


        self.add(axes, axes_labels, Q)
        self.play(Write(axes), Write(axes_labels), Write(Q))
        self.wait(0.1)
        self.next_section()

        data = DataClass.get_example1()
        y0 = (2,)*data.n
        
        curves = always_redraw(lambda: get_graph_points(data, identity, y0))

        
        species = VGroup(*[
            MathTex(f"N_{i}").set_color(c) for i, c in enumerate(colors[:data.n])
        ]).arrange().to_corner(UR)

        self.play(
            Succession(
                Create(curves[:4], run_time=3),
                Write(curves[4:], run_time=1)
            ),
            Write(species, run_time=1)
        )
        self.add(curves, species)
        self.wait(0.1)
        self.next_slide()


        self.play(
            Q.tracker.animate.set_value(12), 
            run_time=4
        )
        self.wait(0.1)
        self.next_slide()

        
        self.play(
            x_max.animate.set_value(100),
            y_max.animate.set_value(1), 
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()

        self.play(
            Q.tracker.animate.set_value(13), 
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()
        
        
        self.play(
            y_max.animate.set_value(6), 
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()

        self.play(
            Q.tracker.animate.set_value(95), 
            run_time=5
        )
        self.wait(0.1)
        self.next_slide()


        self.play(
            Q.tracker.animate.set_value(100), 
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()
        
        self.play(
            x_max.animate.set_value(20),
            y_max.animate.set_value(15), 
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()

        self.play(
            Q.tracker.animate.set_value(1000), 
            run_time=5
        )
        self.wait(0.1)
        self.next_slide()

        self.play(
            Uncreate(axes),
            Uncreate(curves),
            Unwrite(axes_labels), 
            Unwrite(Q),
            Unwrite(species),
            run_time=2
        )
        self.wait(0.1)
        self.next_slide()


class FirstTwoModels(Slide):
    def construct(self):
        graph1 = Tikz(r"""
        \tikzstyle{roundnode} = [draw, circle, text centered];
        \tikzstyle{squarenode} = [draw, regular polygon, regular polygon sides=4, text centered, inner sep=0];
        \tikzstyle{arrow} = [thick, ->, >=stealth];
    
        % Left - Flow
        \node[roundnode] (RF) at (0,8) {$R$};

        \node[squarenode] (NF1) at (0,6) {$N_1$};
        \node (NF1M) at (2,6) {$m_1 N_1$};
        \draw [arrow] (NF1) -- (NF1M);

        \node[squarenode] (NF2) at (0,4) {$N_2$};
        \node (NF2M) at (2,4) {$m_2 N_2$};
        \draw [arrow] (NF2) -- (NF2M);

        \node (DF) at (0,2) {$\vdots\vphantom{lp}$};

        \node[squarenode] (NFN) at (0,0) {$N_n$};
        \node (NFNM) at (2,0) {$m_n N_n$};
        \draw [arrow] (NFN) -- (NFNM);
        
        \draw [arrow] (0,10) -- node[anchor=east] {$Q$} (RF);
        \draw [arrow] (RF) --   node[anchor=east] {$V_0(R)$} (NF1);
        \draw [arrow] (NF1) --  node[anchor=east] {$V_1(N_1)$} (NF2);
        \draw [arrow] (NF2) --  node[anchor=east] {$V_2(N_2)$} (DF);
        \draw [arrow] (DF) --   node[anchor=east] {$V_{n-1}(N_{n-1})$} (NFN);


        \path[arrow] (NF1) edge [loop left] node {$k_1 V_0$} ();
        \path[arrow] (NF2) edge [loop left] node {$k_2 V_1$} ();
        \path[arrow] (NFN) edge [loop left] node {$k_n V_{n-1}$} ();
            """
        ).scale(0.5)
        self.play(Write(graph1), run_time=4)
        self.wait()
        self.next_slide()

        graph2 = Tikz(r"""
        \tikzstyle{roundnode} = [draw, circle, text centered];
        \tikzstyle{squarenode} = [draw, regular polygon, regular polygon sides=4, text centered, inner sep=0];
        \tikzstyle{arrow} = [thick, ->, >=stealth];
    
        \node (BTC) at (11, 10) {};
        \node (BBC) at (11, 0) {};

        \node[roundnode] (RC) at (8,8) {$R$};

        \node[squarenode] (NC1) at (8,6) {$N_1$};
        \draw [arrow] (NC1) -- node[anchor=south] {$m_1 N_1$} ($(BTC)!(NC1)!(BBC)$);

        \node[squarenode] (NC2) at (8,4) {$N_2$};
        \draw [arrow] (NC2) -- node[anchor=south] {$m_2 N_2$} ($(BTC)!(NC2)!(BBC)$);

        \node (DC) at (8,2) {$\vdots\vphantom{lp}$};
        \node (DC2) at ($(BTC)!(DC)!(BBC)$) {$\vdots\vphantom{lp}$};

        \node[squarenode] (NCN) at (8,0) {$N_n$};
        \draw [arrow] (NCN) -- node[anchor=south] {$m_n N_n$} ($(BTC)!(NCN)!(BBC)$);
        
        \draw [arrow] (8,10) -- node[anchor=east] {$Q$} (RC);
        \draw [arrow] (RC) --   node[anchor=east] {$V_0(R)$} (NC1);
        \draw [arrow] (NC1) --  node[anchor=east] {$V_1(N_1)$} (NC2);
        \draw [arrow] (NC2) --  node[anchor=east] {$V_2(N_2)$} (DC);
        \draw [arrow] (DC) --   node[anchor=east] {$V_{n-1}(N_{n-1})$} (NCN);
        
        \draw [arrow] (DC2) |- node[pos=0.75, anchor=south] 
        {$\textstyle\sum\limits_{i=1}^n a_i m_i N_i$} (RC);
        
        \path[arrow] (NC1) edge [loop left] node {$k_1 V_0$} ();
        \path[arrow] (NC2) edge [loop left] node {$k_2 V_1$} ();
        \path[arrow] (NCN) edge [loop left] node (KNVN1) {$k_n V_{n-1}$} ();

        \draw [arrow] ($(KNVN1)!(BTC)!(NCN)$) -- (DC2);
            """
        ).scale(0.5)
        self.play(TransformMatchingShapes(graph1, graph2, fade_transform_mismatches=True), run_time=1)
        self.wait()
        self.next_slide()