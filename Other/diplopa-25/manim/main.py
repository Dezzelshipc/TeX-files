from manim import *
from manim_slides import Slide
from liba import *

colors = [ BLUE, GREEN, PURPLE, RED, ORANGE, YELLOW ]

class ExampleFlow1(Slide):
    def sw(self, time = 0.05):
        self.wait(time)
        self.next_slide()

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
        self.sw()

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
        self.sw()

        self.play(
            Q.tracker.animate.set_value(12), 
            run_time=4
        )
        self.sw()
        
        self.play(
            x_max.animate.set_value(100),
            y_max.animate.set_value(1), 
            run_time=2
        )
        self.sw()
        self.play(
            Q.tracker.animate.set_value(13), 
            run_time=2
        )
        self.sw()        
        
        self.play(
            y_max.animate.set_value(6), 
            run_time=2
        )
        self.sw()
        self.play(
            Q.tracker.animate.set_value(95), 
            run_time=5
        )
        self.sw()

        self.play(
            Q.tracker.animate.set_value(100), 
            run_time=2
        )
        self.sw()        
        self.play(
            x_max.animate.set_value(20),
            y_max.animate.set_value(15), 
            run_time=2
        )
        self.sw()
        self.play(
            Q.tracker.animate.set_value(1000), 
            run_time=5
        )
        self.sw()
        self.play(
            Uncreate(axes),
            Uncreate(curves[:4]),
            Unwrite(curves[4:]),
            Unwrite(axes_labels), 
            Unwrite(Q),
            Unwrite(species),
            run_time=2
        )
        self.sw()
