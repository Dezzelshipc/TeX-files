from manim import *
from manim_slides import Slide

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
