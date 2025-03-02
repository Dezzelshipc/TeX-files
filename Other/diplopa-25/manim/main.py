from manim import *
from manim_slides import Slide

class Start(Slide):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square

        self.next_slide() 

        self.play(Transform(square, circle))  # interpolate the square into the circle
        
        self.next_slide() 
        
        self.play(FadeOut(square))  # fade out animation
