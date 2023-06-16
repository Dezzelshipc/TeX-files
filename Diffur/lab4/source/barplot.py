import random

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from task3 import PseudoRandom


class Test:

    @staticmethod
    def const_test(c=1, n=10000):
        pr = PseudoRandom()
        lst = [pr.generate() // (10 ** (-c)) / (10 ** c) for _ in range(n)]
        # print(lst)
        counts = []

        for i in range(10 ** c):
            counts.append(lst.count(i / (10 ** c)))

        plt.bar(np.linspace(0.0, 1.0 - 1 / (10 ** c), 10 ** c), counts, width=1 / (10 ** c), edgecolor='k',
                align='edge')
        # plt.savefig(f'task3_{c}.pdf')
        plt.show()

    @staticmethod
    def dynamic_test(c=1, n=10000):
        spread = 10

        def animate(n_frame):
            for _ in range(spread):
                new = int(pr.generate() // (10 ** (-c)))
                bar[new].set_height(bar[new].get_height() + 1)
            return bar

        pr = PseudoRandom()

        fig, ax = plt.subplots()
        ax.set_ylim(0, n / (10 ** c) + 100)

        x = np.linspace(0.0, 1.0 - 1 / (10 ** c), 10 ** c)
        bar = plt.bar(x, np.zeros(10 ** c), width=0.99 / (10 ** c), edgecolor='k', align='edge')

        ani = anim.FuncAnimation(fig, animate, blit=True, interval=0, frames=n // spread, repeat=False)
        plt.show()


# Test.dynamic_test(2, 10000)
Test.const_test(2, 100000)