import math

def f(x):
    return math.pow(5*x+2, 1/6)

x = 1

for i in range(10000000):
    x = f(x)
print(x)
