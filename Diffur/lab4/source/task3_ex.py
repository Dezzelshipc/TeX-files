from task3 import PseudoRandom

seed = 10

ps = PseudoRandom(seed)
for i in range(10):
    print(ps.generate())

print()

ps.seed(seed)
for i in range(10):
    print(ps.generate())

