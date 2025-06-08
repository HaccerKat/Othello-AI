import random
import numpy as np
import time

SIZE = 100000
AMOUNT = 10
a = random.sample(range(10**6), SIZE)
weights = np.random.dirichlet(np.ones(SIZE) * 100)

# random.choices -> better when AMOUNT is small due to lower constant overhead
start = time.perf_counter()
selections = random.choices(population=a, weights=weights, k=AMOUNT)
end = time.perf_counter()
print("Time for random.choices:", end - start)

# np.random.choices -> better when AMOUNT is large
start = time.perf_counter()
selections = np.random.choice(a, AMOUNT, p=weights)
end = time.perf_counter()
print("Time for np.random.choices:", end - start)