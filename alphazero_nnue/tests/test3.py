import numpy as np
import time
import math

a = 0x123456789ABCDEF0
b = 0x0F0F0F0F0F0F0F0F
# print(math.log10(b))
a_np = np.uint64(a)
b_np = np.uint64(b)

# Python int
start = time.perf_counter()
for _ in range(10**7):
    x = (a & b) ^ (b << 2)
end = time.perf_counter()
print("Python int:", end - start)

# NumPy uint64
start = time.perf_counter()
for _ in range(10**5):
    x = np.bitwise_or((np.bitwise_and(int(a_np), int(b_np))), (np.left_shift(int(b_np), 2)))
end = time.perf_counter()
print("NumPy uint64:", end - start)