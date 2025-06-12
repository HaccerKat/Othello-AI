import numpy as np
a = np.zeros(3)
print(a)
print(str(a))
with open('test.txt', 'a') as f:
    f.write(str([1]))
