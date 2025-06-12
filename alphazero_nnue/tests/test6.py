import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

alpha1 = 0.01
alpha2 = 0.01

x = np.linspace(0, 1, 500)
y = beta.pdf(x, alpha1, alpha2)

plt.plot(x, y)
plt.title(f"Beta Distribution (α={alpha1}, {alpha2})")
plt.xlabel("x (prob of action 1)")
plt.ylabel("Density")
plt.grid(True)
plt.show()
