import numpy as np
import matplotlib.pyplot as plt

thetas = np.linspace(0, 1, 101)
likelihoods = thetas**6 * (1 - thetas)**4
plt.plot(thetas, likelihoods)
plt.xlabel("θ");
plt.ylabel("L(θ)");
plt.axvline(x=0.6, color='k', linestyle='--')
plt.axis([0, 1, 0, 0.0012])
plt.show()