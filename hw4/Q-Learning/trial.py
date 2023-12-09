import pickle
import numpy as np

eps=pickle.load(open("eps.pkl", "rb"))

import matplotlib.pyplot as plt

plt.xlabel("Number of episodes")
plt.ylabel("Log of total actions taken")
plt.plot(range(1, 10001), np.log(eps))
plt.savefig("fig1.png")
plt.clf()

plt.xlabel("Number of episodes")
plt.ylabel("Total actions taken")
plt.plot(range(1, 10001), eps)
plt.savefig("fig2.png")

