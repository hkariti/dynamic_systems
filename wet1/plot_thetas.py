import json
from matplotlib import pyplot as plt

unstable_file = open("thetas.unstable.json")
half_unstable_file = open("thetas.half_unstable.json")
stable_file = open("thetas.stable.json")

unstable_thetas = json.load(unstable_file)
half_unstable_thetas = json.load(half_unstable_file)
stable_thetas = json.load(stable_file)

plt.figure()
plt.semilogy(unstable_thetas)
plt.semilogy(half_unstable_thetas)
plt.semilogy(stable_thetas)
plt.xlabel('Iteration')
plt.ylabel('Theta [rad]')
plt.title('Pole angle over time')
plt.legend(['$\\theta_{unstable}$', '$0.5 \\theta_{unstable}$', r'$\frac{\pi} {10}$'])
plt.show()
