import json
from matplotlib import pyplot as plt

unstable_file = open("thetas.unstable.force_limit.json")
half_unstable_file = open("thetas.half_unstable.force_limit.json")
stable_file = open("thetas.stable.force_limit.json")

unstable_thetas = json.load(unstable_file)
half_unstable_thetas = json.load(half_unstable_file)
stable_thetas = json.load(stable_file)

plt.figure()
plt.semilogy(unstable_thetas)
plt.semilogy(half_unstable_thetas)
plt.semilogy(stable_thetas)
plt.xlabel('Iteration')
plt.ylabel('Theta [rad]')
plt.title('Pole angle over time (with force limit)')
plt.legend([r'$0.12\pi$', '$0.06\pi$', r'$\frac{\pi} {10}$'])
plt.show()
