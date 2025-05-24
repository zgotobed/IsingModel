import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def metropolis_ising(num_steps, size, beta):
    spins = np.random.choice([-1, 1], size=(size, size))
    net_spins = np.zeros(num_steps)

    for step in tqdm(range(num_steps), desc="Running Metropolis"):
        # Pick a random spin
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)

        s = spins[i, j]
        neighbors = spins[(i+1)%size, j] + spins[(i-1)%size, j] + spins[i, (j+1)%size] + spins[i, (j-1)%size]
        dE = 2 * s * neighbors

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1  # Accept the flip

        net_spins[step] = np.sum(spins)

    return net_spins


# Run and plot
betas = np.linspace(0.1, 1.0, 10)
num_steps = 100000
size = 50
spins_over_time = np.zeros((len(betas), num_steps))

for idx, beta in enumerate(betas):
    print(f"Running for beta = {beta:.2f}")
    spins_over_time[idx] = metropolis_ising(num_steps, size, beta)

# Plot
for i in range(len(betas)):
    plt.plot(spins_over_time[i] / size**2, label=f"β = {betas[i]:.2f}")

plt.xlabel("Steps")
plt.ylabel("Average Spin")
plt.title("Ising Model Magnetization vs Time")
plt.legend()
plt.grid()
plt.show()
