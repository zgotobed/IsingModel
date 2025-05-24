import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def metropolis_ising_time_series(num_steps, size, beta):
    spins = np.random.choice([-1, 1], size=(size, size))
    magnetization = np.zeros(num_steps)

    for step in range(num_steps):
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)

        s = spins[i, j]
        neighbors = spins[(i+1)%size, j] + spins[(i-1)%size, j] + spins[i, (j+1)%size] + spins[i, (j-1)%size]
        dE = 2 * s * neighbors

        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1

        # Store average spin per site
        magnetization[step] = np.sum(spins) / (size**2)

    return magnetization


# Parameters
betas = np.linspace(0.1,1,5)
#betas = np.array([100]) #Debug
num_steps = 100000
size = 50
magnetization_over_time = []

# Run simulation for each beta
for beta in betas:
    print(f"Simulating for β = {beta:.2f}")
    m_time_series = metropolis_ising_time_series(num_steps, size, beta)
    magnetization_over_time.append(m_time_series)

# Plotting
plt.figure(figsize=(10, 6))
for i, beta in enumerate(betas):
    plt.plot(magnetization_over_time[i], label=f'β = {beta:.2f}')

plt.xlabel("Algorithm Steps")
plt.ylabel("Average Spin per Site")
plt.title("Ising Model Magnetization vs Steps for Different beta")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
