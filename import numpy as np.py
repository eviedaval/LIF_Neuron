import numpy as np
import matplotlib.pyplot as plt

def lif_neuron(I, R=1, C=1, V_rest=-70, V_thresh=-55, V_reset=-80, dt=0.1, timesteps=1000):
    """
    Simulate a Leaky Integrate and Fire (LIF) neuron.

    Parameters:
    - I: Input current (constant or array)
    - R: Membrane resistance
    - C: Membrane capacitance
    - V_rest: Resting membrane potential
    - V_thresh: Threshold potential for firing
    - V_reset: Reset potential after firing
    - dt: Time step for numerical integration
    - timesteps: Number of simulation time steps

    Returns:
    - V: Membrane potential over time
    - spikes: Spike times
    """

    V = np.zeros(timesteps)
    spikes = []

    for t in range(1, timesteps):
        dV = (I[t - 1] - (V[t - 1] - V_rest) / R) / C * dt
        V[t] = V[t - 1] + dV

        if V[t] >= V_thresh:
            V[t] = V_reset
            spikes.append(t)

    return V, spikes

# Input current - a step input for illustration purposes
I_step = np.concatenate([np.zeros(500), np.ones(500) * 10])

# Simulate LIF neuron with step input
V_step, spikes_step = lif_neuron(I_step)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(I_step, label='Input Current')
plt.title('Input Current')
plt.xlabel('Time Steps')
plt.ylabel('Current')

plt.subplot(2, 1, 2)
plt.plot(V_step, label='Membrane Potential')
plt.scatter(spikes_step, [V_step[i] for i in spikes_step], c='red', label='Spikes', marker='o')
plt.axhline(y=-55, color='r', linestyle='--', label='Threshold')
plt.title('Leaky Integrate and Fire Neuron')
plt.xlabel('Time Steps')
plt.ylabel('Membrane Potential')

plt.legend()
plt.tight_layout()
plt.show()
