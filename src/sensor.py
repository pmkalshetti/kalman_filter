import numpy as np
import matplotlib.pyplot as plt


class Sensor:
    def __init__(self, pos=(0, 0), vel=(0, 0), std_noise=1):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.std_noise = float(std_noise)

    def read(self):
        """Updates position by one time step and returns new measurement."""
        self.pos += self.vel

        return self.pos + np.random.randn(2) * self.std_noise


if __name__ == "__main__":
    pos = (4, 3)
    vel = (2, 1)
    sensor = Sensor(pos, vel)

    n_measurements = 50
    measurements = np.array([sensor.read() for _ in range(n_measurements)])
    plt.scatter(
        measurements[:, 0], measurements[:, 1],
        label="measurements", facecolors="none", edgecolors="black"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
