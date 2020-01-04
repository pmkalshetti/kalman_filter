import numpy as np
import matplotlib.pyplot as plt


class PositionSensor:
    def __init__(self, pos=(0, 0), vel=(0, 0), std_noise=1):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.std_noise = float(std_noise)

    def read(self):
        """Updates position by one time step and returns new measurement."""
        self.pos += self.vel

        return self.pos + np.random.randn(2) * self.std_noise


if __name__ == "__main__":
    pos = (0, 0)
    vel = (2, .2)
    sensor = PositionSensor(pos, vel)

    n_measurements = 30
    measurements = np.array([sensor.read() for _ in range(n_measurements)])
    measurements *= 0.3048  # feet to meters
    plt.scatter(
        measurements[:, 0], measurements[:, 1],
        label="measurements", facecolors="none", edgecolors="black"
    )
    plt.ylim(-3, 4)
    plt.xlabel("X (in mm)")
    plt.ylabel("Y (in mm)")
    plt.legend()
    plt.show()
