import numpy as np
import matplotlib.pyplot as plt


def drag_force(velocity):
    B_m = 0.0039 + 0.0058 / (1. + np.exp((velocity-35)/5))
    return B_m * velocity


class BallInAir:
    def __init__(self, x0, y0, velocity, angle_deg, noise=[1., 1.]):
        self.x = x0
        self.y = y0

        angle = np.deg2rad(angle_deg)
        self.vel_x = velocity * np.cos(angle)
        self.vel_y = velocity * np.sin(angle)

        self.noise = noise

    def step(self, dt, vel_wind=0.):
        # air drag
        vel_x_wind = self.vel_x - vel_wind
        vel = np.sqrt(vel_x_wind**2 + self.vel_y**2)
        F = drag_force(vel)

        # euler equations
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self.vel_x -= F*vel_x_wind*dt
        self.vel_y -= 9.81*dt + F*self.vel_y*dt

        return (self.x + np.random.randn()*self.noise[0],
                self.y + np.random.randn()*self.noise[1])


def generate_measurements(x=0, y=1, velocity=50, angle_deg=60, dt=1/10,
                          noise=[.3, .3]):
    """x, y: m. velocity: m/s. angle_deg: degree. dt: s. noise: m"""
    ball = BallInAir(x, y, velocity, angle_deg, noise)

    measurements = []
    while y >= 0:  # until ball falls on ground
        x, y = ball.step(dt)
        measurements.append((x, y))
    measurements = np.array(measurements)

    return measurements


if __name__ == "__main__":
    x, y = 0, 1  # m
    velocity = 50  # m/s
    angle_deg = 60  # degree
    dt = 1/10  # s
    noise = [.3, .3]

    measurements = generate_measurements(x, y, velocity, angle_deg, dt, noise)

    plt.scatter(measurements[:, 0], measurements[:, 1])
    plt.axis("equal")
    plt.show()
