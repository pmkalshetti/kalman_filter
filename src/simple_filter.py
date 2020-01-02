import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sensor import Sensor
from matplotlib.patches import Ellipse


def construct_filter(dim_state, dim_measurement,
                     mat_transition, cov_process,
                     mat_measurement, cov_measurement,
                     state_init, cov_state_init):
    kf = cv.KalmanFilter(dim_state, dim_measurement)

    kf.transitionMatrix = mat_transition
    kf.processNoiseCov = cov_process

    kf.measurementMatrix = mat_measurement
    kf.measurementNoiseCov = cov_measurement

    kf.statePost = state_init
    kf.errorCovPost = cov_state_init

    return kf


def design_simple_filter():
    dim_state = 4  # [x, vx, y, vy]
    dim_measurement = 2  # [x, y]

    # A
    # constant velocity model
    dt = 1.  # time step is 1s
    mat_transition = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

    # Q
    # noise is constant for each time period
    # var_process = 0.04 ** 2
    cov_process = np.array([
        [0, 0.001, 0, 0],
        [0.001, 0.002, 0, 0],
        [0, 0, 0, 0.001],
        [0, 0, 0.001, 0.002]
    ])

    # H
    factor_conversion = 1/.3048  # state in m, measurement is feet.
    mat_measurement = np.array([
        [factor_conversion, 0, 0, 0],
        [0, 0, factor_conversion, 0]
    ])

    # R
    # assume independent Gaussian for x and y
    var_measurement = 0.35**2  # feet^2
    cov_measurement = np.array([
        [var_measurement, 0],
        [0, var_measurement]
    ])

    # x
    # initial state
    state_init = np.reshape(np.array([0, 0, 0, 0.]), (dim_state, 1))

    # P
    # initial state covariance
    var_state_init = 500.
    cov_state_init = np.eye(dim_state) * var_state_init

    return (dim_state, dim_measurement,
            mat_transition, cov_process,
            mat_measurement, cov_measurement,
            state_init, cov_state_init)


def plot_state(means, covariances, ax):
    for mean, cov in zip(means, covariances):
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            mean,
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor="none", edgecolor="blue", alpha=0.7
        )
        ax.add_patch(ellipse)


if __name__ == "__main__":
    # design kalman filter
    design = design_simple_filter()
    kalman_filter = construct_filter(*design)

    # simulate measurements
    n_measurements = 30
    pos_true = (0, 0)
    vel_true = (2, .2)
    R_std = 0.35
    sensor = Sensor(pos_true, vel_true, R_std)
    measurements = np.array([sensor.read() for _ in range(n_measurements)])

    # run filter
    list_state, list_cov = [], []
    for measurement in measurements:
        kalman_filter.predict()
        kalman_filter.correct(np.reshape(measurement, (2, 1)))

        list_state.append(np.copy(kalman_filter.statePost)[:, 0])
        list_cov.append(np.copy(kalman_filter.errorCovPost))
    arr_state = np.array(list_state)
    arr_cov = np.array(list_cov)

    fig, ax = plt.subplots()
    plot_state(arr_state[:, ::2], arr_cov, ax)

    ax.scatter(
        measurements[:, 0], measurements[:, 1],
        label="measurements", facecolors="none", edgecolors="black"
    )
    plt.scatter(
        arr_state[:, 0], arr_state[:, 2],
        label="filtered"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
