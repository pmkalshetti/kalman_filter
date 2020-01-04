import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from data import PositionSensor


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


def design_filter():
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


def get_ellipse_from_covariance(mat_cov, val_chisquare=2.4477):
    """95% confidence interval => 2.4477 chisquare value."""
    # calculate largest and smallest eigenvalues and eigenvectors
    val_eig, vec_eig = np.linalg.eig(mat_cov)
    idx_max, idx_min = np.argmax(val_eig), np.argmin(val_eig)
    val_eig_max, val_eig_min = val_eig[idx_max], val_eig[idx_min]
    vec_eig_max = vec_eig[idx_max]

    # angle between X-axis and largest eigenvector
    angle = np.arctan2(vec_eig_max[1], vec_eig_max[0])
    if angle < 0:  # [-pi, pi] --> [0, 2pi]
        angle += 2*np.pi

    # ellipse points
    a = val_chisquare * np.sqrt(val_eig_max)
    b = val_chisquare * np.sqrt(val_eig_min)

    points_along_curve = np.linspace(0, 2*np.pi)
    x_ellipse = a * np.cos(points_along_curve)
    y_ellipse = b * np.sin(points_along_curve)
    points_ellipse_2_N = np.stack([x_ellipse, y_ellipse], axis=0)

    # rotate ellipse
    mat_rotation = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    points_ellipse_2_N = mat_rotation @ points_ellipse_2_N
    points_ellipse = np.transpose(points_ellipse_2_N)

    return points_ellipse


def plot_states(means, covariances, ax):
    for mean, cov in zip(means, covariances):
        points_ellipse = get_ellipse_from_covariance(cov)
        ax.plot(
            points_ellipse[:, 0]+mean[0], points_ellipse[:, 1]+mean[1],
            c="g"
        )


def run_filter(kalman_filter, measurements):
    """Runs kalman filter and returns states and covariances."""
    list_state, list_cov = [], []
    for measurement in measurements:
        kalman_filter.predict()
        kalman_filter.correct(np.reshape(measurement, (2, 1)))

        list_state.append(np.copy(kalman_filter.statePost)[:, 0])

        cov_state = np.copy(kalman_filter.errorCovPost)

        # store covariance between x and y
        cov_x_y = np.array([
            [cov_state[0, 0], cov_state[2, 0]],
            [cov_state[0, 2], cov_state[2, 2]]
        ])
        list_cov.append(cov_x_y)
    arr_state = np.array(list_state)
    arr_cov = np.array(list_cov)

    return arr_state, arr_cov


if __name__ == "__main__":
    # design kalman filter
    design = design_filter()
    kalman_filter = construct_filter(*design)

    # simulate measurements
    n_measurements = 30
    pos_true = (0, 0)
    vel_true = (2, .2)
    R_std = 0.35
    sensor = PositionSensor(pos_true, vel_true, R_std)
    measurements = np.array([sensor.read() for _ in range(n_measurements)])

    # run filter
    arr_state, arr_cov = run_filter(kalman_filter, measurements)

    # plot
    fig, ax = plt.subplots()
    measurements *= 0.3048  # feet to meters
    ax.scatter(
        measurements[:, 0], measurements[:, 1],
        label="measurements", facecolors="none", edgecolors="black"
    )
    plt.plot(
        arr_state[:, 0], arr_state[:, 2],
        label="filtered", lw=2, c="red"
    )
    plot_states(arr_state[:, ::2], arr_cov, ax)
    plt.ylim(-3, 4)

    plt.xlabel("X (in m)")
    plt.ylabel("Y (in m)")
    plt.legend()
    plt.show()
