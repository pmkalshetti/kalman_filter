import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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


def design_filter(x0=0, y0=0, dt=1.,
                  var_process=0.001, var_measurement=1., var_state=500.):
    dim_state = 4  # [x, vx, y, vy]
    dim_measurement = 2  # [x, y]

    # A
    # constant velocity model
    mat_transition = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])

    # Q
    # noise is constant for each time period
    cov_process = np.eye(dim_state) * var_process

    # H
    factor_conversion = 1/.3048  # state in m, measurement is feet.
    mat_measurement = np.array([
        [factor_conversion, 0, 0, 0],
        [0, 0, factor_conversion, 0]
    ])

    # R
    # assume independent Gaussian for x and y
    cov_measurement = np.array([
        [var_measurement, 0],
        [0, var_measurement]
    ])

    # x
    # initial state
    state_init = np.reshape(np.array([x0, 0, y0, 0.]), (dim_state, 1))

    # P
    # initial state covariance
    cov_state_init = np.eye(dim_state) * var_state

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

        # handle occlusion
        if measurement[0] != -1:
            kalman_filter.correct(np.reshape(measurement, (2, 1)))

        state = np.copy(kalman_filter.statePost)[:, 0]
        x, y = state[0], state[2]
        list_state.append((x, y))

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
    # read measurements
    path_measurements = "trajectory.txt"
    measurements = np.loadtxt(path_measurements)

    # design kalman filter
    x_init, y_init = 0., 1.
    velocity_init = 50.
    angle_deg_init = 60
    var_process = 0.1  # Q
    var_measurement = 3.  # R
    var_state = 1.  # P
    dt = 1/20  # assuming fps of video
    design = design_filter(
        x_init, y_init, velocity_init, angle_deg_init, dt,
        var_process, var_measurement, var_state
    )
    kalman_filter = construct_filter(*design)

    # run filter
    arr_state, arr_cov = run_filter(kalman_filter, measurements)

    # plot
    fig, ax = plt.subplots()
    ax.scatter(
        measurements[:, 0], measurements[:, 1],
        label="measurements", facecolors="none", edgecolors="black"
    )
    plt.plot(
        arr_state[:, 0], arr_state[:, 1],
        label="filtered", lw=2, c="red"
    )
    plot_states(arr_state, arr_cov, ax)
    plt.ylim(-3, 4)

    plt.xlabel("X (in m)")
    plt.ylabel("Y (in m)")
    plt.legend()
    plt.show()
