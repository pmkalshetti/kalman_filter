import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def generate_measurements(path_video="src/track_ball/singleball.mov",
                          flag_save=False, flag_show=False):
    capture = cv.VideoCapture(path_video)
    n_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    height_frame = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    width_frame = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))

    # fig, ax = plt.subplots()

    # measurements using background subtraction
    back_sub = cv.createBackgroundSubtractorMOG2()
    measurements = np.zeros((n_frames, 2)) - 1  # init
    idx_frame = -1
    while capture.isOpened():
        idx_frame += 1
        ret, frame = capture.read()
        if frame is None:
            break

        if flag_show:
            cv.imshow("video", frame)
            cv.waitKey(20)
        mask_fg = back_sub.apply(frame)
        ret, img_binary = cv.threshold(
            mask_fg, thresh=220, maxval=255, type=cv.THRESH_BINARY
        )
        contours, hierarchy = cv.findContours(
            img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            for contour in contours:
                area = cv.contourArea(contour)
                if area > 100:
                    m = np.mean(contour, axis=0)
                    measurements[idx_frame] = m[0]

                    if flag_show:
                        cv.imshow("Foreground", mask_fg)
                        # cv.imwrite(f"mask/{idx_frame:02d}.png", mask_fg)
                        # cv.imwrite(f"input/{idx_frame:02d}.png", frame)

                        # ax.clear()
                        # ax.scatter(
                            # m[0][0], m[0][1],
                            # label="measurements", facecolors="none", edgecolors="black"
                        # )
                        # ax.set_xlim(0, width_frame)
                        # ax.set_ylim(height_frame, 0)
                        # ax.legend()
                        # plt.axis("off")
                        # fig.savefig(f"measurements/{idx_frame:02d}.png")

                        cv.waitKey(160)
    capture.release()
    cv.destroyAllWindows()

    if flag_save:
        np.savetxt(
            "src/track_ball/trajectory_ball.txt", measurements, fmt="%.3f"
        )

    if flag_show:
        fig, ax = plt.subplots()
        ax.set_xlim(0, width_frame)
        ax.set_ylim(height_frame, 0)
        ax.plot(measurements[:, 0], measurements[:, 1], "xr")
        plt.show()

    return measurements


if __name__ == "__main__":
    measurements = generate_measurements(flag_save=True, flag_show=True)
