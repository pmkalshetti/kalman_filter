# Kalman Filter

The aim is to track an object based on noisy measurements in 2D. This is achieved using a Kalman Filter.

## 1. Constant Velocity Model
The sensor reads 2D position of the object. We assume a fixed velocity for the object. The final reading is perturbed with Gaussian noise.
```python
python src/constant_velocity_model/filter.py
```
![Constant Velocity Model](media_readme/constant_velocity_model.png)

The ellipse indicates 95% confidence interval for the covariance matrix between x and y.


## 2. Constant Acceleration Model
A ball is thrown in vacuum travelling in a parabola under constant gravitational field. The measurements contain position of ball in 2D with Gaussian noise. Here we use the acceleration as a control input in the Kalman Filter.
```python
python src/constant_acceleration_model/filter.py
```
![Constant Acceleration Model](media_readme/constant_acceleration_model.png)


## 3. Track Ball under Occlusion
A ball travels on the ground with approximately constant velocity. The ball is detected using background subtraction method which gives the noisy measurements. There is an instant in the video when the ball is occluded. Kalman filter is used to track this ball even under occlusion.


# References
1. [Kalman and Bayesian Filters in Python by Roger R. Labbe](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
2. [OpenCV Documentation](https://docs.opencv.org/trunk/dd/d6a/classcv_1_1KalmanFilter.html)
