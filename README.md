# Kalman Filter

The aim is to track an object based on noisy measurements in 2D. This is achieved using a Kalman Filter.

## 1. Constant Velocity Model
The sensor reads 2D position of the object. We assume a fixed velocity for the object. The final reading is perturbed with Gaussian noise.
```python
python src/constant_velocity_model/filter.py
```
![Constant Velocity Model](media_readme/constant_velocity_model.png)

The ellipse indicates 95% confidence interval for the covariance matrix between x and y.
