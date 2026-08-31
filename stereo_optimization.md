## Intro

This project has been adapted to preserve the stereo constraints of a multicamera setup.

Only square root optimization is implemented.

To support this, the Keyframe structure was added to represent the set of frames corresponding to the same timestamp. It contains the IMU pose (T_i_w). Using the Calibration object, attribute of `BalProblem`, one can access the transformation from the IMU to the corresponding cameras (T_c_i). Since the problem state contains only keyframe poses, we must compute errors, residuals, and Jacobians with respect to the IMU pose.

Additionally, intrinsics are no longer optimized and must be provided using a calibration file with the `--calibration-file` argument.

Because we now optimize IMU poses, the structure of a landmark block has been modified. Previously, each camera could have only one observation for a landmark, meaning that in a given landmark block, the column of a camera had only two non-zero rows. Now, since the columns of J_p represent IMU poses, each column can have up to 2*n non-zero rows, where n is the number of cameras in the setup.

Another conceptual difference is that Jacobians were previously computed with respect to the camera pose (d_res_d_cam), but are now computed with respect to the IMU pose (d_res_d_imu). The derivation of this Jacobian can be found in the `linearize_landmark` method of `landmark_block_base.ipp`.

Note that since we no longer optimize intrinsics and use a fixed T_c_i, we cannot normalize the problem, which requires us to use the `--no-normalize` argument.

## How to run

Compile with:

```bash
./scripts/build-rootba.sh
```

A list of pre-exported maps is available in `data/stereo`, along with the calibration files used to generate them in `data/calib`. You can run it with:

```bash
./bin/bal_gui --input data/stereo/MOO01.json --solver-type SQUARE_ROOT --calibration-file data/calib/msdmo_calib.json --no-normalize
```
