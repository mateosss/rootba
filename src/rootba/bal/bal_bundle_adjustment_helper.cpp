/**
BSD 3-Clause License

This file is part of the RootBA project.
https://github.com/NikolausDemmel/rootba

Copyright (c) 2021-2023, Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#include "rootba/bal/bal_bundle_adjustment_helper.hpp"

#include <variant>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace rootba {

template <typename Scalar>
std::tuple<Scalar, Scalar>
BalBundleAdjustmentHelper<Scalar>::compute_error_weight(
    const BalResidualOptions& options, Scalar res_squared) {
  // TODO: create small class for computing weights and pre-compute huber
  // threshold squared

  // Note: Definition of cost is 0.5 ||r(x)||^2 to be in line with ceres

  switch (options.robust_norm) {
    case BalResidualOptions::RobustNorm::HUBER: {
      const Scalar thresh = options.huber_parameter;
      const Scalar huber_weight =
          res_squared < thresh * thresh ? 1.0 : thresh / std::sqrt(res_squared);
      const Scalar error =
          0.5 * (2 - huber_weight) * huber_weight * res_squared;
      return {error, huber_weight};
    }
    case BalResidualOptions::RobustNorm::NONE:
      return {0.5 * res_squared, 1.0};
    default:
      LOG(FATAL) << "unreachable";
  }
}

template <typename Scalar>
void BalBundleAdjustmentHelper<Scalar>::compute_error(
    const BalProblem<Scalar>& bal_problem, const SolverOptions& options,
    ResidualInfo& error) {
  const bool ignore_validity_check = !options.use_projection_validity_check();

  const auto& keyframes = bal_problem.keyframes();
  const auto& calib = bal_problem.calib();

  // body for parallel reduce
  auto body = [&](const tbb::blocked_range<int>& range,
                  ResidualInfoAccu error_accu) {
    for (int r = range.begin(); r != range.end(); ++r) {
      const int lm_id = r;
      const auto& lm = bal_problem.landmarks().at(lm_id);

      for (const auto& [tcid, obs] : lm.obs) {
        FrameIdx frame_idx = tcid.frame_id;
        CamId cam_id = tcid.cam_id;

        const auto& keyframe = keyframes.at(frame_idx);
        const auto& T_i_w = keyframe.T_i_w;

        const auto& T_c_i = calib.T_i_c[cam_id].inverse();
        const auto& cam_model = calib.intrinsics[cam_id];

        // Compute transformation from world to camera frame
        typename BalProblem<Scalar>::SE3 T_c_w = T_c_i * T_i_w;

        VecR res;
        const bool numerically_valid = linearize_point(
            obs.pos, lm.p_w, T_c_w, cam_model, ignore_validity_check, res);

        if (numerically_valid) {
          const Scalar res_squared = res.squaredNorm();
          const auto [weighted_error, weight] =
              compute_error_weight(options.residual, res_squared);
          error_accu.add(numerically_valid, numerically_valid,
                         std::sqrt(res_squared), weighted_error);
        }
      }
    }

    return error_accu;
  };

  // go over all host frames
  tbb::blocked_range<int> range(0, bal_problem.num_landmarks());
  ResidualInfoAccu error_accu = tbb::parallel_deterministic_reduce(
      range, ResidualInfoAccu(), body, ResidualInfoAccu::join);

  // output accumulated error
  error = error_accu.info;
}

template <typename Scalar>
bool BalBundleAdjustmentHelper<Scalar>::linearize_point(
    const Vec2& obs, const Vec3& lm_p_w, const SE3& T_c_w,
    const basalt::GenericCamera<Scalar>& intr, const bool ignore_validity_check,
    VecR& res, MatRP* d_res_d_xi, MatRL* d_res_d_l) {
  Mat4 T_c_w_mat = T_c_w.matrix();

  Vec4 p_c_3d = T_c_w_mat * lm_p_w.homogeneous();

  Mat24 d_res_d_p;
  bool projection_valid;
  if (d_res_d_xi || d_res_d_l) {
    projection_valid = intr.project(p_c_3d, res, &d_res_d_p);
  } else {
    projection_valid = intr.project(p_c_3d, res, nullptr);
  }
  res -= obs;

  projection_valid &= res.array().isFinite().all();

  if (!ignore_validity_check && !projection_valid) {
    return false;
  }

  if (d_res_d_xi) {
    Mat<Scalar, 4, POSE_SIZE> d_point_d_xi;
    d_point_d_xi.template topLeftCorner<3, 3>() = Mat3::Identity();
    d_point_d_xi.template topRightCorner<3, 3>() =
        -SO3::hat(p_c_3d.template head<3>());
    d_point_d_xi.row(3).setZero();
    *d_res_d_xi = d_res_d_p * d_point_d_xi;
  }

  if (d_res_d_l) {
    *d_res_d_l = d_res_d_p * T_c_w_mat.template topLeftCorner<4, 3>();
  }

  return projection_valid;
}

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class BalBundleAdjustmentHelper<float>;
#endif

// The helper in double is used by the ceres iteration callback, so always
// compile it; it should not be a big compilation overhead.
// #ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class BalBundleAdjustmentHelper<double>;
// #endif

}  // namespace rootba
