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
#pragma once

#include <cstdint>
#include <map>

#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/generic_camera.hpp>
#include <glog/logging.h>

#include "rootba/bal/common_types.hpp"
#include "rootba/util/cast.hpp"

namespace rootba {

struct BalDatasetOptions;
struct DatasetSummary;
struct PipelineTimingSummary;

// BAL problem as loaded from the dataset files: Cameras are represented
// as world-to-cam, and landmarks in world coordinates.
//
// For the camera frame we assume the positive z axis pointing forward in view
// direction and in the image, y is poiting down, x to the right. In the
// original BAL formulation, the camera points in negative z axis, y is up in
// the image. Thus when loading the data, we invert the y and z camera axes (y
// also in the image) in the perspective projection, we don't have the "minus"
// like in the original Snavely model.
template <typename Scalar>
class BalProblem {
 public:
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using SE3 = Sophus::SE3<Scalar>;
  using SO3 = Sophus::SO3<Scalar>;

  struct Observation {
    Vec2 pos = Vec2::Zero();
  };

  struct Landmark {
    Vec3 p_w;                       // point position in world coordinates
    Eigen::Vector3<uint8_t> color;  // RGB color
    std::map<TimeCamId, Observation> obs;  // list of frame indices

    void backup() { p_w_backup_ = p_w; }

    void restore() { p_w = p_w_backup_; }

    template <typename Scalar2>
    typename BalProblem<Scalar2>::Landmark cast() const {
      typename BalProblem<Scalar2>::Landmark res;
      res.p_w = p_w.template cast<Scalar2>();
      for (const auto& [frame_id, o] : obs) {
        res.obs[frame_id].pos = o.pos.template cast<Scalar2>();
      }

      return res;
    }

   private:
    Vec3 p_w_backup_;
  };

  // Keyframe represents a rig (set of cameras with fixed relative transforms)
  struct Keyframe {
    // TODO@tsantucci: rename to something like Rig or Pose
    SE3 T_i_w;
    size_t t_ns;

    void backup() { T_i_w_backup_ = T_i_w; }

    void restore() { T_i_w = T_i_w_backup_; }

    void apply_inc_pose(const Vec6& inc) {
      T_i_w = Sophus::se3_expd(inc) * T_i_w;
    }

    template <typename Scalar2>
    typename BalProblem<Scalar2>::Keyframe cast() const {
      typename BalProblem<Scalar2>::Keyframe res;
      res.T_i_w = T_i_w.template cast<Scalar2>();
      res.t_ns = t_ns;
      return res;
    }

   private:
    SE3 T_i_w_backup_;
  };

  using Landmarks = std::vector<Landmark>;
  using Keyframes = std::vector<Keyframe>;
  using Calibration = basalt::Calibration<Scalar>;

  BalProblem() = default;
  explicit BalProblem(const std::string& path);

  void load_basalt(const std::string& path);

  bool save_basalt(const std::string& path);

  bool save_euroc(const std::string& path) const;

  void load_calibration(const std::string& path);

  void add_noise(const double obs_noise_sigma);

  void normalize(double new_scale);

  void perturb(double rotation_sigma, double translation_sigma,
               double landmark_sigma, int seed = -1);

  void filter_obs(double threshold);

  void filter_kf(int min_obs_per_kf);

  void postprocress(const BalDatasetOptions& options,
                    PipelineTimingSummary* timing_summary = nullptr);

  void backup();
  void restore();

  inline const Landmarks& landmarks() const { return landmarks_; }
  inline const Keyframes& keyframes() const { return keyframes_; }
  inline const Calibration& calib() const { return calib_; }

  inline Landmarks& landmarks() { return landmarks_; }
  inline Keyframes& keyframes() { return keyframes_; }

  inline int num_landmarks() const { return signed_cast(landmarks_.size()); }
  inline int num_keyframes() const { return signed_cast(keyframes_.size()); }
  int num_observations() const;
  int max_num_observations_per_lm() const;
  double compute_rcs_sparsity() const;

  bool quiet() const { return quiet_; }
  void set_quiet(const bool quiet) { quiet_ = quiet; }

  template <typename Scalar2>
  BalProblem<Scalar2> copy_cast() const {
    BalProblem<Scalar2> res;

    res.keyframes_.resize(this->keyframes_.size());
    res.landmarks_.resize(this->landmarks_.size());

    for (size_t i = 0; i < this->keyframes_.size(); i++) {
      res.keyframes_[i] = this->keyframes_[i].template cast<Scalar2>();
    }

    for (size_t i = 0; i < this->landmarks_.size(); i++) {
      res.landmarks_[i] = this->landmarks_[i].template cast<Scalar2>();
    }

    res.calib_ = this->calib_.template cast<Scalar2>();

    res.quiet_ = quiet_;

    return res;
  }

  void summarize_problem(DatasetSummary& summary, bool compute_sparsity) const;

  std::string stats_to_string() const;

 private:
  template <typename T>
  friend class BalProblem;

  Landmarks landmarks_;
  Keyframes keyframes_;

  Calibration calib_;

  /// quiet means no INFO level log output
  bool quiet_ = false;
};

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary = nullptr,
    PipelineTimingSummary* timing_summary = nullptr);

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const std::string& path, DatasetSummary* dataset_summary = nullptr,
    PipelineTimingSummary* timing_summary = nullptr);

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem_quiet(const std::string& path);

}  // namespace rootba
