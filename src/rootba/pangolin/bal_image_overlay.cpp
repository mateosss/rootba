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

#include "rootba/pangolin/bal_image_overlay.hpp"

#include <cmath>

#include <pangolin/display/default_font.h>
#include <pangolin/gl/gldraw.h>

namespace rootba {

BalImageOverlay::BalImageOverlay() = default;

template <typename Scalar>
void BalImageOverlay::update(pangolin::ImageView& view,
                             const BalProblem<Scalar>& bal_problem,
                             FrameIdx frame_id) {
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  frame_id_ = frame_id;

  // prepare storage
  const auto& lmdb = bal_problem.landmarks();
  const auto& target_kf = bal_problem.keyframes().at(frame_id);
  const auto& calib = bal_problem.calib();

  kpts_detected_.clear();
  kpts_projected_.clear();
  image_sizes_.clear();
  centers_.clear();

  // Iterate over all cameras in the calibration
  for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); ++cam_id) {
    TimeCamId tcid(frame_id, cam_id);
    const auto& target_cam = calib.intrinsics.at(cam_id);
    const Sophus::SE3<Scalar>& T_c_i = calib.T_i_c.at(cam_id).inverse();

    Sophus::SE3<Scalar> T_c_w = T_c_i * target_kf.T_i_w;

    Vec2d image_size = Vec2d(1, 1);
    std::vector<Vec2d> detected;
    std::vector<Vec2d> projected;

    // compute projections and store observations for this camera
    for (const auto& lm : lmdb) {
      auto obs_it = lm.obs.find(tcid);
      if (obs_it == lm.obs.end()) {
        continue;
      }

      Vec2 obs_pos = obs_it->second.pos;

      if (target_cam.getName() != "bal") {
        obs_pos[0] -= target_cam.getParam()[2];
        obs_pos[1] -= target_cam.getParam()[3];
      }

      detected.emplace_back(obs_pos.template cast<double>());

      // projected position
      Vec4 p_cam = (T_c_w * lm.p_w).homogeneous();
      Vec2 p_proj;
      target_cam.project(p_cam, p_proj);

      if (target_cam.getName() != "bal") {
        p_proj[0] -= target_cam.getParam()[2];
        p_proj[1] -= target_cam.getParam()[3];
      }
      projected.emplace_back(p_proj.template cast<double>());

      image_size =
          image_size.cwiseMax(obs_pos.template cast<double>().cwiseAbs() * 2);
    }

    // Store per-camera data
    if (!detected.empty()) {
      kpts_detected_[cam_id] = std::move(detected);
      kpts_projected_[cam_id] = std::move(projected);
      image_sizes_[cam_id] = image_size;
    }
  }

  // Calculate scale factor based on largest camera image
  Vec2d max_image_size(1, 1);
  for (const auto& [cam_id, img_size] : image_sizes_) {
    max_image_size = max_image_size.cwiseMax(img_size);
  }

  if (max_image_size.minCoeff() < options_.min_image_size) {
    scale_factor_ = options_.min_image_size / max_image_size.minCoeff();
  }
  if (max_image_size.maxCoeff() * scale_factor_ > options_.max_image_size) {
    scale_factor_ = options_.max_image_size / max_image_size.maxCoeff();
  }

  // Calculate grid layout: arrange cameras in a grid
  size_t num_cameras = image_sizes_.size();
  if (num_cameras == 0) return;

  size_t grid_cols = static_cast<size_t>(
      std::ceil(std::sqrt(static_cast<double>(num_cameras))));
  size_t grid_rows = static_cast<size_t>(
      std::ceil(static_cast<double>(num_cameras) / grid_cols));

  // Calculate individual camera view size (with border)
  Vec2d camera_view_size = max_image_size * scale_factor_;
  Vec2d background_size = camera_view_size.array() + 20;  // border offset

  // Calculate composite image size
  double composite_width =
      background_size.x() * grid_cols + 10 * (grid_cols + 1);
  double composite_height =
      background_size.y() * grid_rows + 10 * (grid_rows + 1);

  // Create composite image
  pangolin::ManagedImage<uint8_t> composite_img(composite_width,
                                                composite_height);
  composite_img.Fill(200);  // light background

  // Create individual camera views and compute centers
  size_t cam_idx = 0;
  for (const auto& [cam_id, img_size] : image_sizes_) {
    size_t row = cam_idx / grid_cols;
    size_t col = cam_idx % grid_cols;

    // Calculate position of this camera in the composite
    double x_offset = 10 + col * (background_size.x() + 10);
    double y_offset = 10 + row * (background_size.y() + 10);

    // Draw camera background
    auto cam_bg = composite_img.SubImage(
        x_offset, y_offset, background_size.x(), background_size.y());
    cam_bg.Fill(220);  // light grey border

    // Draw inner image area
    Vec2d cam_size = img_size * scale_factor_;
    Vec2d inner_offset = (background_size - cam_size) / 2;
    auto inner_img = composite_img.SubImage(x_offset + inner_offset.x(),
                                            y_offset + inner_offset.y(),
                                            cam_size.x(), cam_size.y());
    inner_img.Fill(150);  // darker grey

    // Store center position for this camera (in composite coordinates)
    centers_[cam_id] = Vec2d(x_offset + background_size.x() / 2,
                             y_offset + background_size.y() / 2);

    cam_idx++;
  }

  // Set the composite image to the view
  view.SetImage(composite_img);
}

pangolin::ManagedImage<uint8_t> BalImageOverlay::make_background_image(
    Vec2d image_size) {
  image_size *= scale_factor_;

  Vec2d background_size = image_size.array() + 20;  // border offset
  center_ = background_size / 2;

  // create image
  pangolin::ManagedImage<uint8_t> img(background_size.x(), background_size.y());

  // fill whole image in light grey
  img.Fill(220);

  // fill subimage with darker grey
  Vec2d offset = (background_size - image_size) / 2;
  auto subimg =
      img.SubImage(offset.x(), offset.y(), image_size.x(), image_size.y());
  subimg.Fill(150);

  return img;
}

void BalImageOverlay::set_options(const Options& options) {
  options_ = options;
}

void BalImageOverlay::draw() const {
  glLineWidth(1.0);
  glEnable(GL_BLEND);

  // Iterate over all cameras and draw their keypoints in their respective views
  size_t total_keypoints = 0;
  for (const auto& [cam_id, detected] : kpts_detected_) {
    const auto& projected = kpts_projected_.at(cam_id);
    const Vec2d& center = centers_.at(cam_id);

    // draw lines (red)
    glColor3f(1.0, 0.0, 0.0);
    for (size_t i = 0; i < detected.size(); ++i) {
      Eigen::Vector2d p1 = detected.at(i) * scale_factor_ + center;
      Eigen::Vector2d p2 = projected.at(i) * scale_factor_ + center;
      pangolin::glDrawLine(p1(0), p1(1), p2(0), p2(1));
    }

    // draw projected (red)
    glColor3f(1.0, 0.0, 0.0);
    for (const auto& pos : projected) {
      Eigen::Vector2d p = pos * scale_factor_ + center;
      pangolin::glDrawCirclePerimeter(p(0), p(1), options_.circle_radius);
    }

    // draw detected (blue)
    glColor3f(0.0, 0.0, 1.0);
    for (const auto& pos : detected) {
      Eigen::Vector2d p = pos * scale_factor_ + center;
      pangolin::glDrawCirclePerimeter(p(0), p(1), options_.circle_radius);
    }

    // Draw camera label
    glColor3f(1.0, 1.0, 1.0);  // white
    Vec2d label_pos = center - Vec2d(0, center.y() - 5);
    pangolin::default_font()
        .Text("Cam %d: %d pts", cam_id, detected.size())
        .Draw(label_pos.x() - 40, label_pos.y());

    total_keypoints += detected.size();
  }

  // Info text
  glColor3f(1.0, 0.0, 0.0);  // red
  pangolin::default_font()
      .Text("Total: %d keypoints (%d cameras)", total_keypoints,
            kpts_detected_.size())
      .Draw(12, 20);
}

template void BalImageOverlay::update(pangolin::ImageView& view,
                                      const BalProblem<double>& bal_problem,
                                      FrameIdx frame_id);

template void BalImageOverlay::update(pangolin::ImageView& view,
                                      const BalProblem<float>& bal_problem,
                                      FrameIdx frame_id);

}  // namespace rootba
