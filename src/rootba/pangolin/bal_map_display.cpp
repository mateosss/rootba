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

#include "rootba/pangolin/bal_map_display.hpp"

#include "rootba/pangolin/gui_helpers.hpp"
#include "rootba/util/cast.hpp"

namespace rootba {

namespace {

const u_int8_t G_COLOR_POINTS[3]{0, 0, 0};      // black
const u_int8_t G_COLOR_CAMERA[3]{250, 0, 0};    // red
const u_int8_t G_COLOR_SELECTED[3]{0, 250, 0};  // green

constexpr double G_CAM_SELECTED_SCALE = 1.5;

}  // namespace

template <typename Scalar>
void BalMapDisplay::update(const BalProblem<Scalar>& bal_problem) {
  int initial_size = signed_cast(frames_.size());

  if (initial_size < bal_problem.num_keyframes()) {
    // create additional displays
    frames_.reserve(bal_problem.num_keyframes());
    for (int i = initial_size; i < bal_problem.num_keyframes(); ++i) {
      frames_.push_back(std::make_unique<BalFrameDisplay>(i));
    }
  } else {
    // (possibly) remove displays
    frames_.resize(bal_problem.num_keyframes());
  }

  for (int i = 0; i < signed_cast(frames_.size()); ++i) {
    frames_.at(i)->update(bal_problem);
  }

  // update points
  const auto& lmids = bal_problem.landmarks();
  points_.clear();
  colors_.clear();
  points_.reserve(lmids.size());
  colors_.reserve(lmids.size());
  for (const auto& lmid : lmids) {
    points_.push_back(lmid.p_w.template cast<float>());
    colors_.emplace_back(lmid.color);
  }

  // update flag to ensure that buffers are updated on next draw
  need_buffer_update_ = true;
}

void BalMapDisplay::draw(FrameIdx selected_frame,
                         const BalMapDisplay::Options& options) {
  for (int i = 0; i < signed_cast(frames_.size()); ++i) {
    const bool is_selected = i == selected_frame;
    frames_.at(i)->draw(is_selected, options);
  }

  update_buffers();
  draw_pointcloud(false, options);
}

BalFrameDisplay::BalFrameDisplay(FrameIdx frame_id) : frame_id_(frame_id) {}

template <typename Scalar>
void BalFrameDisplay::update(const BalProblem<Scalar>& bal_problem) {
  // update pose
  T_w_i_ = bal_problem.keyframes()
               .at(frame_id_)
               .T_i_w.inverse()
               .template cast<double>();
}

void BalFrameDisplay::draw(bool selected,
                           const BalMapDisplay::Options& options) {
  draw_camera(selected, options);
}

bool BalMapDisplay::update_buffers() {
  // early abort if nothing changed
  if (!need_buffer_update_) {
    return false;
  }

  vertex_buffer_.Clear();
  vertex_buffer_.Update(points_);
  color_buffer_.Clear();
  color_buffer_.Update(colors_);

  need_buffer_update_ = false;
  return true;
}

void BalFrameDisplay::draw_camera(bool selected,
                                  const BalMapDisplay::Options& options) {
  if (selected) {
    render_camera(T_w_i_.matrix(), options.cam_weight, G_COLOR_SELECTED,
                  options.cam_size * G_CAM_SELECTED_SCALE);
  } else {
    render_camera(T_w_i_.matrix(), options.cam_weight, G_COLOR_CAMERA,
                  options.cam_size);
  }
}

void BalMapDisplay::draw_pointcloud(bool selected,
                                    const BalMapDisplay::Options& options) {
  UNUSED(selected);
  if (vertex_buffer_.size() == 0) {
    return;
  }

  CHECK(color_buffer_.size() == vertex_buffer_.size());

  glDisable(GL_LIGHTING);

  int point_size = options.point_size;

  glPointSize(point_size);

  vertex_buffer_.Bind();
  glVertexPointer(vertex_buffer_.count_per_element, vertex_buffer_.datatype, 0,
                  nullptr);

  color_buffer_.Bind();
  glColorPointer(color_buffer_.count_per_element, color_buffer_.datatype, 0,
                 nullptr);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDrawArrays(GL_POINTS, 0, vertex_buffer_.size());
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  vertex_buffer_.Unbind();
  color_buffer_.Unbind();
}

template void BalMapDisplay::update(const BalProblem<double>& bal_problem);

template void BalMapDisplay::update(const BalProblem<float>& bal_problem);

template void BalFrameDisplay::update(const BalProblem<double>& bal_problem);

template void BalFrameDisplay::update(const BalProblem<float>& bal_problem);

}  // namespace rootba
