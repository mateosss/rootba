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

#include "rootba/bal/bal_problem.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <utility>

#include <absl/container/flat_hash_set.h>
#include <basalt/serialization/headers_serialization.h>
#include <cereal/archives/binary.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "rootba/bal/bal_dataset_options.hpp"
#include "rootba/bal/bal_pipeline_summary.hpp"
#include "rootba/bal/bal_problem_io.hpp"
#include "rootba/cg/block_sparse_matrix.hpp"
#include "rootba/util/format.hpp"
#include "rootba/util/stl_utils.hpp"
#include "rootba/util/time_utils.hpp"

namespace rootba {

namespace {  // helper

template <typename T>
void fscan_or_throw(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    throw std::runtime_error("");
  }
}

template <typename T, int N>
void fscan_or_throw(FILE* fptr, Eigen::Matrix<T, N, 1>& values) {
  for (int i = 0; i < values.size(); ++i) {
    fscan_or_throw(fptr, "%lf", values.data() + i);
  }
}

void readcommentline_or_throw(FILE* fptr) {
  char buffer[1000];
  bool comment_ok = false;
  while (fgets(buffer, 1000, fptr) != nullptr) {
    size_t len = strlen(buffer);

    if (len == 0) {
      throw std::runtime_error("empty line; expected comment...");
    }

    // first part of line, check # character
    if (!comment_ok) {
      if (buffer[0] == '#') {
        comment_ok = true;
      } else {
        throw std::runtime_error("non-comment line; expected comment...");
      }
    }

    // check if we reached eol
    if (buffer[len - 1] == '\n') {
      return;
    }
  }

  // fgets failed
  throw std::runtime_error("could not read comment line");
}

template <typename T, int N, class RandomEngine>
Eigen::Matrix<T, N, 1> perturbation(const T sigma, RandomEngine& eng) {
  std::normal_distribution<T> normal;
  Eigen::Matrix<T, N, 1> vec;
  vec.setZero();
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] += normal(eng) * sigma;
  }
  return vec;
}

template <typename T>
T median_destructive(std::vector<T>& data) {
  int n = data.size();
  auto mid_point = data.begin() + n / 2;
  std::nth_element(data.begin(), mid_point, data.end());
  return *mid_point;
}

BalDatasetOptions::DatasetType autodetect_input_type(const std::string& path) {
  using std::filesystem::is_directory;
  const std::string filename = std::filesystem::path(path).filename();

  if (ends_with(filename, ".cereal")) {
    return BalDatasetOptions::DatasetType::ROOTBA;
  } else if (std::string::npos != filename.find("bundle")) {
    return BalDatasetOptions::DatasetType::BUNDLER;
  } else if (is_directory(path) && is_directory(path + "/sparse")) {
    return BalDatasetOptions::DatasetType::COLMAP;
  } else {
    // default to BAL
    return BalDatasetOptions::DatasetType::BAL;
  }
}

class BalProblemSaver : public FileSaver<cereal::BinaryOutputArchive> {
 public:
  using Scalar = double;

  inline BalProblemSaver(std::string path,
                         const BalProblem<Scalar>& bal_problem)
      : FileSaver(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool save_impl(cereal::BinaryOutputArchive& archive) override {
    archive(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  const BalProblem<Scalar>& bal_problem_;
};

class BalProblemLoader : public FileLoader<cereal::BinaryInputArchive> {
 public:
  using Scalar = double;

  inline BalProblemLoader(std::string path, BalProblem<Scalar>& bal_problem)
      : FileLoader(BAL_PROBLEM_FILE_INFO, std::move(path)),
        bal_problem_(bal_problem) {}

 protected:
  inline bool load_impl() override {
    (*archive_)(bal_problem_);
    return true;
  }

  inline std::string format_summary() const override {
    return bal_problem_.stats_to_string();
  }

 private:
  BalProblem<Scalar>& bal_problem_;
};

}  // namespace

template <typename Scalar>
BalProblem<Scalar>::BalProblem(const std::string& path) {
  load_basalt(path);
}

template <typename Scalar>
void BalProblem<Scalar>::load_basalt(const std::string& path_str) {
  using Quaternion = Eigen::Quaternion<Scalar>;
  using std::getline;
  using std::ifstream;
  using std::istringstream;
  using std::string;
  using std::unordered_map;
  using std::filesystem::is_directory;
  using std::filesystem::path;

  // Read and parse JSON file
  ifstream file(path_str);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path_str);
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse JSON: " + string(e.what()));
  }

  // Clear existing data
  keyframes_.clear();
  landmarks_.clear();

  // Load keyframes - use map to handle arbitrary IDs
  unordered_map<size_t, size_t> kf_id_map;  // JSON id -> vector index
  if (j.contains("keyframes")) {
    const auto& kfs_json = j["keyframes"];
    keyframes_.resize(kfs_json.size());

    size_t idx = 0;
    for (const auto& kf_json : kfs_json) {
      size_t id = kf_json["id"];
      kf_id_map[id] = idx;

      const auto& T_w_i_data = kf_json["T_w_i"];

      // T_w_i is [qw, qx, qy, qz, tx, ty, tz]
      Quaternion q(static_cast<Scalar>(T_w_i_data[0]),
                   static_cast<Scalar>(T_w_i_data[1]),
                   static_cast<Scalar>(T_w_i_data[2]),
                   static_cast<Scalar>(T_w_i_data[3]));
      Vec3 t(static_cast<Scalar>(T_w_i_data[4]),
             static_cast<Scalar>(T_w_i_data[5]),
             static_cast<Scalar>(T_w_i_data[6]));

      keyframes_[idx].T_i_w = SE3(q, t).inverse();
      keyframes_[idx].t_ns = id;
      idx++;
    }
  }

  // Load landmarks - use map to handle arbitrary IDs
  unordered_map<size_t, size_t> lm_id_map;  // JSON id -> vector index
  if (j.contains("landmarks")) {
    const auto& lms_json = j["landmarks"];
    landmarks_.resize(lms_json.size());

    size_t idx = 0;
    for (const auto& lm_json : lms_json) {
      size_t id = lm_json["id"];
      lm_id_map[id] = idx;

      const auto& p_w_data = lm_json["p_w"];

      landmarks_[idx].p_w = Vec3(static_cast<Scalar>(p_w_data[0]),
                                 static_cast<Scalar>(p_w_data[1]),
                                 static_cast<Scalar>(p_w_data[2]));
      landmarks_[idx].color = Eigen::Vector3<uint8_t>(128, 128, 128);
      idx++;
    }
  }

  // Load observations
  if (j.contains("observations")) {
    const auto& obs_json = j["observations"];

    for (const auto& obs : obs_json) {
      size_t kf_id = obs["kf_id"];
      size_t cam_id = obs["cam_id"];
      size_t lm_id = obs["lm_id"];
      const auto& pos_data = obs["pos"];

      // Map JSON IDs to vector indices
      size_t kf_idx = kf_id_map.at(kf_id);
      size_t lm_idx = lm_id_map.at(lm_id);

      TimeCamId tcid(kf_idx, cam_id);
      Observation observation;
      observation.pos = Vec2(static_cast<Scalar>(pos_data[0]),
                             static_cast<Scalar>(pos_data[1]));

      landmarks_[lm_idx].obs[tcid] = observation;
    }
  }

  if (!quiet_) {
    LOG(INFO) << "Loaded Basalt format: " << num_keyframes() << " keyframes, "
              << num_landmarks() << " landmarks, " << num_observations()
              << " observations";
  }
}

template <typename Scalar>
bool BalProblem<Scalar>::save_basalt(const std::string& path) {
  nlohmann::json j;

  // Save keyframes
  j["keyframes"] = nlohmann::json::array();
  for (size_t i = 0; i < keyframes_.size(); ++i) {
    const auto& kf = keyframes_[i];
    nlohmann::json kf_json;
    kf_json["id"] = kf.t_ns;

    SE3 T_w_i = kf.T_i_w.inverse();

    auto q = T_w_i.unit_quaternion();
    auto t = T_w_i.translation();

    kf_json["T_w_i"] = {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};

    j["keyframes"].push_back(kf_json);
  }

  // Save landmarks
  j["landmarks"] = nlohmann::json::array();
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    const auto& lm = landmarks_[i];
    nlohmann::json lm_json;
    lm_json["id"] = i;
    lm_json["p_w"] = {lm.p_w.x(), lm.p_w.y(), lm.p_w.z()};
    j["landmarks"].push_back(lm_json);
  }

  // Save observations
  j["observations"] = nlohmann::json::array();
  for (size_t lm_id = 0; lm_id < landmarks_.size(); ++lm_id) {
    const auto& lm = landmarks_[lm_id];
    for (const auto& [tcid, obs] : lm.obs) {
      nlohmann::json obs_json;
      obs_json["kf_id"] = tcid.frame_id;
      obs_json["cam_id"] = tcid.cam_id;
      obs_json["lm_id"] = lm_id;
      obs_json["pos"] = {obs.pos.x(), obs.pos.y()};
      j["observations"].push_back(obs_json);
    }
  }

  // Write to file
  std::ofstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Could not open file for writing: '" << path << "'";
    return false;
  }

  file << j.dump(2);  // Pretty print with 2-space indentation
  file.close();

  if (!quiet_) {
    LOG(INFO) << "Saved Basalt format: " << num_keyframes() << " keyframes, "
              << num_landmarks() << " landmarks, " << num_observations()
              << " observations to " << path;
  }

  return true;
}

template <typename Scalar>
bool BalProblem<Scalar>::save_euroc(const std::string& path) const {
  // Save trajectory in euroc format:
  // timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x
  // [],q_RS_y [],q_RS_z []
  std::ofstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Could not open file for writing: '" << path << "'";
    return false;
  }

  file << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
          "[],q_RS_x [],q_RS_y [],q_RS_z []\n";

  for (const auto& kf : keyframes_) {
    SE3 T_w_i = kf.T_i_w.inverse();
    auto q = T_w_i.unit_quaternion();
    auto t = T_w_i.translation();
    file << kf.t_ns << "," << t.x() << "," << t.y() << "," << t.z() << ","
         << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
  }

  file.close();

  if (!quiet_) {
    LOG(INFO) << "Saved trajectory with " << num_keyframes() << " keyframes to "
              << path;
  }
  return true;
}

template <typename Scalar>
void BalProblem<Scalar>::load_calibration(const std::string& path) {
  basalt::Calibration<double> calib;

  CHECK(!path.empty()) << "Calibration path is empty";

  std::ifstream calib_file(path);
  CHECK(calib_file.is_open()) << "Could not open calibration file: " << path
                              << " (cwd: " << std::filesystem::current_path()
                              << ", errno: " << std::strerror(errno) << ")";

  cereal::JSONInputArchive archive(calib_file);
  archive(calib);
  LOG(INFO) << "Loaded calibration from file: " << path;

  calib_ = calib.template cast<Scalar>();
}

template <typename Scalar>
void BalProblem<Scalar>::add_noise(const double obs_noise_sigma) {
  CHECK_GE(obs_noise_sigma, 0.0);

  if (obs_noise_sigma > 0) {
    if (!quiet_) {
      LOG(INFO) << "Adding noise to observations (sigma: {})"_format(
          obs_noise_sigma);
    }
  } else {
    return;
  }

  std::random_device r;
  std::default_random_engine eng{r()};

  for (auto& lm : landmarks_) {
    lm.p_w += perturbation<Scalar, 3>(obs_noise_sigma, eng);
  }
}

template <typename Scalar>
void BalProblem<Scalar>::normalize(const double new_scale) {
  // TODO: try out normalization mentioned in MCBA paper to see if it has
  // additional benefit on numerics (note that we already have jacobian scaling)

  // compute median point coordinates (x,y,z)
  std::vector<Scalar> tmp(num_landmarks());
  Vec3 median;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < num_landmarks(); ++i) {
      tmp[i] = landmarks_[i].p_w(j);
    }
    median(j) = median_destructive(tmp);
  }

  // compute median absolute deviation (l1-norm)
  for (int i = 0; i < num_landmarks(); ++i) {
    tmp[i] = (landmarks_[i].p_w - median).template lpNorm<1>();
  }
  const Scalar median_abs_deviation = median_destructive(tmp);

  // normalize scale to constant
  const Scalar scale = new_scale / median_abs_deviation;

  if (!quiet_) {
    LOG(INFO) << "Normalizing BAL problem (median: " << median.transpose()
              << ", MAD: " << median_abs_deviation << ", scale: " << scale
              << ")";
  }

  // update landmarks: X = scale * (X - median)
  for (auto& lm : landmarks_) {
    lm.p_w = scale * (lm.p_w - median);
  }

  for (auto& kf : keyframes_) {
    SE3 T_w_i = kf.T_i_w.inverse();
    T_w_i.translation() = scale * (T_w_i.translation() - median);
    kf.T_i_w = T_w_i.inverse();
  }

  for (auto& T_c_i : calib_.T_i_c) {
    T_c_i.translation() *= scale;
  }
}

template <typename Scalar>
void BalProblem<Scalar>::filter_obs(const double threshold) {
  CHECK_GE(threshold, 0.0);

  if (threshold > 0) {
    if (!quiet_) {
      LOG(INFO) << "Filtering observations with z < {}"_format(threshold);
    }
  } else {
    return;
  }

  // Remove observations with depth of 3D point in camera frame closer than
  // threshold.
  for (auto& lm : landmarks_) {
    for (auto it = lm.obs.cbegin(); it != lm.obs.cend();) {
      TimeCamId tcid = it->first;
      const auto& kf = keyframes_.at(tcid.frame_id);
      SE3 T_c_w = calib_.T_i_c[tcid.cam_id].inverse() * kf.T_i_w;
      Vec3 p3d_cam = T_c_w * lm.p_w;

      if (p3d_cam.z() < threshold) {
        it = lm.obs.erase(it);
      } else {
        ++it;
      }
    }
  }

  Landmarks filtered_landmarks;

  // Filter landmarks with number of observations less than 2
  std::copy_if(landmarks_.begin(), landmarks_.end(),
               std::back_inserter(filtered_landmarks),
               [](const auto& lm) { return lm.obs.size() >= 2; });

  landmarks_ = std::move(filtered_landmarks);
}

template <typename Scalar>
void BalProblem<Scalar>::filter_kf(int min_obs_per_kf) {
  CHECK_GE(min_obs_per_kf, 0);

  if (min_obs_per_kf <= 0) {
    return;
  }

  if (!quiet_) {
    LOG(INFO) << "Filtering keyframes with fewer than {} observations"_format(
        min_obs_per_kf);
  }

  // Count total observations per keyframe
  std::vector<int> obs_per_kf(keyframes_.size(), 0);
  for (const auto& lm : landmarks_) {
    for (const auto& [tcid, obs] : lm.obs) {
      obs_per_kf[tcid.frame_id]++;
    }
  }

  // Build a map from old keyframe index -> new index (-1 if removed)
  std::vector<int> kf_new_idx(keyframes_.size(), -1);
  Keyframes filtered_keyframes;
  for (size_t i = 0; i < keyframes_.size(); ++i) {
    if (obs_per_kf[i] >= min_obs_per_kf) {
      kf_new_idx[i] = static_cast<int>(filtered_keyframes.size());
      filtered_keyframes.push_back(keyframes_[i]);
    }
  }

  const int num_removed = static_cast<int>(keyframes_.size()) -
                          static_cast<int>(filtered_keyframes.size());

  if (num_removed == 0) {
    return;
  }

  keyframes_ = std::move(filtered_keyframes);

  // Update observations: remove those pointing to removed keyframes,
  // and re-index the remaining ones.
  for (auto& lm : landmarks_) {
    std::map<TimeCamId, Observation> new_obs;
    for (auto& [tcid, obs] : lm.obs) {
      int new_idx = kf_new_idx[tcid.frame_id];
      if (new_idx >= 0) {
        TimeCamId new_tcid(static_cast<size_t>(new_idx), tcid.cam_id);
        new_obs[new_tcid] = obs;
      }
    }
    lm.obs = std::move(new_obs);
  }

  // Remove landmarks with fewer than 2 observations
  Landmarks filtered_landmarks;
  std::copy_if(landmarks_.begin(), landmarks_.end(),
               std::back_inserter(filtered_landmarks),
               [](const auto& lm) { return lm.obs.size() >= 2; });
  landmarks_ = std::move(filtered_landmarks);

  if (!quiet_) {
    LOG(INFO) << "After filter_kf: removed {} keyframes, {} keyframes, "
                 "{} landmarks, {} observations remaining"_format(
                     num_removed, num_keyframes(), num_landmarks(),
                     num_observations());
  }
}

template <typename Scalar>
void BalProblem<Scalar>::perturb(double rotation_sigma,
                                 double translation_sigma,
                                 double landmark_sigma, int seed) {
  // TODO@tsantucci: perturb keyframes, not cameras
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);
  CHECK_GE(landmark_sigma, 0.0);

  if (rotation_sigma > 0 || translation_sigma > 0 || landmark_sigma > 0) {
    if (!quiet_) {
      LOG(INFO) << "Perturbing state (seed: {}): R: {}, t: {}, p: {}"
                   ""_format(seed, rotation_sigma, translation_sigma,
                             landmark_sigma);
    }
  }

  std::random_device r;
  std::default_random_engine eng =
      seed < 0
          ? std::default_random_engine{std::random_device{}()}
          : std::default_random_engine{
                static_cast<std::default_random_engine::result_type>(seed)};

  if (rotation_sigma > 0 || translation_sigma > 0) {
    // TODO@tsantucci: implement keyframe perturbation
  }

  // perturb landmarks
  if (landmark_sigma > 0) {
    for (auto& lm : landmarks_) {
      lm.p_w += perturbation<Scalar, 3>(landmark_sigma, eng);
    }
  }
}

template <class Scalar>
void BalProblem<Scalar>::postprocress(const BalDatasetOptions& options,
                                      PipelineTimingSummary* timing_summary) {
  Timer t;

  if (options.save_output) {
    save_basalt(options.output_optimized_path + "/output.json");
  }

  if (timing_summary) {
    timing_summary->postprocess_time = t.elapsed();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::backup() {
  for (auto& lm : landmarks_) {
    lm.backup();
  }
  for (auto& kf : keyframes_) {
    kf.backup();
  }
}

template <typename Scalar>
void BalProblem<Scalar>::restore() {
  for (auto& lm : landmarks_) {
    lm.restore();
  }
  for (auto& kf : keyframes_) {
    kf.restore();
  }
}

template <typename Scalar>
int BalProblem<Scalar>::num_observations() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num += lm.obs.size();
  }
  return num;
}

template <typename Scalar>
int BalProblem<Scalar>::max_num_observations_per_lm() const {
  int num = 0;
  for (auto& lm : landmarks_) {
    num = std::max(num, static_cast<int>(lm.obs.size()));
  }
  return num;
}

namespace {  // helper

template <class T>
struct is_map {
  static constexpr bool value = false;
};

template <class Key, class Value>
struct is_map<std::map<Key, Value>> {
  static constexpr bool value = true;
};

// NOLINTNEXTLINE
struct default_initialized_atomic_bool : public std::atomic<bool> {
  default_initialized_atomic_bool() { store(false, std::memory_order_relaxed); }
};

}  // namespace

template <typename Scalar>
double BalProblem<Scalar>::compute_rcs_sparsity() const {
  // TODO@tsantucci: implement this for the new keyframe approach
#if 0

  const int num_cams = num_cameras();
  const int num_rcs_blocks = num_cams * num_cams;

  // Note: absl::flat_hash_set<int> is noticably faster than
  // std::unordered_set<int> and a lot faster than
  // std::unordered_set<pair<size_t, size_t>>. An array of bool is a lot
  // faster still, but might need a lot of memory for problems with many
  // cameras.

#if 0
  // absl::flat_hash_set<int> cam_pairs;
  Eigen::VectorX<bool> mask =
      Eigen::VectorX<bool>::Constant(num_rcs_blocks, false);

  for (const auto& lm : landmarks_) {
    for (const auto& [cam_idx_i, _] : lm.obs) {
      for (const auto& [cam_idx_j, _] : lm.obs) {
        if (cam_idx_j < cam_idx_i) {
          int index = cam_idx_i * num_cams + cam_idx_j;
          // cam_pairs.emplace(index);
          mask(index) = true;
        } else {
          // NOTE: the early abort with 'break' assumes ordered lm.obs
          static_assert(is_map<decltype(lm.obs)>::value);
          break;
        }
      }
    }
  }

  // const int num_non_zero_rcs_blocks = num_cams + 2 * cam_pairs.size();
  const int num_non_zero_rcs_blocks = num_cams + 2 * mask.count();
#else

  std::vector<default_initialized_atomic_bool> mask2(num_rcs_blocks);

  // TODO: verify that we really don't need memory barrier before and after
  // parallel for

  auto body = [&](const tbb::blocked_range<size_t>& range) {
    for (size_t r = range.begin(); r != range.end(); ++r) {
      const auto& lm = landmarks_[r];
      for (const auto& [cam_idx_i, _] : lm.obs) {
        for (const auto& [cam_idx_j, _] : lm.obs) {
          if (cam_idx_j < cam_idx_i) {
            int index = cam_idx_i * num_cams + cam_idx_j;
            mask2[index].store(true, std::memory_order_relaxed);
          } else {
            // NOTE: the early abort with 'break' assumes ordered lm.obs
            static_assert(is_map<decltype(lm.obs)>::value);
            break;
          }
        }
      }
    }
  };

  tbb::blocked_range<size_t> range(0, landmarks_.size());
  tbb::parallel_for(range, body);

  const int num_non_zero_rcs_blocks =
      num_cams + 2 * std::count(mask2.begin(), mask2.end(), true);
#endif

  return 1. - num_non_zero_rcs_blocks / double(num_rcs_blocks);
#endif

  return 0.0;
}

template <class Scalar>
void BalProblem<Scalar>::summarize_problem(DatasetSummary& summary,
                                           bool compute_sparsity) const {
  summary.type = "bal";
  summary.num_keyframes = num_keyframes();
  summary.num_landmarks = num_landmarks();
  summary.num_observations = num_observations();

  if (compute_sparsity) {
    // can be a bit expensive for dense problems, so compute only when needed
    Timer timer;
    summary.rcs_sparsity = compute_rcs_sparsity();

    if (!quiet_) {
      // output runtime for this computation, b/c it can be quite large for
      // denser problems (so we notice when we should work in improving runtime)
      LOG(INFO) << "Computed RCS sparsity: {:.2f} ({:.3f}s)"_format(
          summary.rcs_sparsity, timer.elapsed());
    }
  }

  auto stats = [](const ArrXd& data) {
    DatasetSummary::Stats res;
    res.mean = data.mean();
    res.min = data.minCoeff();
    res.max = data.maxCoeff();
    res.stddev = std::sqrt((data - res.mean).square().sum() / data.size());
    return res;
  };

  // per landmark observation stats
  {
    ArrXd per_lm_obs(num_landmarks());
    for (int i = 0; i < num_landmarks(); ++i) {
      per_lm_obs(i) = landmarks_.at(i).obs.size();
    }
    summary.per_lm_obs = stats(per_lm_obs);
    CHECK_NEAR(summary.per_lm_obs.mean,
               double(num_observations()) / num_landmarks(), 1e-9);
  }

  // no per hostframe landmark stats
  summary.per_host_lms = DatasetSummary::Stats();
}

template <typename Scalar>
std::string BalProblem<Scalar>::stats_to_string() const {
  DatasetSummary summary;
  summarize_problem(summary, false);

  return "BAL problem stats: {} keyframes, {} lms, {} obs, per-lm-obs: "
         "{:.1f}+-{:.1f}/{}/{}"
         ""_format(num_keyframes(), num_landmarks(), num_observations(),
                   summary.per_lm_obs.mean, summary.per_lm_obs.stddev,
                   int(summary.per_lm_obs.min), int(summary.per_lm_obs.max));
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  // random seed
  if (options.random_seed >= 0) {
    std::srand(options.random_seed);
  }

  Timer timer;

  // auto detect input type
  BalDatasetOptions::DatasetType input_type = options.input_type;
  if (BalDatasetOptions::DatasetType::AUTO == input_type) {
    input_type = autodetect_input_type(options.input);
    if (!options.quiet) {
      LOG(INFO) << "Autodetected input dataset type as {}."_format(
          wise_enum::to_string(input_type));
    }
  }

  // load dataset as double
  BalProblem<double> bal_problem;
  bal_problem.set_quiet(options.quiet);
  bal_problem.load_basalt(options.input);
  bal_problem.load_calibration(options.calibration_file);

  const double time_load = timer.reset();

  // normalize to fixed scale and center (as double, since there are some
  // overflow issues with float for large problems)
  if (options.normalize) {
    bal_problem.normalize(options.normalization_scale);
  }

  // perturb state if sigmas are positive
  bal_problem.perturb(options.rotation_sigma, options.translation_sigma,
                      options.point_sigma, options.random_seed);

  // Filter observations of points closer than threshold to the camera
  bal_problem.filter_obs(options.init_depth_threshold);

  // Filter keyframes with too few observations
  bal_problem.filter_kf(options.min_obs_per_kf);

  // convert to Scalar if needed
  BalProblem<Scalar> res;
  if constexpr (std::is_same_v<Scalar, double>) {
    res = std::move(bal_problem);
  } else {
    res = bal_problem.copy_cast<Scalar>();
  }

  const double time_preprocess = timer.reset();

  if (timing_summary) {
    timing_summary->load_time = time_load;
    timing_summary->preprocess_time = time_preprocess;
  }

  if (dataset_summary) {
    dataset_summary->input_path = options.input;
    res.summarize_problem(*dataset_summary, true);
  }

  // print some info
  if (!options.quiet) {
    LOG(INFO) << res.stats_to_string();
  }

  return res;
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary) {
  BalDatasetOptions options;
  options.input = path;
  return load_normalized_bal_problem<Scalar>(options, dataset_summary,
                                             timing_summary);
}

template <class Scalar>
BalProblem<Scalar> load_normalized_bal_problem_quiet(const std::string& path) {
  BalDatasetOptions options;
  options.input = path;
  options.quiet = true;
  return load_normalized_bal_problem<Scalar>(options);
}

#ifdef ROOTBA_INSTANTIATIONS_FLOAT
template class BalProblem<float>;

template BalProblem<float> load_normalized_bal_problem<float>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem<float>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<float> load_normalized_bal_problem_quiet<float>(
    const std::string& path);
#endif

// BalProblem in double is used by the ceres solver and GUI, so always
// compile it; it should not be a big compilation overhead.
// #ifdef ROOTBA_INSTANTIATIONS_DOUBLE
template class BalProblem<double>;

template BalProblem<double> load_normalized_bal_problem<double>(
    const BalDatasetOptions& options, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem<double>(
    const std::string& path, DatasetSummary* dataset_summary,
    PipelineTimingSummary* timing_summary);

template BalProblem<double> load_normalized_bal_problem_quiet<double>(
    const std::string& path);
// #endif

}  // namespace rootba
