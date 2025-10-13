/**
 * @file cuvslam2.h

 * @copyright Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.\n\n
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// TODO(vikuznetsov): move common definitions to a common header for all cuvslam interfaces?
#include "cuvslam.h"

namespace cuvslam {

/**
 * Use this function to check the version of the library you are using.
 * Any one of the pointers could be null.
 * @param[out] major   - major version
 * @param[out] minor   - minor version
 * @return - detailed version in string format
 */
CUVSLAM_API
std::string_view GetVersion(int32_t* major, int32_t* minor);

/**
 * Set verbosity. The higher the value, the more output from the library. 0 (default) for no output.
 * @param[in] verbosity new verbosity value
 */
CUVSLAM_API
void SetVerbosity(int verbosity);

/**
 * Warms up GPU, creates CUDA runtime context.
 * This function is not mandatory to call, but helps to save some time in tracker initialization.
 */
CUVSLAM_API
void WarmUpGPU();

/**
 * Static-size array of 32-bit floats
 */
template <std::size_t N>
using Array = std::array<float, N>;

/**
 * Static-size array of 32-bit integers
 */
template <std::size_t N>
using IntArray = std::array<int32_t, N>;

/**
 * Transformation from one frame to another
 */
struct Pose {
  Array<4> rotation = {0, 0, 0, 1};  ///< rotation quaternion in (x, y, z, w) order
  Array<3> translation = {0, 0, 0};  ///< translation vector
};

/**
 * 6x6 covariance matrix
 */
using PoseCovariance = Array<6 * 6>;

/**
 * @brief Distortion model with parameters
 *
 * Common definitions used below:
 * * principal point \f$(c_x, c_y)\f$\n
 * * focal length \f$(f_x, f_y)\f$\n
 *
 * Supported values of distortion_model:
 * - Pinhole (0 parameters):\n
 *   no distortion, same as Brown with \f$k_0=k_1=k_2=p_0=p_1=0\f$\n
 *
 * - Fisheye (4 parameters):\n
 *   Also known as equidistant distortion model for pinhole cameras.\n
 *   Coefficients k1, k2, k3, k4 are 100% compatible with ethz-asl/kalibr/pinhole-equi and OpenCV::fisheye.\n
 *   See: "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses"\n
 *   by Juho Kannala and Sami S. Brandt for further information.\n
 *   Please note, this approach (pinhole+undistort) has a limitation and works only field-of-view below 180 deg.\n
 *   For the TUMVI dataset FOV is ~190 deg.\n
 *   EuRoC and ORB_SLAM3 use a different approach (direct project/unproject without pinhole) and support >180 deg, so\n
 *   their coefficient is incompatible with this model.\n
 *
 *   * 0-3: fisheye distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$\n
 *   .
 *   Each 3D point \f$(x, y, z)\f$ is projected in the following way:\n
 *   \f$(u, v) = (c_x, c_y) + diag(f_x, f_y) * (distortedR(r) * (x_n, y_n) / r)\f$\n
 *   where:\n
 *      \f$distortedR(r) =
 *          \arctan(r) * (1 + k_1 * \arctan^2(r) + k_2 * \arctan^4(r) + k_3 * \arctan^6(r) + k_4 * \arctan^8(r))\f$\n
 *      \f$x_n = \frac{x}{z}\f$\n
 *      \f$y_n = \frac{y}{z}\f$\n
 *      \f$r = \sqrt{(x_n)^2 + (y_n)^2}\f$\n
 *
 * - Brown (5 parameters):
 *   * 0-2: radial distortion coefficients \f$(k_1, k_2, k_3)\f$\n
 *   * 3-4: tangential distortion coefficients \f$(p_1, p_2)\f$\n
 *   .
 *   Each 3D point \f$(x, y, z)\f$ is projected in the following way:\n
 *      \f$(u, v) = (c_x, c_y) + diag(f_x, f_y) * (radial * (x_n, y_n) + tangentialDistortion)\f$\n
 *   where:\n
 *      \f$radial = (1 + k_1 * r^2 + k_2 * r^4 + k_3 * r^6)\f$\n
 *      \f$tangentialDistortion.x = 2 * p_1 * x_n * y_n + p_2 * (r^2 + 2 * x_n^2)\f$\n
 *      \f$tangentialDistortion.y = p_1 * (r^2 + 2 * y_n^2) + 2 * p_2 * x_n * y_n\f$\n
 *      \f$x_n = \frac{x}{z}\f$\n
 *      \f$y_n = \frac{y}{z}\f$\n
 *      \f$r = \sqrt{(x_n)^2 + (y_n)^2}\f$\n
 *
 * - Polynomial (8 parameters):\n
 *   Coefficients are compatible with first 8 coefficients of OpenCV distortion model.\n
 *
 *   * 0-1: radial distortion coefficients \f$(k_1, k_2)\f$\n
 *   * 2-3: tangential distortion coefficients \f$(p_1, p_2)\f$\n
 *   * 4-7: radial distortion coefficients \f$(k_3, k_4, k_5, k_6)\f$\n
 *   .
 *   Each 3D point \f$(x, y, z)\f$ is projected in the following way:\n
 *     \f$(u, v) = (c_x, c_y) + diag(f_x, f_y) * (radial * (x_n, y_n) + tangentialDistortion)\f$\n
 *   where:\n
 *      \f$radial = \frac{1 + k_1 * r^2 + k_2 * r^4 + k_3 * r^6}{1 + k_4 * r^2 + k_5 * r^4 + k_6 * r^6}\f$\n
 *      \f$tangentialDistortion.x = 2 * p_1 * x_n * y_n + p_2 * (r^2 + 2 * x_n^2)\f$\n
 *      \f$tangentialDistortion.y = p_1 * (r^2 + 2 * y_n^2) + 2 * p_2 * x_n * y_n\f$\n
 *      \f$x_n = \frac{x}{z}\f$\n
 *      \f$y_n = \frac{y}{z}\f$\n
 *      \f$r = \sqrt{(x_n)^2 + (y_n)^2}\f$\n
 */
struct Distortion {
  enum class Model : uint8_t {
    Pinhole,
    Fisheye,
    Brown,
    Polynomial,
  };

  Model model{Model::Pinhole};    ///< distortion model, @see Model
  std::vector<float> parameters;  ///< array of distortion parameters depending on model
};

/**
 * @brief Camera parameters
 *
 * Describes intrinsic and extrinsic parameters of a camera and per-camera settings.
 *
 * For camera coordinate system top left pixel has (0, 0) coordinate (y is down, x is right).
 * It's compatible with ROS CameraInfo/OpenCV.
 */
struct Camera {
  IntArray<2> size;          ///< image size in pixels (width, height)
  Array<2> principal;        ///< principal point
  Array<2> focal;            ///< focal length
  Pose rig_from_camera;      ///< transformation from coordinate frame of the camera to frame of the rig
  Distortion distortion;     ///< distortion params
  int32_t border_top{0};     ///< top border to ignore in pixels (0 to use full frame)
  int32_t border_bottom{0};  ///< bottom border to ignore in pixels (0 to use full frame)
  int32_t border_left{0};    ///< left border to ignore in pixels (0 to use full frame)
  int32_t border_right{0};   ///< right border to ignore in pixels (0 to use full frame)
};

/**
 * IMU Calibration parameters
 */
struct ImuCalibration {
  Pose rig_from_imu;                  /**< Rig from imu transformation.
                                           vRig = rig_from_imu * vImu
                                           - vImu - vector in imu coordinate system
                                           - vRig - vector in rig coordinate system */
  float gyroscope_noise_density;      ///< \f$rad / (s * \sqrt{hz})\f$
  float gyroscope_random_walk;        ///< \f$rad / (s^2 * \sqrt{hz})\f$
  float accelerometer_noise_density;  ///< \f$m / (s^2 * \sqrt{hz})\f$
  float accelerometer_random_walk;    ///< \f$m / (s^3 * \sqrt{hz})\f$
  float frequency;                    ///< \f$hz\f$
};

/**
 * Rig consisting of cameras and 0 or 1 IMU sensors
 */
struct Rig {
  std::vector<Camera> cameras;       ///< Cameras
  std::vector<ImuCalibration> imus;  ///< IMU sensors; 0 or 1 sensor is supported now
};

/**
 * Image
 */
struct ImageData {
  enum class Encoding : uint8_t { MONO8, RGB8 };
  enum class DataType : uint8_t { UINT8, UINT16, FLOAT32 };

  const void* pixels;  ///< Pixels must be stored row-wise
  int32_t width;       ///< image width must match what was provided in CUVSLAM_Camera
  int32_t height;      ///< image height must match what was provided in CUVSLAM_Camera
  int32_t pitch;       ///< bytes per image row including padding for GPU memory images, ignored for CPU images
  Encoding encoding;   ///< grayscale (8 bit) and RGB (8 bit) are supported now
  DataType data_type;  ///< image data type
  bool is_gpu_mem;     ///< is pixels pointer points to GPU or CPU memory buffer
};

struct Image : public ImageData {
  int64_t timestamp_ns;   ///< Image timestamp in nanoseconds
  uint32_t camera_index;  ///< index of the camera in the rig
};

/**
 * IMU measurement
 */
struct ImuMeasurement {
  int64_t timestamp_ns;           ///< IMU measurement timestamp in nanoseconds
  Array<3> linear_accelerations;  ///< in meters per squared second
  Array<3> angular_velocities;    ///< in radians per second
};

/**
 * Pose with timestamp
 */
struct PoseStamped {
  int64_t timestamp_ns;  ///< Pose timestamp in nanoseconds
  Pose pose;             ///< Pose (transformation between two coordinate frames)
};

/**
 * Pose with covariance
 * Pose covariance is defined via matrix exponential:
 * for a random zero-mean perturbation `u` in the tangent space
 * random pose is determined by `mean_pose * exp(u)`.
 */
struct PoseWithCovariance {
  Pose pose;                 ///< Pose (transformation between two coordinate frames)
  PoseCovariance covariance; /**< Row-major representation of the 6x6 covariance matrix
                              The orientation parameters use a fixed-axis representation.
                              In order, the parameters are:
                              (rotation about X axis, rotation about Y axis, rotation about Z axis, x, y, z)
                              Rotation in radians, translation in meters.*/
};

/**
 * Rig pose estimate from the tracker.
 * The rig coordinate frame is user-defined and depends on the extrinsic parameters of the cameras.
 * The cameras' coordinate frames may not match the rig coordinate frame - depending on camers extrinsics.
 * The world coordinate frame is an arbitrary 3D coordinate frame. It coincides with the rig coordinate frame at the
 * first frame.
 */
struct PoseEstimate {
  int64_t timestamp_ns;                              ///< Pose timestamp (in nanoseconds) will match image timestamp
  std::optional<PoseWithCovariance> world_from_rig;  ///< Transform from rig coordinate frame to world coordinate frame
};

/**
 * Observation
 * 2D point with coordinates in image
 */
struct Observation {
  uint64_t id;            ///< observation id
  float u;                ///< 0 <= u < image width
  float v;                ///< 0 <= v < image height
  uint32_t camera_index;  ///< camera index
};

/**
 * Landmark
 * 3D point with coordinates in world frame
 */
struct Landmark {
  uint64_t id;      ///< landmark id
  Array<3> coords;  ///< x, y, z in world frame
};

/**
 * Visual Inertial Odometry (VIO) Tracker
 */
class CUVSLAM_API Odometry {
public:
  using ImageSet = std::vector<Image>;
  using Gravity = Array<3>;

  /**
   * Multicamera mode defines which cameras will be used for mono SOF (primary cameras)
   */
  enum class MulticameraMode : uint8_t {
    /// primary cameras auto selection, each secondary camera must be connected to only one primary camera
    Performance,
    /// all cameras are primary cameras
    Precision,
    /// primary cameras auto selection, secondary cameras may be connected to more than one primary camera
    Moderate,
  };

  /// @brief Odometry mode defines which odometry tracker will be used
  enum class OdometryMode : uint8_t { Multicamera, Inertial, RGBD, Mono };

  /**
   * RGBD odometry settings
   */
  struct RGBDSettings {
    /// Scale of provided depth measurements, default: 1.
    float depth_scale_factor = 1.f;

    /// Depth camera id.
    /// @note Depth image is supposed to be pixel-to-pixel aligned with any RGB camera image.
    /// This field specifies camera id, that depth is aligned with. Default: -1.
    int32_t depth_camera_id = -1;

    /// Allows stereo 2D tracking between depth-aligned camera and any other camera. Default: false.
    bool enable_depth_stereo_tracking = false;
  };

  /**
   * Configuration parameters of the VIO tracker
   */
  struct Config {
    /// Multicamera mode. @see MulticameraMode
    MulticameraMode multicam_mode = MulticameraMode::Precision;
    /// @see OdometryMode
    OdometryMode odometry_mode = OdometryMode::Multicamera;
    /// Enable tracking using GPU
    bool use_gpu = true;
    /// Enable SBA asynchronous mode.
    bool async_sba = true;
    /**
     * Enable internal pose prediction mechanism.
     * If frame rate is high enough it improves tracking performance and stability.
     * As a general rule it is better to use a pose prediction mechanism
     * tailored to a specific application. If you have an IMU, consider using
     * it to provide pose predictions to cuVSLAM.
     */
    bool use_motion_model = true;
    /// Enable image denoising. Disable if the input images have already passed through a denoising filter.
    bool use_denoising = false;
    /// Enable fast and robust tracking between rectified cameras with principal points on the horizontal line.
    bool horizontal_stereo_camera = false;
    /// Enable GetLastObservations. Warning: export flags slow down execution and result in additional memory usage.
    bool enable_observations_export = false;
    /// Enable GetLastLandmarks. Warning: export flags slow down execution and result in additional memory usage.
    bool enable_landmarks_export = false;
    /// Enable GetFinalLandmarks. Warning: export flags slow down execution and result in additional memory usage.
    bool enable_final_landmarks_export = false;
    /// Maximum frame delta in seconds. Odometry will warn if time delta between frames is higher than the threshold.
    float max_frame_delta_s = 1.f;
    /// Directory where input data will be dumped in edex format.
    std::string_view debug_dump_directory;
    /// Enable IMU debug mode.
    bool debug_imu_mode = false;
    /// RGBD odometry settings. @see RGBDSettings
    RGBDSettings rgbd_settings;
  };

  // TODO(vikuznetsov): remove when https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165 is fixed
  static Config GetDefaultConfig() { return Config{}; }

  struct State {
    struct Context;
    using ContextMap = std::unordered_map<uint8_t, std::shared_ptr<Context>>;

    uint64_t frame_id;
    int64_t timestamp_ns;
    Pose delta;
    bool keyframe;
    bool heating;
    std::optional<Gravity> gravity;
    std::vector<Observation> observations;
    std::vector<Landmark> landmarks;
    ContextMap context;
  };

  /**
   * Construct a tracker
   * @param[in] rig  rig setup
   * @param[in] cfg  tracker configuration
   * @throws std::runtime_error if tracker fails to initialize, std::invalid_argument if rig or config is invalid
   */
  Odometry(const Rig& rig, const Config& cfg = GetDefaultConfig());

  Odometry(Odometry&& other) noexcept;

  ~Odometry();

  /**
   * @brief Track a rig pose using current frame
   *
   * Track current frame synchronously: the function blocks until the tracker has computed a pose.
   * By default, this function uses visual odometry to compute a pose.
   * If visual odometry tracker fails to compute a pose, the function returns the position
   * calculated from a user-provided IMU data.
   * If after several calls of Track() visual odometry is not able to recover,
   * then invalid pose will be returned.
   * The track will output poses in the same coordinate system until a loss of tracking.
   * Image timestamps have to match. cuVSLAM will use timestamp of the image taken with camera 0.
   * If a camera rig provides "almost synchronized" frames,
   * the timestamps should be within `Config::frame_sync_threshold_ns`.
   * The number of images for tracker should not exceed rig->num_cameras.
   * @param[in]  images  an array of synchronized images no more than rig->num_cameras
   * @param[in]  masks  (Optional) an array of corresponding masks no more than rig->num_cameras
   * @param[in]  depths  (Optional) an array of corresponding depth images no more than rig->num_cameras
   * @return On success `PoseEstimate` contains estimated rig pose, on failure its `is_valid` flag will be `false`
   * @throws std::invalid_argument if image parameters are invalid, std::runtime_error in case of unexpected errors
   */
  PoseEstimate Track(const ImageSet& images, const ImageSet& masks = {}, const ImageSet& depths = {});

  /**
   * @brief Register IMU measurement
   *
   * If visual odometry loses camera position, it briefly continues execution
   * using user-provided IMU measurements while trying to recover the position.
   * You should call this function several times between image acquisition.
   *
   * - tracker.Track
   * - tracker.RegisterImuMeasurement
   * - ...
   * - tracker.RegisterImuMeasurement
   * - tracker.Track
   *
   * Imu measurement and frame image both have timestamps, so it is important to call these functions in
   * strict ascending order of timestamps. RegisterImuMeasurement is thread-safe so you can call
   * RegisterImuMeasurement and Track in parallel.
   *
   * @param[in] sensor_index Sensor index; must be 0, as only one sensor is supported now
   * @param[in] imu IMU measurements
   * @throws std::invalid_argument if IMU fusion is disabled
   * @see Track
   */
  void RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu);

  /**
   * Get an array of observations from the last VO frame for a specific camera
   * @param[in] camera_index Index of the camera to get observations for
   * @throws std::invalid_argument if stats export is disabled
   * @see Observation
   */
  std::vector<Observation> GetLastObservations(uint32_t camera_index) const;

  /**
   * Get an array of landmarks from the last VO frame;
   * Landmarks are 3D points in the last camera frame.
   * @throws std::invalid_argument if stats export is disabled
   * @see Landmark
   */
  std::vector<Landmark> GetLastLandmarks() const;

  /**
   * Get gravity vector in the last VO frame
   * @return Optional gravity vector. Empty if gravity is not yet available.
   * @throws std::invalid_argument if IMU fusion is disabled
   */
  std::optional<Gravity> GetLastGravity() const;

  /**
   * @brief Get tracker state
   * @param[out] state Odometry state to be filled
   * @throws std::invalid_argument if stats export is disabled
   */
  void GetState(Odometry::State& state) const;

  /**
   * @brief Get all final landmarks from all frames;
   * Landmarks are 3D points in the odometry start frame.
   * @return std::unordered_map<uint64_t, Array<3>>
   * @throws std::invalid_argument if stats export is disabled
   * @see Landmark
   */
  std::unordered_map<uint64_t, Array<3>> GetFinalLandmarks() const;

  /**
   * Get primary camera indices used for tracking
   * @return Vector of primary camera indices
   */
  const std::vector<uint8_t>& GetPrimaryCameras() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

/**
 * Result type that can hold either success data or error information.
 * To be used in callbacks, error_message should not outlive the callback scope.
 */
// TODO(C++23): replace with std::expected
template <typename T>
struct Result {
  std::optional<T> data;
  std::string_view error_message;

  static Result<T> Success(T&& value) { return Result<T>{std::move(value), ""}; }

  static Result<T> Error(std::string_view message) { return Result<T>{std::nullopt, message}; }
};

/**
 * Simultaneous Localization and Mapping (SLAM)
 */
class CUVSLAM_API Slam {
public:
  using ImageSet = std::vector<Image>;

  /**
   * SLAM configuration parameters
   */
  struct Config {
    /// Enable GPU use for SLAM
    bool use_gpu = true;
    /// Synchronous mode (does not run a separate work thread if true)
    bool sync_mode = false;
    /// Enable reading internal data from SLAM
    bool enable_reading_internals = false;
    /// Planar constraints. SLAM poses will be modified so that the camera moves on a horizontal plane.
    bool planar_constraints = false;
    /// Special SLAM mode for visual map building in case ground truth is present.
    /// Not realtime, no loop closure, no map global optimization, SBA must be in main thread
    bool gt_align_mode = false;
    /// Size of map cell. Default is 0 (the size will be calculated from the camera baseline).
    float map_cell_size = 0.0f;
    /// Maximum number of poses in SLAM pose graph. 300 is suitable for real-time mapping.
    /// The special value 0 means unlimited pose-graph.
    uint32_t max_map_size = 300;
    /// Minimum time interval between loop closure events in milliseconds.
    /// 1000 is suitable for real-time mapping.
    uint32_t throttling_time_ms = 0;
  };

  // TODO(vikuznetsov): remove when https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165 is fixed
  static Config GetDefaultConfig() { return Config{}; }

  /**
   * Localization settings for use in LocalizeInMap
   */
  struct LocalizationSettings {
    float horizontal_search_radius;  ///< horizontal search radius in meters
    float vertical_search_radius;    ///< vertical search radius in meters
    float horizontal_step;           ///< horizontal step in meters
    float vertical_step;             ///< vertical step in meters
    float angular_step_rads;         ///< angular step around vertical axis in radians
    bool enable_reading_internals;   ///< enable reading internal data from SLAM
  };

  /**
   * Metrics
   */
  struct Metrics {
    int64_t timestamp_ns;                  ///< timestamp of these measurements (in nanoseconds)
    bool lc_status;                        ///< loop closure status
    bool pgo_status;                       ///< pose graph optimization status
    uint32_t lc_selected_landmarks_count;  ///< Count of Landmarks Selected
    uint32_t lc_tracked_landmarks_count;   ///< Count of Landmarks Tracked
    uint32_t lc_pnp_landmarks_count;       ///< Count of Landmarks in PNP
    uint32_t lc_good_landmarks_count;      ///< Count of Landmarks in LC
  };

  /**
   * Data layer for SLAM
   */
  enum class DataLayer : uint8_t {
    Map,                    ///< Landmarks of the map
    LoopClosure,            ///< Map's landmarks that are visible in the last loop closure event
    PoseGraph,              ///< Pose Graph
    LocalizerProbes,        ///< Localizer probes
    LocalizerMap,           ///< Landmarks of the Localizer map (opened database)
    LocalizerObservations,  ///< Landmarks that are visible in the localization
    LocalizerLoopClosure,   ///< Landmarks that are visible in the final loop closure of the localization
    Max,
  };

  /**
   * Pose graph node
   */
  struct PoseGraphNode {
    uint64_t id;     ///< node identifier
    Pose node_pose;  ///< node pose
  };

  /**
   * Pose graph edge
   */
  struct PoseGraphEdge {
    uint64_t node_from;         ///< node id
    uint64_t node_to;           ///< node id
    Pose transform;             ///< transform
    PoseCovariance covariance;  ///< covariance
  };

  /**
   * Pose graph
   */
  struct PoseGraph {
    std::vector<PoseGraphNode> nodes;  ///< nodes list
    std::vector<PoseGraphEdge> edges;  ///< edges list
  };

  /**
   * Landmark with additional information
   */
  struct Landmark {
    uint64_t id;      ///< identifier
    float weight;     ///< weight
    Array<3> coords;  ///< x, y, z in world frame
  };

  /**
   * Landmarks array
   */
  struct Landmarks {
    uint64_t timestamp_ns;            ///< timestamp of landmarks in nanoseconds
    std::vector<Landmark> landmarks;  ///< landmarks list
  };

  /**
   * Localizer probe
   */
  struct LocalizerProbe {
    uint64_t id;                ///< probe identifier
    Pose guess_pose;            ///< input hint
    Pose exact_result_pose;     ///< exact pose if localizer success
    float weight;               ///< input weight
    float exact_result_weight;  ///< result weight
    bool solved;                ///< true for solved, false for unsolved
  };

  /**
   * Localizer probes array
   */
  struct LocalizerProbes {
    uint64_t timestamp_ns;               ///< timestamp of localizer try in nanoseconds
    float size;                          ///< size of search area
    std::vector<LocalizerProbe> probes;  ///< list of probes
  };

  /**
   * Construct a SLAM instance with rig and primary cameras
   * @param[in] rig Camera rig configuration
   * @param[in] primary_cameras Vector of primary camera indices
   * @param[in] config SLAM configuration
   * @throws std::runtime_error if SLAM initialization fails
   */
  Slam(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config = GetDefaultConfig());

  Slam(Slam&& other) noexcept;

  ~Slam();

  /**
   * Process tracking results manually. This should be called after each successful tracking operation.
   * @param[in] state Odometry state containing all tracking data
   * @return On success `Pose` contains rig pose estimated by SLAM
   */
  Pose Track(const Odometry::State& state);

  /**
   * Set rig pose estimated by customer.
   * @param[in] pose rig pose estimated by customer
   */
  void SetSlamPose(const Pose& pose);

  /**
   * Get all SLAM poses for each frame.
   * @param[in] max_poses_count maximum number of poses to return
   * @param[out] poses Vector of poses with timestamps
   * This call could be blocked by slam thread.
   */
  void GetAllSlamPoses(std::vector<PoseStamped>& poses, uint32_t max_poses_count = 0) const;

  /**
   * Save SLAM database (map) to folder asynchronously.
   * This folder will be created, if it does not exist.
   * Contents of the folder will be overwritten.
   * @param[in] folder_name Folder name, where SLAM database (map) will be saved
   * @param[in] callback Callback function to be called when save is complete
   */
  void SaveMap(const std::string_view& folder_name, std::function<void(bool success)> callback) const;

  using LocalizationCallback = std::function<void(const Result<Pose>& result)>;
  /**
   * Localize in the existing database (map) asynchronously.
   * Finds the position of the camera in existing SLAM database (map).
   * If successful, moves the SLAM pose to the found position.
   * @param[in] folder_name Folder name, which stores saved SLAM database (map)
   * @param[in] guess_pose Pointer to the proposed pose, where the robot might be
   * @param[in] images Observed images. Will be used if Config::slam_sync_mode = true
   * @param[in] settings Localization settings
   * @param[in] callback Callback function to be called when localization is complete
   * Errors will be reported in the callback.
   */
  void LocalizeInMap(const std::string_view& folder_name, const Pose& guess_pose, const ImageSet& images,
                     LocalizationSettings settings, LocalizationCallback callback);

  /**
   * Get SLAM metrics.
   * @param[out] metrics SLAM metrics
   */
  void GetSlamMetrics(Metrics& metrics) const;

  /**
   * Get list of last 10 loop closure poses with timestamps.
   * @param[out] poses Vector of poses with timestamps
   */
  void GetLoopClosurePoses(std::vector<PoseStamped>& poses) const;

  /**
   * Enable or disable landmarks layer reading.
   * @param[in] layer Data layer to enable/disable
   * @param[in] max_items_count Maximum items number
   */
  void EnableReadingData(DataLayer layer, uint32_t max_items_count);

  /**
   * Disable reading data layer.
   * @param[in] layer Data layer to disable
   */
  void DisableReadingData(DataLayer layer);

  /**
   * Read landmarks.
   * @param[in] layer Data layer to read
   * @return Landmark info array
   */
  std::shared_ptr<const Landmarks> ReadLandmarks(DataLayer layer);

  /**
   * Read pose graph.
   * @return Pose graph
   */
  std::shared_ptr<const PoseGraph> ReadPoseGraph();

  /**
   * Read localizer probes.
   * @return Localizer probes
   */
  std::shared_ptr<const LocalizerProbes> ReadLocalizerProbes();

  /**
   * Merge existing maps into one map.
   * @param[in] databases Input array of directories with existing databases
   * @param[in] output_folder Directory to save output database
   * @throws std::runtime_error if merge fails
   */
  static void MergeMaps(const Rig& rig, const std::vector<std::string_view>& databases,
                        const std::string_view& output_folder);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace cuvslam
