"""
Data recording script using:
  - Intel RealSense L515  : RGB-D scene capture  (replaces original L515)
  - Intel RealSense D435i : Human pose estimation via MediaPipe  +  IMU orientation
                            (replaces 3x T265 tracking cameras + Leap Motion gloves)

Per-frame output  (saved under <out_directory>/frame_<N>/)
  color_image.jpg          : RGB image from L515
  depth_image.png          : depth image from L515  (÷4 metric, same as original)
  pose.txt                 : 4×4 rotation matrix derived from D435i IMU
  right_arm_keypoints.txt  : (5, 3) 3-D world coords – right elbow/wrist/index/pinky/thumb
  left_arm_keypoints.txt   : (5, 3) 3-D world coords – left  elbow/wrist/index/pinky/thumb

Usage:
  python data_recording.py -s -o ./saved_data
  python data_recording.py -s -v -o ./saved_data --total_frame 5000
  python data_recording.py -s -o ./saved_data --l515_serial ABC123 --d435i_serial XYZ456
"""

import argparse
import copy
import time
import os
import sys
import shutil
import concurrent.futures

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import mediapipe as mp
from scipy.spatial.transform import Rotation
from enum import IntEnum


# ── Camera constants ──────────────────────────────────────────────────────────

# Coordinate correction applied to the IMU-derived pose before visualisation
# (mirrors the original between_cam transform used for the T265/L515 offset)
BETWEEN_CAM = np.eye(4)
BETWEEN_CAM[:3, :3] = np.array([[1.,  0.,  0.],
                                  [0., -1.,  0.],
                                  [0.,  0., -1.]])
BETWEEN_CAM[:3, 3] = np.array([0.0, 0.076, 0.0])

# L515 pinhole intrinsics at 1024×768 (depth stream)
L515_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(
    1024, 768,
    742.390625, 742.3359375,
    474.3203125, 391.91796875,
)


class Preset(IntEnum):
    Custom        = 0
    Default       = 1
    Hand          = 2
    HighAccuracy  = 3
    HighDensity   = 4
    MediumDensity = 5


# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Joints of interest: elbow, wrist, index-tip, pinky-tip, thumb-tip
#   Right side: 14, 16, 20, 18, 22
#   Left  side: 13, 15, 19, 17, 21
RIGHT_ARM_IDX = [14, 16, 20, 18, 22]
LEFT_ARM_IDX  = [13, 15, 19, 17, 21]

JOINT_NAMES = ["elbow", "wrist", "index_tip", "pinky_tip", "thumb_tip"]


# ─────────────────────────────────────────────────────────────────────────────
#  save_frame  (thread-safe – operates only on pre-copied data)
# ─────────────────────────────────────────────────────────────────────────────
def save_frame(
    frame_id,
    out_directory,
    color_buffer,
    depth_buffer,
    pose_buffer,
    right_arm_buffer,
    left_arm_buffer,
):
    """
    Save all modalities for a single frame into  <out_directory>/frame_<id>/.

    Files written
    -------------
    color_image.jpg          RGB from L515
    depth_image.png          depth from L515 (uint16)
    pose.txt                 4×4 matrix (D435i IMU rotation, translation=0)
    right_arm_keypoints.txt  (5, 3) right-arm 3-D world landmarks  [m]
    left_arm_keypoints.txt   (5, 3) left-arm  3-D world landmarks  [m]
    """
    frame_dir = os.path.join(out_directory, f"frame_{frame_id}")
    os.makedirs(frame_dir, exist_ok=True)

    # RGB image  (color_buffer stores numpy RGB arrays)
    cv2.imwrite(
        os.path.join(frame_dir, "color_image.jpg"),
        color_buffer[frame_id][:, :, ::-1],   # RGB → BGR for cv2
    )

    # Depth image
    cv2.imwrite(
        os.path.join(frame_dir, "depth_image.png"),
        depth_buffer[frame_id],
    )

    # IMU-derived pose (4×4)
    np.savetxt(os.path.join(frame_dir, "pose.txt"), pose_buffer[frame_id])

    # Body keypoints  (5 joints × 3 coordinates, in metres)
    np.savetxt(
        os.path.join(frame_dir, "right_arm_keypoints.txt"),
        right_arm_buffer[frame_id],
        header=" ".join(JOINT_NAMES),
    )
    np.savetxt(
        os.path.join(frame_dir, "left_arm_keypoints.txt"),
        left_arm_buffer[frame_id],
        header=" ".join(JOINT_NAMES),
    )

    return f"frame {frame_id + 1} saved"


# ─────────────────────────────────────────────────────────────────────────────
#  ComplementaryFilter  –  quaternion from accelerometer + gyroscope
# ─────────────────────────────────────────────────────────────────────────────
class ComplementaryFilter:
    """
    Estimates device orientation as a quaternion [w, x, y, z] by fusing:
      • Accelerometer  – absolute roll / pitch reference (gravity vector)
      • Gyroscope      – angular velocity integration  (smooth, drifts in yaw)

    The complementary filter blends the two via SLERP:
        q_out = SLERP(q_accel, q_gyro, alpha)
    where  alpha=0.98  means 98 % weight on the gyro prediction and
    2 % correction from the accelerometer every frame.

    Note: yaw (heading) is not observable from the accelerometer alone and
    will drift slowly over time.  For drift-free yaw a magnetometer is needed.
    """

    def __init__(self, alpha: float = 0.98):
        self.alpha    = alpha
        self.q        = np.array([1.0, 0.0, 0.0, 0.0])   # [w, x, y, z]
        self.last_ts  = None

    # ── public API ────────────────────────────────────────────────────────────
    def update(self, accel: np.ndarray, gyro: np.ndarray, timestamp_s: float) -> np.ndarray:
        """
        Update the orientation estimate.

        Parameters
        ----------
        accel       : (3,) raw accelerometer reading  [m/s²]
        gyro        : (3,) raw gyroscope reading       [rad/s]
        timestamp_s : monotonic timestamp              [s]

        Returns
        -------
        q : (4,)  quaternion  [w, x, y, z]
        """
        if self.last_ts is None:
            self.last_ts = timestamp_s
            self.q = self._accel_to_quat(accel)
            return self.q.copy()

        dt = timestamp_s - self.last_ts
        self.last_ts = timestamp_s
        if dt <= 0.0:
            return self.q.copy()

        # ── Gyro integration (predict) ───────────────────────────────────────
        q_scipy  = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])  # [x,y,z,w]
        delta    = Rotation.from_rotvec(gyro * dt)
        q_gyro_s = (Rotation.from_quat(q_scipy) * delta).as_quat()          # [x,y,z,w]
        q_gyro   = np.array([q_gyro_s[3], q_gyro_s[0], q_gyro_s[1], q_gyro_s[2]])

        # ── Accel correction (roll + pitch) ──────────────────────────────────
        if np.linalg.norm(accel) > 1e-3:
            q_accel = self._accel_to_quat(accel)
        else:
            q_accel = q_gyro.copy()

        # ── SLERP blend ───────────────────────────────────────────────────────
        self.q = self._slerp(q_accel, q_gyro, self.alpha)
        return self.q.copy()

    def quat_to_pose4x4(self) -> np.ndarray:
        """
        Convert the current quaternion [w, x, y, z] to a 4×4 pose matrix.
        Translation is set to zero (IMU does not track position).
        """
        q = self.q   # [w, x, y, z]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()   # scipy: [x,y,z,w]
        pose = np.eye(4)
        pose[:3, :3] = rot
        return pose

    # ── private helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _accel_to_quat(accel: np.ndarray) -> np.ndarray:
        """Roll + pitch quaternion from gravity vector; yaw forced to 0."""
        a     = accel / np.linalg.norm(accel)
        roll  = np.arctan2(a[1], a[2])
        pitch = np.arctan2(-a[0], np.sqrt(a[1] ** 2 + a[2] ** 2))
        q_s   = Rotation.from_euler('xy', [roll, pitch]).as_quat()   # [x,y,z,w]
        return np.array([q_s[3], q_s[0], q_s[1], q_s[2]])            # [w,x,y,z]

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation; t=1 → q1."""
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if dot < 0.0:
            q1  = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q)
        theta_0 = np.arccos(dot)
        theta   = theta_0 * t
        s1      = np.sin(theta)    / np.sin(theta_0)
        s0      = np.cos(theta)    - dot * s1
        q       = s0 * q0 + s1 * q1
        return q / np.linalg.norm(q)


# ─────────────────────────────────────────────────────────────────────────────
#  DataRecorder  –  main recording class
# ─────────────────────────────────────────────────────────────────────────────
class DataRecorder:
    """
    Captures synchronised data from two RealSense cameras:

      L515   – RGB-D scene  (color_image.jpg + depth_image.png)
      D435i  – IMU orientation  +  human pose via MediaPipe
                (pose.txt  +  *_arm_keypoints.txt)

    Structure mirrors the original RealsenseProcessor in data_recording.py.
    """

    def __init__(
        self,
        l515_serial:          str,
        d435i_serial:         str,
        total_frame:          int  = 10000,
        store_frame:          bool = False,
        out_directory:        str  = "./saved_data",
        enable_visualization: bool = False,
    ):
        self.l515_serial          = l515_serial
        self.d435i_serial         = d435i_serial
        self.total_frame          = total_frame
        self.store_frame          = store_frame
        self.out_directory        = out_directory
        self.enable_visualization = enable_visualization

        # ── Data buffers (filled during capture, flushed to disk afterwards) ─
        self.color_buffer     = []   # (H, W, 3) uint8   RGB from L515
        self.depth_buffer     = []   # (H, W)    uint16  depth from L515
        self.pose_buffer      = []   # (4, 4)    float64 IMU rotation matrix
        self.right_arm_buffer = []   # (5, 3)    float32 MediaPipe world coords
        self.left_arm_buffer  = []   # (5, 3)    float32 MediaPipe world coords

        self.imu_filter = ComplementaryFilter(alpha=0.98)
        self.vis        = None

        # Latest IMU readings – written by pipeline callback, read in main loop
        self._last_accel: np.ndarray = np.array([0.0, 0.0, 9.8])
        self._last_gyro:  np.ndarray = np.zeros(3)
        self._last_accel_ts: float   = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    def configure_stream(self):
        """
        Initialise hardware streams:
          1. L515  – depth (1024×768) + colour (1920×1080) at 30 fps
          2. D435i – colour (640×480) + accelerometer + gyroscope
          3. MediaPipe Pose model
          4. Open3D visualiser window  (optional)
        """

        # ── D435i colour pipeline ────────────────────────────────────────────
        d435i_ctx           = rs.context()
        self.d435i_pipeline = rs.pipeline(d435i_ctx)
        d435i_cfg           = rs.config()
        d435i_cfg.enable_device(self.d435i_serial)
        d435i_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.d435i_pipeline.start(d435i_cfg)

        # Warm up D435i colour
        print("Warming up D435i...")
        for _ in range(10):
            self.d435i_pipeline.wait_for_frames()
        print("D435i ready.")

        # ── D435i IMU – callback-based (separate pipeline) ───────────────────
        # Pipeline wait_for_frames() cannot synchronise video + IMU reliably;
        # using a frame-arrival callback avoids the 5-second timeout issue.
        def _imu_callback(frame):
            if frame.is_motion_frame():
                motion = frame.as_motion_frame()
                data   = motion.get_motion_data()
                ts_s   = frame.get_timestamp() * 1e-3
                if frame.get_profile().stream_type() == rs.stream.accel:
                    self._last_accel    = np.array([data.x, data.y, data.z])
                    self._last_accel_ts = ts_s
                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    self._last_gyro = np.array([data.x, data.y, data.z])

        imu_ctx           = rs.context()
        self.imu_pipeline = rs.pipeline(imu_ctx)
        imu_cfg           = rs.config()
        imu_cfg.enable_device(self.d435i_serial)
        imu_cfg.enable_stream(rs.stream.accel)
        imu_cfg.enable_stream(rs.stream.gyro)

        self.imu_pipeline.start(imu_cfg, _imu_callback)
        print("D435i IMU callback started.")

        # ── L515 (own context, same pattern as original multi-camera script) ──
        l515_ctx           = rs.context()
        self.l515_pipeline = rs.pipeline(l515_ctx)
        l515_cfg           = rs.config()
        l515_cfg.enable_device(self.l515_serial)
        l515_cfg.enable_stream(rs.stream.depth, 1024,  768, rs.format.z16,  30)
        l515_cfg.enable_stream(rs.stream.color,  960,  540, rs.format.rgb8, 30)

        l515_profile = self.l515_pipeline.start(l515_cfg)
        depth_sensor = l515_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
        self.l515_depth_scale = depth_sensor.get_depth_scale()

        self.l515_align = rs.align(rs.stream.color)

        # Warm up L515
        print("Warming up L515...")
        for _ in range(10):
            self.l515_pipeline.wait_for_frames()
        print("L515 ready.")

        # ── MediaPipe Pose ─────────────────────────────────────────────────────
        self.mp_pose    = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Open3D visualiser ─────────────────────────────────────────────────
        if self.enable_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("DataRecorder – L515 point cloud")
            self.vis.get_view_control().change_field_of_view(step=1.0)

    # ─────────────────────────────────────────────────────────────────────────
    #  L515 helpers
    # ─────────────────────────────────────────────────────────────────────────
    def get_rgbd_frame_from_l515(self):
        """
        Capture one depth-aligned RGB-D frame from the L515.

        Returns
        -------
        rgbd       : o3d.geometry.RGBDImage | None   (None when vis disabled)
        depth_np   : np.ndarray  (H, W) uint16
        color_np   : np.ndarray  (H, W, 3) uint8 RGB
        """
        frames  = self.l515_pipeline.wait_for_frames()
        aligned = self.l515_align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # L515 raw depth → metric: divide by 4  (same as original script)
        depth_np = np.asanyarray(depth_frame.get_data()) // 4
        color_np = np.asanyarray(color_frame.get_data())   # RGB

        rgbd = None
        if self.enable_visualization:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np),
                depth_trunc=4.0,
                convert_rgb_to_intensity=False,
            )
        return rgbd, depth_np, color_np

    # ─────────────────────────────────────────────────────────────────────────
    #  D435i helpers
    # ─────────────────────────────────────────────────────────────────────────
    def get_d435i_frameset(self):
        """
        Grab one colour frameset from the D435i pipeline.
        IMU data is received separately via callback.
        """
        return self.d435i_pipeline.wait_for_frames()

    def get_imu_quaternion(self) -> np.ndarray:
        """
        Update complementary filter with the latest IMU readings (stored by
        the callback) and return the current orientation as a 4×4 matrix.

        Returns
        -------
        pose_4x4 : np.ndarray (4, 4)
        """
        self.imu_filter.update(self._last_accel, self._last_gyro, self._last_accel_ts)
        return self.imu_filter.quat_to_pose4x4()

    def get_human_pose(self, d435i_frameset):
        """
        Run MediaPipe Pose on the D435i colour frame and return 3-D world
        landmark positions (in metres, relative to the subject's hip centre)
        for the lower arms and hands on both sides.

        Joints extracted (5 per side):
            elbow, wrist, index_tip, pinky_tip, thumb_tip

        Parameters
        ----------
        d435i_frameset : rs.composite_frame

        Returns
        -------
        right_kp : np.ndarray (5, 3)   zeros if person not detected
        left_kp  : np.ndarray (5, 3)   zeros if person not detected
        """
        color_frame = d435i_frameset.get_color_frame()
        color_bgr   = np.asanyarray(color_frame.get_data())   # BGR
        color_rgb   = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        results  = self.pose_model.process(color_rgb)

        right_kp = np.zeros((5, 3), dtype=np.float32)
        left_kp  = np.zeros((5, 3), dtype=np.float32)

        if results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark
            for i, idx in enumerate(RIGHT_ARM_IDX):
                right_kp[i] = [wl[idx].x, wl[idx].y, wl[idx].z]
            for i, idx in enumerate(LEFT_ARM_IDX):
                left_kp[i]  = [wl[idx].x, wl[idx].y, wl[idx].z]

        return right_kp, left_kp

    # ─────────────────────────────────────────────────────────────────────────
    #  Main recording loop
    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(self):
        """
        Main capture loop – structure mirrors process_frame() in the original
        data_recording.py.

        Steps each iteration:
          1. Capture RGB-D from L515
          2. Capture D435i composite frameset  (colour + accel + gyro)
          3. Compute IMU quaternion → 4×4 pose matrix
          4. Run MediaPipe Pose on D435i colour frame  → arm keypoints
          5. Update Open3D live visualisation  (optional)
          6. Append data to buffers
        After loop: flush buffers to disk in parallel threads.
        """
        frame_count = 0
        first_frame = True
        pcd         = None
        coord_frame = None
        prev_pose   = None
        view_params = None

        try:
            while frame_count < self.total_frame:

                # ── 1. RGB-D from L515 ────────────────────────────────────────
                rgbd, depth_np, color_np = self.get_rgbd_frame_from_l515()

                # ── 2. D435i composite frameset ───────────────────────────────
                d435i_frameset = self.get_d435i_frameset()

                # ── 3. IMU quaternion → 4×4 pose matrix ──────────────────────
                pose_4x4 = self.get_imu_quaternion()

                # ── 4. Human pose estimation (MediaPipe) ─────────────────────
                right_kp, left_kp = self.get_human_pose(d435i_frameset)

                # ── 5. Live Open3D visualisation ──────────────────────────────
                corrected_pose = pose_4x4 @ BETWEEN_CAM

                if self.enable_visualization and rgbd is not None:
                    if first_frame:
                        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, L515_INTRINSIC)
                        pcd.transform(corrected_pose)

                        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                            size=0.3)
                        coord_frame.transform(corrected_pose)
                        prev_pose = copy.deepcopy(corrected_pose)

                        self.vis.add_geometry(pcd)
                        self.vis.add_geometry(coord_frame)
                        view_params = (
                            self.vis.get_view_control()
                            .convert_to_pinhole_camera_parameters()
                        )
                    else:
                        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd, L515_INTRINSIC)
                        new_pcd.transform(corrected_pose)

                        coord_frame.transform(np.linalg.inv(prev_pose))
                        coord_frame.transform(corrected_pose)
                        prev_pose = copy.deepcopy(corrected_pose)

                        pcd.points = new_pcd.points
                        pcd.colors = new_pcd.colors
                        self.vis.update_geometry(pcd)
                        self.vis.update_geometry(coord_frame)
                        self.vis.get_view_control().convert_from_pinhole_camera_parameters(
                            view_params)

                    self.vis.poll_events()
                    self.vis.update_renderer()

                # ── 6. Append to buffers ──────────────────────────────────────
                if self.store_frame:
                    self.color_buffer.append(copy.deepcopy(color_np))
                    self.depth_buffer.append(copy.deepcopy(depth_np))
                    self.pose_buffer.append(copy.deepcopy(pose_4x4))
                    self.right_arm_buffer.append(copy.deepcopy(right_kp))
                    self.left_arm_buffer.append(copy.deepcopy(left_kp))

                first_frame  = False
                frame_count += 1
                print(f"streamed frame {frame_count}")

        except Exception as exc:
            print(f"An error occurred: {exc}")
            raise
        finally:
            self.l515_pipeline.stop()
            self.d435i_pipeline.stop()
            self.imu_pipeline.stop()
            self.pose_model.close()
            if self.vis is not None:
                self.vis.destroy_window()

            # ── Flush buffers to disk in parallel (same as original) ──────────
            if self.store_frame and frame_count > 0:
                print("Saving frames ...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            save_frame,
                            fid,
                            self.out_directory,
                            self.color_buffer,
                            self.depth_buffer,
                            self.pose_buffer,
                            self.right_arm_buffer,
                            self.left_arm_buffer,
                        )
                        for fid in range(frame_count)
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        print(fut.result(), f"  /  total: {frame_count}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    recorder = DataRecorder(
        l515_serial=args.l515_serial,
        d435i_serial=args.d435i_serial,
        total_frame=args.total_frame,
        store_frame=args.store_frame,
        out_directory=args.out_directory,
        enable_visualization=args.enable_vis,
    )
    recorder.configure_stream()
    recorder.process_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record RGB-D (L515) + human pose & IMU (D435i)."
    )
    parser.add_argument(
        "-s", "--store_frame",
        action="store_true",
        help="Save captured frames to disk",
    )
    parser.add_argument(
        "-v", "--enable_vis",
        action="store_true",
        help="Enable Open3D live point-cloud visualisation",
    )
    parser.add_argument(
        "-o", "--out_directory",
        type=str,
        default="./saved_data",
        help="Output directory for saved frames (default: ./saved_data)",
    )
    parser.add_argument(
        "--l515_serial",
        type=str,
        default="",
        help="Serial number of the Intel RealSense L515 (leave empty for auto-detect)",
    )
    parser.add_argument(
        "--d435i_serial",
        type=str,
        default="",
        help="Serial number of the Intel RealSense D435i (leave empty for auto-detect)",
    )
    parser.add_argument(
        "--total_frame",
        type=int,
        default=10000,
        help="Total number of frames to capture (default: 10000)",
    )

    args = parser.parse_args()

    if args.store_frame:
        if os.path.exists(args.out_directory):
            resp = input(
                f"{args.out_directory} already exists. Override? (y/n): "
            ).strip().lower()
            if resp != "y":
                print("Exiting without overriding.")
                sys.exit()
            shutil.rmtree(args.out_directory)
        os.makedirs(args.out_directory, exist_ok=True)

    main(args)
