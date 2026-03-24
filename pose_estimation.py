import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# COCO keypoint indices used
IDX_L_ELBOW = 7
IDX_R_ELBOW = 8
IDX_L_WRIST = 9   # left hand
IDX_R_WRIST = 10  # right hand


class MadgwickFilter:
    """Madgwick AHRS filter: fuses gyro + accel into quaternion [w, x, y, z]."""
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_ts = None

    def update(self, gyro, accel, ts):
        if self.last_ts is None:
            self.last_ts = ts
            return self.q.copy()
        dt = ts - self.last_ts
        self.last_ts = ts
        if dt <= 0:
            return self.q.copy()

        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        norm = np.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return self.q.copy()
        ax, ay, az = ax/norm, ay/norm, az/norm

        F = np.array([
            2*(q1*q3 - q0*q2) - ax,
            2*(q0*q1 + q2*q3) - ay,
            2*(0.5 - q1**2 - q2**2) - az,
        ])
        J = np.array([
            [-2*q2,  2*q3, -2*q0,  2*q1],
            [ 2*q1,  2*q0,  2*q3,  2*q2],
            [    0, -4*q1, -4*q2,     0],
        ])
        step = J.T @ F
        n = np.linalg.norm(step)
        if n > 0:
            step /= n

        qDot = 0.5 * np.array([
            -q1*gx - q2*gy - q3*gz,
             q0*gx + q2*gz - q3*gy,
             q0*gy - q1*gz + q3*gx,
             q0*gz + q1*gy - q2*gx,
        ]) - self.beta * step

        self.q += qDot * dt
        self.q /= np.linalg.norm(self.q)
        return self.q.copy()


def get_3d_point(depth_frame, intrinsics, px, py):
    depth = depth_frame.get_distance(px, py)
    if depth == 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)


def run():
    model = YOLO("yolov8n-pose.pt")

    # --- IMU state ---
    latest_gyro = np.zeros(3)
    latest_accel = np.zeros(3)
    madgwick = MadgwickFilter(beta=0.1)
    latest_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # --- Pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f)

    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    win_name = "Pose + IMU — D435i (press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 480)

    print("Warming up camera (60 frames)...")
    for _ in range(60):
        pipeline.wait_for_frames()
    print("Streaming... Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Poll IMU from frameset
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame  = frames.first_or_default(rs.stream.gyro)
            if accel_frame and accel_frame.is_motion_frame():
                d = accel_frame.as_motion_frame().get_motion_data()
                latest_accel = np.array([d.x, d.y, d.z])
            if gyro_frame and gyro_frame.is_motion_frame():
                d = gyro_frame.as_motion_frame().get_motion_data()
                latest_gyro = np.array([d.x, d.y, d.z])
            ts = frames.get_timestamp() / 1000.0
            latest_quat = madgwick.update(latest_gyro, latest_accel, ts)

            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            info_image = np.zeros((480, 640, 3), dtype=np.uint8)
            y = 20

            # --- YOLOv8 Pose ---
            results = model(color_image, verbose=False)
            if results and results[0].keypoints is not None:
                kpts = results[0].keypoints.xy.cpu().numpy()  # shape: (N, 17, 2)

                # Annotate frame
                color_image = results[0].plot(img=color_image)

                if len(kpts) > 0:
                    kp = kpts[0]  # first detected person

                    raw_joints = {
                        "L_ELBOW": kp[IDX_L_ELBOW],
                        "R_ELBOW": kp[IDX_R_ELBOW],
                        "L_HAND":  kp[IDX_L_WRIST],
                        "R_HAND":  kp[IDX_R_WRIST],
                    }

                    # Compute forearm midpoints
                    for side, ek, hk, mk in [
                        ("L", "L_ELBOW", "L_HAND", "L_FOREARM_MID"),
                        ("R", "R_ELBOW", "R_HAND", "R_FOREARM_MID"),
                    ]:
                        raw_joints[mk] = (raw_joints[ek] + raw_joints[hk]) / 2.0

                    cv2.putText(info_image, "--- Joint Positions (m) ---", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 24

                    for name, (px, py_) in raw_joints.items():
                        px_i = max(0, min(int(px),  w - 1))
                        py_i = max(0, min(int(py_), h - 1))

                        pt = get_3d_point(depth_frame, intrinsics, px_i, py_i)
                        cv2.circle(color_image, (px_i, py_i), 5, (0, 255, 255), -1)

                        if pt:
                            label = f"{name}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})"
                        else:
                            label = f"{name}: depth unavailable"

                        cv2.putText(info_image, label, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                        y += 26

            # --- Camera quaternion from IMU ---
            quat = latest_quat.copy()

            y += 10
            cv2.putText(info_image, "--- Camera Quaternion ---", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 24
            cv2.putText(info_image,
                        f"w={quat[0]:.3f}  x={quat[1]:.3f}  y={quat[2]:.3f}  z={quat[3]:.3f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 255, 128), 1)

            stacked = np.hstack((color_image, info_image))
            cv2.imshow(win_name, stacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    run()
