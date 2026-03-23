import pyrealsense2 as rs
import numpy as np
import cv2

def test_d435i():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense device found. Check USB connection.")
        return

    dev = devices[0]
    print(f"Device found: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial number: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware version: {dev.get_info(rs.camera_info.firmware_version)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} meters/unit")

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    print("\nWarming up camera (60 frames)...")
    for _ in range(60):
        pipeline.wait_for_frames()
    print("Streaming... Press 'q' to quit.")

    win_name = "D435i Test — Color | Depth (press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1920, 480)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            print(f"\rColor mean: {color_image.mean():.1f}  Depth mean: {depth_colormap.mean():.1f}", end="")

            # Show center pixel depth
            h, w = depth_colormap.shape[:2]
            cx, cy = w // 2, h // 2
            dist = depth_frame.get_distance(cx, cy)
            cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(color_image, f"Depth: {dist:.3f} m", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            stacked = np.hstack((color_image, depth_colormap))
            cv2.imshow(win_name, stacked)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    test_d435i()
