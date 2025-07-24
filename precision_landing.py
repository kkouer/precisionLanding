import cv2
import numpy as np
import argparse
import logging
import time
from pymavlink import mavutil
import math
import yaml

try:
    import apriltag
except Exception as e:
    raise SystemExit("apriltag library required: pip install apriltag")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("standalone-precision-landing")

# Calibration globals
camera_matrix = None
camera_dist = None

def load_calibration(path):
    global camera_matrix, camera_dist
    if not path:
        return False

    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

            # Ensure required fields exist in the YAML
            if 'camera_matrix' in data and 'distortion_coeffs' in data:
                camera_matrix_data = data['camera_matrix']
                distortion_coeffs_data = data['distortion_coeffs']
                
                # Extract camera matrix values
                focal_length = camera_matrix_data.get('focal_length_pixels', 0)
                principal_point_x = camera_matrix_data.get('principal_point_x', 0)
                principal_point_y = camera_matrix_data.get('principal_point_y', 0)

                # Create camera matrix (assuming it's a 3x3 matrix)
                camera_matrix = np.array([[focal_length, 0, principal_point_x],
                                          [0, focal_length, principal_point_y],
                                          [0, 0, 1]])

                # Extract distortion coefficients
                k1 = distortion_coeffs_data.get('k1', 0)
                k2 = distortion_coeffs_data.get('k2', 0)
                k3 = distortion_coeffs_data.get('k3', 0)
                p1 = distortion_coeffs_data.get('p1', 0)
                p2 = distortion_coeffs_data.get('p2', 0)

                # Create distortion coefficients array
                camera_dist = np.array([k1, k2, k3, p1, p2])

                logger.info("Loaded calibration from %s", path)
                return True
            else:
                logger.warning("Invalid calibration file format")
                return False
    except Exception as e:
        logger.error("Failed to load calibration %s: %s", path, e)
        return False

def has_calibration():
    return camera_matrix is not None and camera_dist is not None

def undistort_image(img):
    if not has_calibration():
        return img
    try:
        return cv2.undistort(img, camera_matrix, camera_dist)
    except Exception as e:
        logger.error("Undistort error: %s", e)
        return img

class RTSPStreamReader:
    def __init__(self):
        self.cap = None
    def start(self, url):
        gst_pipeline = (
            "rtspsrc location=rtsp://192.168.144.108:554/stream=0 latency=100 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        )
        #self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        if not self.cap.isOpened():
            logger.error("Could not open stream %s", url)
            return False
        ret, _ = self.cap.read()
        if not ret:
            logger.error("Failed to read initial frame from %s", url)
            self.cap.release()
            self.cap = None
            return False
        return True
    def get_frame(self):
        if not self.cap:
            return False, None
        ret, frame = self.cap.read()
        return ret and frame is not None, frame
    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# class RTSPStreamReader:
#     def __init__(self):
#         self.cap = None
    
#     def start(self, url):
#         self.cap = cv2.VideoCapture(url)
#         if not self.cap.isOpened():
#             logger.error("Could not open stream %s", url)
#             return False
#         return True
    
#     def undistort_image(self, img):
#         if has_calibration():
#             return cv2.undistort(img, camera_matrix, camera_dist)
#         return img
    
#     def get_frame(self):
#         if self.cap is None:
#             return False, None
#         return self.cap.read()
    
#     def stop(self):
#         if self.cap is not None:
#             self.cap.release()
#             self.cap = None

# AprilTag detection

def detect_april_tag(image, tag_family='tag36h11', target_id=-1, decimate=1):
    options = apriltag.DetectorOptions(families=tag_family,
                                       quad_decimate=float(decimate))
    detector = apriltag.Detector(options)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    detections = detector.detect(gray)
    valid = []
    for d in detections:
        print(f"AprilTag id: {int(d.tag_id)}")
        if target_id != -1 and d.tag_id != target_id:
            continue
        corners = d.corners
        width = np.max(corners[:,0]) - np.min(corners[:,0])
        height = np.max(corners[:,1]) - np.min(corners[:,1])
        valid.append({
            'tag_id': int(d.tag_id),
            'center_x': float(d.center[0]),
            'center_y': float(d.center[1]),
            'width': float(width),
            'height': float(height)
        })
    if not valid:
        return None
    return min(valid, key=lambda x: x['tag_id'])

# FOV helpers

def calculate_vertical_fov(hfov_deg, width, height):
    hfov_rad = math.radians(hfov_deg)
    aspect = height / width
    vfov_rad = 2 * math.atan(math.tan(hfov_rad/2) * aspect)
    return math.degrees(vfov_rad)

def calculate_angular_offsets(cx, cy, width, height, hfov_deg, vfov_deg):
    image_center_x = width / 2
    image_center_y = height / 2
    norm_x = (cx - image_center_x) / (width / 2)
    norm_y = (cy - image_center_y) / (height / 2)
    angle_x = norm_x * math.radians(hfov_deg / 2)
    angle_y = norm_y * math.radians(vfov_deg / 2)
    return angle_x, angle_y

def estimate_target_size(w_px, h_px, img_w, img_h, hfov_deg, vfov_deg):
    px_per_deg_h = img_w / hfov_deg
    px_per_deg_v = img_h / vfov_deg
    size_x_deg = w_px / px_per_deg_h
    size_y_deg = h_px / px_per_deg_v
    return math.radians(size_x_deg), math.radians(size_y_deg)

# MAVLink via pymavlink
def open_mavlink(device, baudrate, sysid):
    try:
        conn = mavutil.mavlink_connection(device, baud=baudrate,
                                           source_system=sysid)
        conn.wait_heartbeat()
        logger.info("MAVLink connected on %s at %d baud (sysid %d)",
                    device, baudrate, sysid)
        return conn
    except Exception as e:
        logger.error("Failed to open MAVLink on %s: %s", device, e)
        return None

def send_landing_target(mav, tag_id, center_x, center_y, w_px, h_px,
                        img_w, img_h, hfov, vfov):
    angle_x, angle_y = calculate_angular_offsets(center_x, center_y,
                                                 img_w, img_h, hfov, vfov)
    size_x, size_y = estimate_target_size(w_px, h_px, img_w, img_h,
                                          hfov, vfov)
    mav.mav.landing_target_send(
        int(time.time()*1e6),
        tag_id,
        mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
        angle_x,
        angle_y,
        0.0,
        size_x,
        size_y
    )
    logger.debug("Sent LANDING_TARGET for ID %s", tag_id)

# Main loop

def main():
    parser = argparse.ArgumentParser(description="Standalone precision landing")
    parser.add_argument('--rtsp', required=True, help='RTSP stream URL')
    parser.add_argument('--calibration', help='Calibration yaml file path')
    parser.add_argument('--hfov', type=float, default=81.0,
                        help='Camera horizontal FOV in degrees')
    parser.add_argument('--device', default='/dev/ttyACM0',
                        help='Serial device for MAVLink')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate for MAVLink')
    parser.add_argument('--sysid', type=int, default=255,
                        help='MAVLink system id for this script')
    parser.add_argument('--tag-id', type=int, default=-1,
                        help='AprilTag ID to track (-1 for any)')
    args = parser.parse_args()

    if args.calibration:
        load_calibration(args.calibration)

    reader = RTSPStreamReader()
    if not reader.start(args.rtsp):
        return

    # mav = open_mavlink(args.device, args.baudrate, args.sysid)
    # if not mav:
    #     return

    logger.info("Starting detection loop")
    while True:
        ret, frame = reader.get_frame()
        if not ret:
            time.sleep(0.05)
            continue
        if has_calibration():
            frame = undistort_image(frame)
        height, width = frame.shape[:2]
        vfov = calculate_vertical_fov(args.hfov, width, height)
        det = detect_april_tag(frame, target_id=args.tag_id)
        # if det:
        #     send_landing_target(mav, det['tag_id'], det['center_x'],
        #                        det['center_y'], det['width'], det['height'],
        #                        width, height, args.hfov, vfov)
        time.sleep(0.01)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting")
