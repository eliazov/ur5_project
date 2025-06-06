#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # o usa un modello custom addestrato
        
        # CV Bridge per convertire immagini ROS
        self.bridge = CvBridge()
        
        # TF2 per trasformazioni coordinate
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Parametri camera (da calibrare)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = "camera_rgb_frame"
        self.base_frame = "base_link"
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw/image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/image_raw/depth_image', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/image_raw/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.pose_array_pub = self.create_publisher(PoseArray, '/detected_objects', 10)
        self.classification_pub = self.create_publisher(String, '/object_classification', 10)
        self.debug_image_pub = self.create_publisher(Image, '/vision_debug', 10)
        
        # Variabili per immagini
        self.latest_color_image = None
        self.latest_depth_image = None
        
        # Mapping classi YOLO -> nomi oggetti
        self.class_mapping = {
            0: 'cube',      # box/cube
            1: 'cylinder',  # bottle/cylinder  
            2: 'sphere',    # sports ball/sphere
            3: 'prism'      # custom class se addestri il modello
        }
        
        # Timer per processing periodico
        self.timer = self.create_timer(0.1, self.process_images)
        
        self.get_logger().info("Vision Node initialized with YOLO")

    def camera_info_callback(self, msg):
        """Callback per parametri intrinseci camera"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Callback per immagine RGB"""
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")

    def depth_callback(self, msg):
        """Callback per immagine depth"""
        try:
            if msg.encoding == "32FC1":
                # Se è 32FC1, i valori sono float in metri. CvBridge lo gestirà.
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                # self.latest_depth_image avrà dtype=np.float32
                # self.get_logger().info(f"Depth image dtype: {self.latest_depth_image.dtype}") # Per debug
            elif msg.encoding == "16UC1":
                # Se fosse 16UC1 (mm), CvBridge lo gestisce.
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                # Prova una conversione generica se l'encoding non è direttamente supportato
                # o logga un errore e non aggiornare.
                # Per robustezza, potresti voler convertire esplicitamente a '32FC1' o '16UC1'
                # se sai cosa aspettarti o come gestire altre conversioni.
                # Per ora, tentiamo 'passthrough' che spesso funziona o converte in un tipo di base.
                self.get_logger().warn(f"Unexpected depth image encoding '{msg.encoding}'. Attempting passthrough.")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def process_images(self):
        """Processo principale di computer vision"""
        if self.latest_color_image is None or self.latest_depth_image is None:
            return
        
        if self.camera_matrix is None:
            return
            
        # Rileva oggetti con YOLO
        detections = self.detect_objects(self.latest_color_image)
        
        if not detections:
            return
            
        # Converti detection in pose 3D
        poses, classifications = self.detections_to_poses(detections)
        
        if poses:
            # Pubblica pose array
            self.publish_poses(poses)
            
            # Pubblica classificazioni
            self.publish_classifications(classifications)
            
            # Pubblica immagine debug
            debug_image = self.draw_detections(self.latest_color_image.copy(), detections)
            self.publish_debug_image(debug_image)

    def detect_objects(self, image):
        """Rileva oggetti usando YOLO"""
        try:
            results = self.model(image, conf=0.5, iou=0.4)
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Estrai informazioni detection
                        xyxy = box.xyxy[0].cpu().numpy()  # bbox coordinates
                        conf = box.conf[0].cpu().numpy()  # confidence
                        cls = int(box.cls[0].cpu().numpy())  # class
                        
                        # Calcola centro bbox
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        
                        detection = {
                            'bbox': xyxy,
                            'confidence': conf,
                            'class': cls,
                            'center': (center_x, center_y),
                            'class_name': self.class_mapping.get(cls, 'unknown')
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLO detection: {e}")
            return []

    def detections_to_poses(self, detections):
        """Converte detection 2D in pose 3D usando depth"""
        poses = []
        classifications = []
        
        for i, detection in enumerate(detections):
            try:
                # Ottieni coordinate pixel
                center_x, center_y = detection['center']
                
                # Ottieni profondità
                depth_value = self.get_depth_at_pixel(center_x, center_y)
                if depth_value == 0 or depth_value > 2000:  # Filtro valori invalidi
                    continue
                
                # Converti pixel in coordinate 3D camera
                depth_m = depth_value / 1000.0  # mm to m
                camera_point = self.pixel_to_camera_coords(center_x, center_y, depth_m)
                
                # Trasforma in coordinate base_link
                world_pose = self.transform_to_base_frame(camera_point)
                
                if world_pose:
                    poses.append(world_pose)
                    classifications.append({
                        'object_id': f'object_{i}',
                        'class_name': detection['class_name']
                    })
                    
            except Exception as e:
                self.get_logger().error(f"Error converting detection to pose: {e}")
                continue
        
        return poses, classifications

    def get_depth_at_pixel(self, x, y):
        """Ottiene valore depth per pixel specifico"""
        if self.latest_depth_image is None:
            return 0
            
        height, width = self.latest_depth_image.shape
        if 0 <= x < width and 0 <= y < height:
            # Media su piccola area per ridurre noise
            roi_size = 5
            x1 = max(0, x - roi_size//2)
            x2 = min(width, x + roi_size//2)
            y1 = max(0, y - roi_size//2)
            y2 = min(height, y + roi_size//2)
            
            roi = self.latest_depth_image[y1:y2, x1:x2]
            valid_depths = roi[roi > 0]
            
            if len(valid_depths) > 0:
                return np.median(valid_depths)
        
        return 0

    def pixel_to_camera_coords(self, u, v, depth):
        """Converte coordinate pixel in coordinate 3D camera"""
        if self.camera_matrix is None:
            return None
            
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])

    def transform_to_base_frame(self, camera_point):
        """Trasforma punto da frame camera a base_link"""
        try:
            # Crea stamped point
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = self.camera_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(camera_point[0])
            point_stamped.point.y = float(camera_point[1])
            point_stamped.point.z = float(camera_point[2])
            
            # Trasforma
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time())
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            # Crea Pose
            pose = Pose()
            pose.position.x = transformed_point.point.x
            pose.position.y = transformed_point.point.y
            pose.position.z = transformed_point.point.z
            pose.orientation.w = 1.0  # Orientamento neutro
            
            return pose
            
        except Exception as e:
            self.get_logger().error(f"Transform error: {e}")
            return None

    def publish_poses(self, poses):
        """Pubblica array di pose"""
        pose_array = PoseArray()
        pose_array.header.frame_id = self.base_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.poses = poses
        
        self.pose_array_pub.publish(pose_array)

    def publish_classifications(self, classifications):
        """Pubblica classificazioni oggetti"""
        for cls in classifications:
            msg = String()
            msg.data = f"{cls['object_id']}:{cls['class_name']}"
            self.classification_pub.publish(msg)

    def draw_detections(self, image, detections):
        """Disegna detection su immagine per debug"""
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            
            # Disegna bounding box
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (0, 255, 0), 2)
            
            # Disegna label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, 
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Disegna centro
            center = detection['center']
            cv2.circle(image, center, 5, (255, 0, 0), -1)
        
        return image

    def publish_debug_image(self, image):
        """Pubblica immagine debug"""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()