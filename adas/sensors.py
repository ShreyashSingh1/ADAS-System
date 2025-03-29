# ADAS Sensor Processing Module
import numpy as np
import cv2

class CameraProcessor:
    """Processes data from camera sensors"""
    
    def __init__(self, model=None):
        """Initialize camera processor with optional pre-trained model"""
        self.camera_model = model
    
    def preprocess_data(self, image):
        """Preprocess camera image data
        
        Args:
            image: Raw camera image
            
        Returns:
            List of detected objects with properties
        """
        # Normalize image
        normalized_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        # If no model is provided, use a simple detection method for demo
        if self.camera_model is None:
            # Simple placeholder detection (in real system, would use ML model)
            # This is just for demonstration purposes
            objects = self._simple_detection(image)
            return objects
        
        # Apply object detection using model
        results = self.camera_model(normalized_img)
        
        # Extract object data with additional attributes
        objects = []
        for obj in results.pandas().xyxy[0].to_dict('records'):
            # Calculate additional features
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Store enhanced object data
            objects.append({
                **obj,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'centroid': ((obj['xmin'] + obj['xmax']) / 2, (obj['ymin'] + obj['ymax']) / 2)
            })
        
        return objects
    
    def _simple_detection(self, image):
        """Simple object detection for demonstration purposes"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            # Filter small contours
            if cv2.contourArea(contour) < 500:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create object data
            objects.append({
                'id': f'cam_{i}',
                'type': 'Object',
                'confidence': 0.8,
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h,
                'area': w * h,
                'aspect_ratio': w / h if h > 0 else 0,
                'centroid': (x + w/2, y + h/2)
            })
        
        return objects


class LidarProcessor:
    """Processes data from LIDAR sensors"""
    
    def __init__(self):
        """Initialize LIDAR processor"""
        pass
    
    def preprocess_data(self, point_cloud):
        """Preprocess LIDAR point cloud data
        
        Args:
            point_cloud: Raw LIDAR point cloud
            
        Returns:
            List of detected objects with properties
        """
        # Remove ground plane points
        non_ground_points = self._remove_ground_plane(point_cloud)
        
        # Cluster remaining points into objects
        clustered_objects = self._cluster_points(non_ground_points)
        
        # Extract features for each object
        lidar_objects = []
        for i, cluster in enumerate(clustered_objects):
            # Calculate bounding box, dimensions, orientation
            bbox, dimensions, orientation = self._calculate_object_geometry(cluster)
            
            # Calculate distance from vehicle
            distance = np.min(np.linalg.norm(cluster, axis=1)) if len(cluster) > 0 else 0
            
            lidar_objects.append({
                'id': f'lidar_{i}',
                'points': cluster,
                'bbox': bbox,
                'dimensions': dimensions,
                'orientation': orientation,
                'distance': distance
            })
        
        return lidar_objects
    
    def _remove_ground_plane(self, point_cloud):
        """Remove ground plane points from point cloud"""
        # Simple implementation for demonstration
        # In a real system, would use RANSAC or similar algorithm
        if len(point_cloud) == 0:
            return []
            
        # Assume ground is at z=0, filter points with z > threshold
        non_ground_mask = point_cloud[:, 2] > 0.3  # 30cm threshold
        return point_cloud[non_ground_mask]
    
    def _cluster_points(self, points):
        """Cluster points into distinct objects"""
        # Simple implementation for demonstration
        # In a real system, would use DBSCAN or similar algorithm
        if len(points) == 0:
            return []
            
        # For demo, just return all points as a single cluster
        return [points]
    
    def _calculate_object_geometry(self, cluster):
        """Calculate geometric features of an object"""
        # Simple implementation for demonstration
        if len(cluster) == 0:
            return (0, 0, 0, 0, 0, 0), (0, 0, 0), 0
            
        # Calculate min/max in each dimension
        min_vals = np.min(cluster, axis=0)
        max_vals = np.max(cluster, axis=0)
        
        # Bounding box (xmin, ymin, zmin, xmax, ymax, zmax)
        bbox = (min_vals[0], min_vals[1], min_vals[2], 
                max_vals[0], max_vals[1], max_vals[2])
        
        # Dimensions (length, width, height)
        dimensions = (max_vals[0] - min_vals[0],
                     max_vals[1] - min_vals[1],
                     max_vals[2] - min_vals[2])
        
        # Simple orientation (just a placeholder)
        orientation = 0.0
        
        return bbox, dimensions, orientation


class RadarProcessor:
    """Processes data from radar sensors"""
    
    def __init__(self):
        """Initialize radar processor"""
        pass
    
    def preprocess_data(self, radar_signals):
        """Preprocess radar signals
        
        Args:
            radar_signals: Raw radar signals
            
        Returns:
            List of detected objects with properties
        """
        # Filter noise and clutter
        filtered_signals = self._apply_cfar_detection(radar_signals)
        
        # Track objects over time
        tracked_objects = self._radar_tracking(filtered_signals)
        
        # Extract features
        radar_objects = []
        for i, obj in enumerate(tracked_objects):
            radar_objects.append({
                'id': f'radar_{i}',
                'distance': obj.get('distance', 0),
                'radial_velocity': obj.get('velocity', 0),
                'angle': obj.get('angle', 0),
                'rcs': obj.get('radar_cross_section', 0),  # For object classification
                'tracking_id': obj.get('id', f'track_{i}')
            })
        
        return radar_objects
    
    def _apply_cfar_detection(self, radar_signals):
        """Apply Constant False Alarm Rate detection to filter noise"""
        # Simple implementation for demonstration
        # In a real system, would implement proper CFAR algorithm
        if not radar_signals or len(radar_signals) == 0:
            return []
            
        # For demo, just return the input signals
        return radar_signals
    
    def _radar_tracking(self, filtered_signals):
        """Track objects over time using radar signals"""
        # Simple implementation for demonstration
        # In a real system, would implement JPDA or similar algorithm
        if not filtered_signals or len(filtered_signals) == 0:
            return []
            
        # For demo, convert signals to objects
        tracked_objects = []
        for i, signal in enumerate(filtered_signals):
            tracked_objects.append({
                'id': f'track_{i}',
                'distance': signal.get('distance', 0),
                'velocity': signal.get('velocity', 0),
                'angle': signal.get('angle', 0),
                'radar_cross_section': signal.get('rcs', 0)
            })
        
        return tracked_objects