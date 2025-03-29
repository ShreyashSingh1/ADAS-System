# ADAS Sensor Fusion Module
import numpy as np
import cv2

class SensorFusion:
    """Fuses data from multiple sensors to create a unified view of the environment"""
    
    def __init__(self):
        """Initialize the sensor fusion module"""
        self.object_tracker = None
    
    def calibrate_sensors(self, calibration_params):
        """Calibrate sensors to a common coordinate system
        
        Args:
            calibration_params: Dictionary containing calibration matrices
        """
        self.calibration_params = calibration_params
    
    def project_lidar_to_camera(self, lidar_point):
        """Project a LIDAR point to camera image plane
        
        Args:
            lidar_point: 3D point from LIDAR
            
        Returns:
            2D point in camera image coordinates
        """
        if not hasattr(self, 'calibration_params'):
            raise ValueError("Sensors must be calibrated first")
            
        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(lidar_point, 1)
        
        # Apply projection matrix
        calibration_matrix = self.calibration_params.get('lidar_to_camera', np.eye(4))
        projected_point = np.dot(calibration_matrix, point_homogeneous)
        
        # Convert to image coordinates
        if projected_point[2] == 0:  # Avoid division by zero
            return None
            
        pixel_x = projected_point[0] / projected_point[2]
        pixel_y = projected_point[1] / projected_point[2]
        
        return pixel_x, pixel_y
    
    def match_objects_across_sensors(self, camera_objects, lidar_objects, radar_objects):
        """Match objects detected by different sensors
        
        Args:
            camera_objects: Objects detected by camera
            lidar_objects: Objects detected by LIDAR
            radar_objects: Objects detected by radar
            
        Returns:
            List of matched objects with combined properties
        """
        matched_objects = []
        
        # Simple matching based on spatial proximity
        # In a real system, would use more sophisticated matching algorithms
        
        # Start with camera objects as base
        for cam_obj in camera_objects:
            matched_obj = {**cam_obj, 'source': ['camera']}
            
            # Try to match with LIDAR objects
            cam_center = cam_obj.get('centroid', (0, 0))
            for lidar_obj in lidar_objects:
                # Project LIDAR bbox center to camera coordinates
                if hasattr(self, 'calibration_params'):
                    lidar_center = (
                        (lidar_obj['bbox'][0] + lidar_obj['bbox'][3]) / 2,
                        (lidar_obj['bbox'][1] + lidar_obj['bbox'][4]) / 2,
                        (lidar_obj['bbox'][2] + lidar_obj['bbox'][5]) / 2
                    )
                    projected_center = self.project_lidar_to_camera(lidar_center)
                    
                    if projected_center is not None:
                        # Check if projection is within camera bbox
                        if (cam_obj.get('xmin', 0) <= projected_center[0] <= cam_obj.get('xmax', 0) and
                            cam_obj.get('ymin', 0) <= projected_center[1] <= cam_obj.get('ymax', 0)):
                            # Match found
                            matched_obj.update({
                                'lidar_id': lidar_obj.get('id'),
                                'distance': lidar_obj.get('distance'),
                                'dimensions': lidar_obj.get('dimensions'),
                                'source': matched_obj['source'] + ['lidar']
                            })
                            break
            
            # Try to match with radar objects
            for radar_obj in radar_objects:
                # Simple distance-based matching for demo
                # In a real system, would use more sophisticated matching
                if 'distance' in matched_obj and abs(matched_obj['distance'] - radar_obj.get('distance', 0)) < 2.0:
                    # Match found
                    matched_obj.update({
                        'radar_id': radar_obj.get('id'),
                        'radial_velocity': radar_obj.get('radial_velocity'),
                        'source': matched_obj['source'] + ['radar']
                    })
                    break
            
            matched_objects.append(matched_obj)
        
        # Add LIDAR objects that weren't matched to any camera object
        for lidar_obj in lidar_objects:
            if not any(lidar_obj.get('id') == matched.get('lidar_id') for matched in matched_objects):
                matched_objects.append({**lidar_obj, 'source': ['lidar']})
        
        # Add radar objects that weren't matched to any other object
        for radar_obj in radar_objects:
            if not any(radar_obj.get('id') == matched.get('radar_id') for matched in matched_objects):
                matched_objects.append({**radar_obj, 'source': ['radar']})
        
        return matched_objects
    
    def calculate_sensor_agreement_confidence(self, obj):
        """Calculate confidence based on sensor agreement
        
        Args:
            obj: Object with sensor sources
            
        Returns:
            Confidence score (0-1)
        """
        # Simple confidence calculation based on number of sensors that detected the object
        sources = obj.get('source', [])
        
        if len(sources) == 3:  # All three sensors
            return 0.95
        elif len(sources) == 2:  # Two sensors
            return 0.8
        else:  # Single sensor
            # Base confidence on the sensor type
            if 'camera' in sources:
                return obj.get('confidence', 0.6)
            elif 'lidar' in sources:
                return 0.7
            elif 'radar' in sources:
                return 0.65
            else:
                return 0.5
    
    def merge_sensor_attributes(self, obj):
        """Merge attributes from different sensors
        
        Args:
            obj: Object with attributes from multiple sensors
            
        Returns:
            Object with merged attributes
        """
        merged = {}
        
        # Copy all attributes
        for key, value in obj.items():
            merged[key] = value
        
        # Calculate position using most accurate sensor
        sources = obj.get('source', [])
        if 'lidar' in sources and 'distance' in obj:
            # LIDAR provides most accurate distance
            merged['position_confidence'] = 'high'
        elif 'radar' in sources and 'distance' in obj:
            # Radar provides good distance but less accurate position
            merged['position_confidence'] = 'medium'
        elif 'camera' in sources:
            # Camera provides least accurate distance
            merged['position_confidence'] = 'low'
        
        # Calculate velocity using most accurate sensor
        if 'radar' in sources and 'radial_velocity' in obj:
            # Radar provides most accurate velocity
            merged['velocity_confidence'] = 'high'
        elif 'lidar' in sources and 'tracking_id' in obj:
            # LIDAR can provide velocity through tracking
            merged['velocity_confidence'] = 'medium'
        elif 'camera' in sources and 'tracking_id' in obj:
            # Camera can provide velocity through tracking
            merged['velocity_confidence'] = 'low'
        
        return merged
    
    def fuse_sensor_data(self, camera_objects, lidar_objects, radar_objects):
        """Fuse data from multiple sensors
        
        Args:
            camera_objects: Objects detected by camera
            lidar_objects: Objects detected by LIDAR
            radar_objects: Objects detected by radar
            
        Returns:
            List of fused objects with combined properties
        """
        # Match objects across sensors
        matched_objects = self.match_objects_across_sensors(
            camera_objects, 
            lidar_objects, 
            radar_objects
        )
        
        # Enhance object attributes with sensor-specific information
        enhanced_objects = []
        for obj in matched_objects:
            # Calculate confidence based on sensor agreement
            sensor_confidence = self.calculate_sensor_agreement_confidence(obj)
            
            # Merge attributes from all sensors
            merged_attributes = self.merge_sensor_attributes(obj)
            
            # Create enhanced object
            enhanced_obj = {
                **merged_attributes,
                'fusion_confidence': sensor_confidence,
            }
            
            # Add tracking ID if available
            if 'tracking_id' in obj:
                enhanced_obj['tracking_id'] = obj['tracking_id']
            
            enhanced_objects.append(enhanced_obj)
        
        return enhanced_objects