#!/usr/bin/env python3
# ADAS System Demo

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from adas.sensors import CameraProcessor, LidarProcessor, RadarProcessor
from adas.fusion import SensorFusion
from adas.collision import CollisionDetector, CollisionAvoidance

# Add imports for visualization
from adas.visualization import ADASVisualizer

class ADASSystem:
    """Main ADAS system that integrates all components"""
    
    def __init__(self):
        """Initialize ADAS system components"""
        self.camera_processor = CameraProcessor()
        self.lidar_processor = LidarProcessor()
        self.radar_processor = RadarProcessor()
        self.sensor_fusion = SensorFusion()
        self.collision_detector = CollisionDetector()
        self.collision_avoidance = CollisionAvoidance()
        
        # Initialize calibration parameters
        self.initialize_calibration()
    
    def initialize_calibration(self):
        """Initialize sensor calibration parameters"""
        # Simple identity calibration for demo
        calibration_params = {
            'lidar_to_camera': np.eye(4),
            'radar_to_camera': np.eye(4)
        }
        self.sensor_fusion.calibrate_sensors(calibration_params)
    
    def process_camera_data(self, image, objects=None):
        """Process camera data
        
        Args:
            image: Camera image
            objects: Optional list of objects (for demo)
            
        Returns:
            Processed camera objects
        """
        if objects is not None:
            # For demo, use provided objects
            return objects
        else:
            # Process image with camera processor
            return self.camera_processor.preprocess_data(image)
    
    def process_lidar_data(self, point_cloud):
        """Process LIDAR data
        
        Args:
            point_cloud: LIDAR point cloud
            
        Returns:
            Processed LIDAR objects
        """
        return self.lidar_processor.preprocess_data(point_cloud)
    
    def process_radar_data(self, radar_signals):
        """Process radar data
        
        Args:
            radar_signals: Radar signals
            
        Returns:
            Processed radar objects
        """
        return self.radar_processor.preprocess_data(radar_signals)
    
    def fuse_sensor_data(self, camera_objects, lidar_objects, radar_objects):
        """Fuse data from multiple sensors
        
        Args:
            camera_objects: Objects detected by camera
            lidar_objects: Objects detected by LIDAR
            radar_objects: Objects detected by radar
            
        Returns:
            Fused objects
        """
        return self.sensor_fusion.fuse_sensor_data(camera_objects, lidar_objects, radar_objects)
    
    def detect_obstacles(self, fused_objects, current_path):
        """Detect obstacles in the vehicle's path
        
        Args:
            fused_objects: Fused sensor objects
            current_path: Current planned path
            
        Returns:
            List of obstacles with risk assessment
        """
        return self.collision_detector.identify_obstacles(fused_objects, current_path)
    
    def generate_response(self, obstacles):
        """Generate collision avoidance response
        
        Args:
            obstacles: List of obstacles with risk assessment
            
        Returns:
            Response with alert and control actions
        """
        return self.collision_avoidance.generate_response(obstacles, {})

def generate_sample_image(width=1280, height=720):
    """Generate a sample road image with vehicles
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Sample image
    """
    # Create road background
    image = np.ones((height, width, 3), dtype=np.uint8) * 120  # Gray road
    
    # Draw road markings
    cv2.rectangle(image, (0, 0), (width, height), (70, 70, 70), -1)  # Dark gray road
    
    # Draw lane markings
    lane_width = width // 3
    for i in range(1, 3):
        x = i * lane_width
        cv2.line(image, (x, 0), (x, height), (255, 255, 255), 2)  # White lane markings
    
    # Draw dashed center line
    center_x = width // 2
    for y in range(0, height, 40):
        cv2.line(image, (center_x, y), (center_x, y + 20), (255, 255, 255), 2)
    
    # Draw horizon
    horizon_y = height // 3
    cv2.line(image, (0, horizon_y), (width, horizon_y), (200, 200, 200), 1)
    
    # Draw sky
    cv2.rectangle(image, (0, 0), (width, horizon_y), (255, 255, 255), -1)  # White sky
    
    return image

def add_vehicles(image, num_vehicles=3):
    """Add vehicles to the image
    
    Args:
        image: Input image
        num_vehicles: Number of vehicles to add
        
    Returns:
        Image with vehicles, list of vehicle objects
    """
    height, width = image.shape[:2]
    result = image.copy()
    
    # Define lane centers
    lane_width = width // 3
    lane_centers = [lane_width // 2, lane_width + lane_width // 2, 2 * lane_width + lane_width // 2]
    
    # Create vehicle objects
    vehicles = []
    
    # Add ego vehicle (not visible in image)
    ego_vehicle = {
        'id': 'ego',
        'type': 'car',
        'position': (width // 2, height - 50),
        'width': 80,
        'height': 120,
        'color': (0, 0, 255)  # Red
    }
    
    # Add other vehicles
    for i in range(num_vehicles):
        # Randomly select lane
        lane = np.random.randint(0, 3)
        lane_center = lane_centers[lane]
        
        # Randomly select position
        y_pos = np.random.randint(height // 3 + 50, height - 200)
        
        # Randomly select vehicle type
        vehicle_type = np.random.choice(['car', 'truck', 'motorcycle'])
        
        # Set vehicle dimensions based on type
        if vehicle_type == 'car':
            width_v = 70
            height_v = 100
            color = (0, 0, 255)  # Red
        elif vehicle_type == 'truck':
            width_v = 80
            height_v = 130
            color = (0, 0, 255)  # Red
        else:  # motorcycle
            width_v = 30
            height_v = 60
            color = (0, 165, 255)  # Orange
        
        # Calculate bounding box
        xmin = lane_center - width_v // 2
        ymin = y_pos - height_v // 2
        xmax = xmin + width_v
        ymax = ymin + height_v
        
        # Draw vehicle
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, -1)
        
        # Add details to make it look like a vehicle
        if vehicle_type in ['car', 'truck']:
            # Add windows
            window_width = width_v * 0.8
            window_height = height_v * 0.3
            window_x = xmin + (width_v - window_width) // 2
            window_y = ymin + height_v * 0.2
            cv2.rectangle(result, (int(window_x), int(window_y)), 
                         (int(window_x + window_width), int(window_y + window_height)), 
                         (200, 200, 200), -1)
            
            # Add wheels
            wheel_radius = height_v * 0.1
            wheel_y = ymin + height_v - wheel_radius
            # Left wheel
            cv2.circle(result, (int(xmin + wheel_radius), int(wheel_y)), 
                      int(wheel_radius), (0, 0, 0), -1)
            # Right wheel
            cv2.circle(result, (int(xmax - wheel_radius), int(wheel_y)), 
                      int(wheel_radius), (0, 0, 0), -1)
        
        # Calculate distance based on y position
        distance = 50 * (height - y_pos) / height
        
        # Create vehicle object
        vehicle = {
            'id': f'vehicle_{i}',
            'type': vehicle_type,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'centroid': (xmin + width_v // 2, ymin + height_v // 2),
            'distance': distance,
            'confidence': 0.9,
            'radial_velocity': np.random.uniform(-5, 5)  # Random velocity
        }
        
        vehicles.append(vehicle)
    
    return result, vehicles

def add_pedestrians(image, num_pedestrians=2):
    """Add pedestrians to the image
    
    Args:
        image: Input image
        num_pedestrians: Number of pedestrians to add
        
    Returns:
        Image with pedestrians, list of pedestrian objects
    """
    height, width = image.shape[:2]
    result = image.copy()
    
    # Create pedestrian objects
    pedestrians = []
    
    # Add pedestrians
    for i in range(num_pedestrians):
        # Randomly select position
        x_pos = np.random.randint(100, width - 100)
        y_pos = np.random.randint(height // 3 + 50, height - 100)
        
        # Set pedestrian dimensions
        width_p = 30
        height_p = 70
        
        # Calculate bounding box
        xmin = x_pos - width_p // 2
        ymin = y_pos - height_p // 2
        xmax = xmin + width_p
        ymax = ymin + height_p
        
        # Draw pedestrian
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)  # Green
        
        # Add head
        head_radius = width_p // 2
        head_y = ymin + head_radius
        cv2.circle(result, (x_pos, head_y), head_radius, (0, 200, 0), -1)
        
        # Calculate distance based on y position
        distance = 50 * (height - y_pos) / height
        
        # Create pedestrian object
        pedestrian = {
            'id': f'pedestrian_{i}',
            'type': 'pedestrian',
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'centroid': (x_pos, ymin + height_p // 2),
            'distance': distance,
            'confidence': 0.85,
            'radial_velocity': np.random.uniform(-2, 2)  # Random velocity
        }
        
        pedestrians.append(pedestrian)
    
    return result, pedestrians

def generate_vehicle_path(image, num_points=10):
    """Generate a simple vehicle path for demonstration
    
    Args:
        image: Input image
        num_points: Number of path points
        
    Returns:
        List of path points
    """
    height, width = image.shape[:2]
    
    # Create a simple path down the center lane
    path = []
    lane_width = width // 3
    center_x = lane_width + lane_width // 2  # Center of middle lane
    
    # Generate points from bottom to top
    for i in range(num_points):
        y = height - (i * height // num_points)
        # Add some slight curvature
        x = center_x + int(20 * np.sin(i / num_points * np.pi))
        path.append((x, y))
    
    return path

def main():
    """Main function to run the ADAS demo"""
    # Initialize ADAS system
    adas_system = ADASSystem()
    visualizer = ADASVisualizer()
    
    # Generate sample road image
    road_image = generate_sample_image()
    
    # Add vehicles and pedestrians
    image_with_vehicles, vehicles = add_vehicles(road_image, num_vehicles=4)
    final_image, pedestrians = add_pedestrians(image_with_vehicles, num_pedestrians=3)
    
    # Combine all objects
    all_objects = vehicles + pedestrians
    
    # Generate a sample path
    current_path = generate_vehicle_path(final_image)
    
    # Process camera data (using pre-detected objects for demo)
    camera_objects = adas_system.process_camera_data(final_image, all_objects)
    
    # Create empty LIDAR and radar data for demo
    lidar_objects = []
    radar_objects = []
    
    # Fuse sensor data
    fused_objects = adas_system.fuse_sensor_data(camera_objects, lidar_objects, radar_objects)
    
    # Detect obstacles
    obstacles = adas_system.detect_obstacles(fused_objects, current_path)
    
    # Generate response
    response = adas_system.generate_response(obstacles)
    
    # Visualize results
    visualization = visualizer.visualize_adas_results(
        final_image, 
        fused_objects, 
        obstacles, 
        response, 
        current_path
    )
    
    # Save visualization
    visualizer.save_visualization(visualization, 'output.png')
    
    # Display visualization
    visualizer.display_visualization(visualization)
    
    print("ADAS Demo completed. Visualization saved to 'output.png'")
    
    # Print response information
    print("\nADAS Response:")
    print(f"Response Level: {response.get('response_level', 'none')}")
    print(f"Visual Alert: {response.get('visual_alert', 'none')}")
    print(f"Audible Alert: {response.get('audible_alert', 'none')}")
    print(f"Braking: {response.get('braking', 0.0)}")
    print(f"Steering: {response.get('steering', 'none')}")

if __name__ == "__main__":
    main()