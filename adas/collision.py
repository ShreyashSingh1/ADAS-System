# ADAS Collision Detection and Avoidance Module
import numpy as np
import cv2

class CollisionDetector:
    """Detects potential collisions and assesses risk levels"""
    
    def __init__(self):
        """Initialize collision detector"""
        pass
    
    def identify_obstacles(self, fused_objects, current_path):
        """Identify obstacles that are in or near the vehicle's path
        
        Args:
            fused_objects: List of fused objects from sensor fusion
            current_path: Current planned path of the vehicle
            
        Returns:
            List of obstacles with risk assessment
        """
        obstacles = []
        
        for obj in fused_objects:
            # Calculate distance to path
            min_distance = self._calculate_min_distance_to_path(obj, current_path)
            
            # Calculate time to intersection
            tti = self._calculate_time_to_intersection(obj, current_path)
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(obj, min_distance, tti)
            
            if risk_level > 0:
                obstacles.append({
                    **obj,
                    'distance_to_path': min_distance,
                    'time_to_intersection': tti,
                    'risk_level': risk_level
                })
        
        return obstacles
    
    def _calculate_min_distance_to_path(self, obj, path):
        """Calculate minimum distance from object to path
        
        Args:
            obj: Object with position information
            path: List of path points
            
        Returns:
            Minimum distance to path
        """
        # Simple implementation for demonstration
        # In a real system, would calculate actual distance to path
        
        # If no path or empty object, return large distance
        if not path or not obj:
            return 100.0
            
        # Get object position
        if 'centroid' in obj:
            obj_pos = obj['centroid']
        elif 'position' in obj:
            obj_pos = obj['position']
        else:
            # If no position information, assume far away
            return 100.0
        
        # Calculate minimum distance to any path point
        min_dist = float('inf')
        for point in path:
            dist = np.sqrt((obj_pos[0] - point[0])**2 + (obj_pos[1] - point[1])**2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_time_to_intersection(self, obj, path):
        """Calculate time to intersection with path
        
        Args:
            obj: Object with position and velocity information
            path: List of path points
            
        Returns:
            Time to intersection in seconds
        """
        # Simple implementation for demonstration
        # In a real system, would calculate actual time to intersection
        
        # If no velocity information, use distance as proxy
        if 'radial_velocity' not in obj:
            # Convert distance to time assuming constant velocity
            distance = obj.get('distance', 100.0)
            return distance / 10.0  # Assume 10 m/s if no velocity
        
        # Calculate time to intersection using velocity
        distance = obj.get('distance', 100.0)
        velocity = abs(obj['radial_velocity'])  # Use absolute value
        
        # Avoid division by zero
        if velocity < 0.1:
            return 100.0  # Large value for very slow objects
            
        return distance / velocity
    
    def _calculate_risk_level(self, obj, distance_to_path, time_to_intersection):
        """Calculate risk level based on distance and time
        
        Args:
            obj: Object information
            distance_to_path: Minimum distance to path
            time_to_intersection: Time to intersection
            
        Returns:
            Risk level (0-1)
        """
        # Simple risk calculation for demonstration
        # In a real system, would use more sophisticated risk model
        
        # If object is far from path, low risk
        if distance_to_path > 5.0:
            return 0.0
        
        # If time to intersection is large, low risk
        if time_to_intersection > 10.0:
            return 0.0
        
        # Calculate risk based on distance and time
        distance_factor = 1.0 - min(distance_to_path / 5.0, 1.0)
        time_factor = 1.0 - min(time_to_intersection / 10.0, 1.0)
        
        # Combine factors with weights
        risk = 0.7 * distance_factor + 0.3 * time_factor
        
        # Adjust risk based on object type if available
        if 'type' in obj:
            obj_type = obj['type'].lower()
            if 'pedestrian' in obj_type:
                risk *= 1.5  # Higher risk for pedestrians
            elif 'cyclist' in obj_type:
                risk *= 1.3  # Higher risk for cyclists
        
        # Ensure risk is in [0, 1]
        return max(0.0, min(risk, 1.0))


class CollisionAvoidance:
    """Implements collision avoidance strategies"""
    
    def __init__(self):
        """Initialize collision avoidance system"""
        pass
    
    def assess_overall_risk(self, obstacles):
        """Assess overall risk level based on all obstacles
        
        Args:
            obstacles: List of obstacles with risk assessment
            
        Returns:
            Overall risk level (0-1)
        """
        if not obstacles:
            return 0.0
        
        # Calculate maximum risk
        max_risk = max(obj.get('risk_level', 0.0) for obj in obstacles)
        
        # Calculate average risk of top 3 obstacles
        sorted_obstacles = sorted(obstacles, key=lambda x: x.get('risk_level', 0.0), reverse=True)
        top_obstacles = sorted_obstacles[:min(3, len(sorted_obstacles))]
        avg_risk = sum(obj.get('risk_level', 0.0) for obj in top_obstacles) / len(top_obstacles)
        
        # Combine max and average with weights
        overall_risk = 0.7 * max_risk + 0.3 * avg_risk
        
        return overall_risk
    
    def get_avoidance_path(self, obstacles, current_path, vehicle_state):
        """Generate an avoidance path to avoid obstacles
        
        Args:
            obstacles: List of obstacles with risk assessment
            current_path: Current planned path
            vehicle_state: Current vehicle state
            
        Returns:
            Avoidance path as list of points
        """
        # Simple implementation for demonstration
        # In a real system, would use more sophisticated path planning
        
        if not obstacles or not current_path:
            return current_path
        
        # Get highest risk obstacle
        highest_risk_obstacle = max(obstacles, key=lambda x: x.get('risk_level', 0.0))
        
        # Simple avoidance: shift path away from obstacle
        avoidance_path = []
        for point in current_path:
            # Get obstacle position
            if 'centroid' in highest_risk_obstacle:
                obstacle_pos = highest_risk_obstacle['centroid']
            elif 'position' in highest_risk_obstacle:
                obstacle_pos = highest_risk_obstacle['position']
            else:
                # If no position information, keep original path
                avoidance_path.append(point)
                continue
            
            # Calculate vector from obstacle to path point
            vector = [point[0] - obstacle_pos[0], point[1] - obstacle_pos[1]]
            
            # Normalize vector
            magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
            if magnitude > 0.001:  # Avoid division by zero
                vector = [v / magnitude for v in vector]
            else:
                vector = [1.0, 0.0]  # Default direction if too close
            
            # Shift point away from obstacle
            shift_distance = 2.0  # Shift by 2 meters
            shifted_point = [point[0] + shift_distance * vector[0],
                            point[1] + shift_distance * vector[1]]
            
            avoidance_path.append(shifted_point)
        
        return avoidance_path
    
    def get_emergency_avoidance_path(self, obstacles, current_path, vehicle_state):
        """Generate an emergency avoidance path for critical situations
        
        Args:
            obstacles: List of obstacles with risk assessment
            current_path: Current planned path
            vehicle_state: Current vehicle state
            
        Returns:
            Emergency avoidance path as list of points
        """
        # For emergency, use more aggressive avoidance
        # Similar to regular avoidance but with larger shift
        
        if not obstacles or not current_path:
            return current_path
        
        # Get highest risk obstacle
        highest_risk_obstacle = max(obstacles, key=lambda x: x.get('risk_level', 0.0))
        
        # Emergency avoidance: shift path away from obstacle with larger margin
        emergency_path = []
        for point in current_path:
            # Get obstacle position
            if 'centroid' in highest_risk_obstacle:
                obstacle_pos = highest_risk_obstacle['centroid']
            elif 'position' in highest_risk_obstacle:
                obstacle_pos = highest_risk_obstacle['position']
            else:
                # If no position information, keep original path
                emergency_path.append(point)
                continue
            
            # Calculate vector from obstacle to path point
            vector = [point[0] - obstacle_pos[0], point[1] - obstacle_pos[1]]
            
            # Normalize vector
            magnitude = np.sqrt(vector[0]**2 + vector[1]**2)
            if magnitude > 0.001:  # Avoid division by zero
                vector = [v / magnitude for v in vector]
            else:
                vector = [1.0, 0.0]  # Default direction if too close
            
            # Shift point away from obstacle with larger margin
            shift_distance = 4.0  # Shift by 4 meters for emergency
            shifted_point = [point[0] + shift_distance * vector[0],
                            point[1] + shift_distance * vector[1]]
            
            emergency_path.append(shifted_point)
        
        return emergency_path
    
    def generate_response(self, obstacles, vehicle_state):
        """Generate appropriate response based on risk assessment
        
        Args:
            obstacles: List of obstacles with risk assessment
            vehicle_state: Current vehicle state
            
        Returns:
            Response with alert and control actions
        """
        # Assess overall risk situation
        overall_risk = self.assess_overall_risk(obstacles)
        
        # Determine appropriate response level
        if overall_risk >= 0.8:  # Critical risk
            return {
                'response_level': 'critical',
                'visual_alert': 'IMMEDIATE DANGER',
                'audible_alert': 'CONTINUOUS_HIGH_BEEP',
                'braking': 0.9,  # Maximum braking
                'steering': 'emergency_avoidance'
            }
        elif overall_risk >= 0.6:  # High risk
            return {
                'response_level': 'high',
                'visual_alert': 'COLLISION WARNING',
                'audible_alert': 'HIGH_BEEP',
                'braking': 0.6,
                'steering': 'avoidance'
            }
        elif overall_risk >= 0.4:  # Medium risk
            return {
                'response_level': 'medium',
                'visual_alert': 'WARNING',
                'audible_alert': 'LOW_BEEP',
                'braking': 0.3,
                'steering': None  # Rely on driver steering
            }
        elif overall_risk >= 0.2:  # Low risk
            return {
                'response_level': 'low',
                'visual_alert': 'CAUTION',
                'audible_alert': None,
                'braking': 0,
                'steering': None
            }
        else:
            return {
                'response_level': 'none',
                'visual_alert': None,
                'audible_alert': None,
                'braking': 0,
                'steering': None
            }