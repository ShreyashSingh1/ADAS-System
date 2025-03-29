# ADAS Visualization Module
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ADASVisualizer:
    """Visualizes ADAS system results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'safe': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow
            'danger': (0, 165, 255),   # Orange
            'critical': (0, 0, 255)    # Red
        }
    
    def draw_objects(self, image, objects):
        """Draw detected objects on the image
        
        Args:
            image: Input image
            objects: List of detected objects
            
        Returns:
            Image with objects drawn
        """
        result = image.copy()
        
        for obj in objects:
            # Skip objects without bounding box
            if not all(k in obj for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                continue
                
            # Get bounding box
            xmin = int(obj['xmin'])
            ymin = int(obj['ymin'])
            xmax = int(obj['xmax'])
            ymax = int(obj['ymax'])
            
            # Determine color based on object type
            if 'type' in obj:
                if obj['type'] == 'pedestrian':
                    color = (0, 255, 0)  # Green
                elif obj['type'] == 'car':
                    color = (0, 0, 255)  # Red
                elif obj['type'] == 'truck':
                    color = (0, 0, 255)  # Red
                elif obj['type'] == 'motorcycle':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (255, 0, 0)  # Blue
            else:
                color = (255, 0, 0)  # Blue
            
            # Draw bounding box
            cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw object info
            obj_type = obj.get('type', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            label = f"{obj_type}: {confidence:.2f}"
            
            # Draw label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (xmin, ymin - label_size[1] - 5), 
                         (xmin + label_size[0], ymin), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def draw_risk_assessment(self, image, obstacles):
        """Draw risk assessment for obstacles
        
        Args:
            image: Input image
            obstacles: List of obstacles with risk assessment
            
        Returns:
            Image with risk assessment drawn
        """
        result = image.copy()
        
        for obj in obstacles:
            # Skip objects without bounding box
            if not all(k in obj for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                continue
                
            # Get bounding box
            xmin = int(obj['xmin'])
            ymin = int(obj['ymin'])
            xmax = int(obj['xmax'])
            ymax = int(obj['ymax'])
            
            # Determine color based on risk level
            risk_level = obj.get('risk_level', 0.0)
            if risk_level >= 0.8:
                color = self.colors['critical']
            elif risk_level >= 0.6:
                color = self.colors['danger']
            elif risk_level >= 0.4:
                color = self.colors['warning']
            else:
                color = self.colors['safe']
            
            # Draw bounding box
            cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw risk info
            risk_text = f"Risk: {risk_level:.2f}"
            distance = obj.get('distance', 0.0)
            distance_text = f"Dist: {distance:.1f}m"
            tti = obj.get('time_to_intersection', float('inf'))
            tti_text = f"TTI: {tti:.1f}s" if tti < 100 else "TTI: N/A"
            
            # Draw text with background
            y_offset = ymin - 5
            for text in [risk_text, distance_text, tti_text]:
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_offset -= text_size[1] + 5
                
                # Draw text background
                cv2.rectangle(result, (xmin, y_offset), 
                             (xmin + text_size[0], y_offset + text_size[1]), 
                             color, -1)
                
                # Draw text
                cv2.putText(result, text, (xmin, y_offset + text_size[1] - 1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def draw_collision_warning(self, image, response):
        """Draw collision warning based on response
        
        Args:
            image: Input image
            response: Collision avoidance response
            
        Returns:
            Image with collision warning drawn
        """
        result = image.copy()
        height, width = result.shape[:2]
        
        if not response:
            return result
            
        # Get response information
        response_level = response.get('response_level', 'none')
        visual_alert = response.get('visual_alert', '')
        audible_alert = response.get('audible_alert', None)
        braking = response.get('braking', 0.0)
        steering = response.get('steering', None)
        
        # Determine alert color
        if response_level == 'critical':
            alert_color = self.colors['critical']
        elif response_level == 'high':
            alert_color = self.colors['danger']
        elif response_level == 'medium':
            alert_color = self.colors['warning']
        else:
            alert_color = self.colors['safe']
        
        # Draw alert box if there's a visual alert
        if visual_alert:
            # Draw alert background
            alert_width = min(width - 20, 400)
            alert_height = 60
            alert_x = (width - alert_width) // 2
            alert_y = 30
            
            # Draw semi-transparent overlay
            overlay = result.copy()
            cv2.rectangle(overlay, (alert_x, alert_y), 
                         (alert_x + alert_width, alert_y + alert_height), 
                         alert_color, -1)
            cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
            
            # Draw border
            cv2.rectangle(result, (alert_x, alert_y), 
                         (alert_x + alert_width, alert_y + alert_height), 
                         alert_color, 2)
            
            # Draw alert text
            cv2.putText(result, visual_alert, (alert_x + 10, alert_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw audible alert indicator if present
            if audible_alert:
                audio_text = f"Audio: {audible_alert}"
                cv2.putText(result, audio_text, (alert_x + 10, alert_y + alert_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw control information
        control_bg_height = 80
        control_bg_y = height - control_bg_height
        
        # Draw semi-transparent control panel background
        overlay = result.copy()
        cv2.rectangle(overlay, (0, control_bg_y), 
                     (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        
        # Draw braking information
        braking_text = f"Braking: {braking:.2f}"
        cv2.putText(result, braking_text, (20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw braking bar
        bar_width = 150
        bar_height = 15
        bar_x = 150
        bar_y = height - 55
        
        # Draw empty bar
        cv2.rectangle(result, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Draw filled bar based on braking value
        filled_width = int(bar_width * braking)
        if filled_width > 0:
            # Color based on braking intensity
            if braking > 0.7:
                bar_color = self.colors['critical']
            elif braking > 0.4:
                bar_color = self.colors['danger']
            else:
                bar_color = self.colors['warning']
                
            cv2.rectangle(result, (bar_x, bar_y), 
                         (bar_x + filled_width, bar_y + bar_height), 
                         bar_color, -1)
        
        # Draw steering information
        steering_text = f"Steering: {steering if steering else 'None'}"
        cv2.putText(result, steering_text, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def draw_path(self, image, path, color=(0, 255, 0), thickness=2):
        """Draw path on the image
        
        Args:
            image: Input image
            path: List of path points
            color: Path color
            thickness: Line thickness
            
        Returns:
            Image with path drawn
        """
        result = image.copy()
        
        if not path or len(path) < 2:
            return result
            
        # Draw path as connected line segments
        for i in range(len(path) - 1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i+1][0]), int(path[i+1][1]))
            cv2.line(result, pt1, pt2, color, thickness)
        
        return result
    
    def visualize_adas_results(self, image, fused_objects, obstacles, response, current_path=None, alternative_path=None):
        """Visualize complete ADAS results
        
        Args:
            image: Input image
            fused_objects: List of fused objects
            obstacles: List of obstacles with risk assessment
            response: Collision avoidance response
            current_path: Current planned path
            alternative_path: Alternative path if available
            
        Returns:
            Visualization image
        """
        # Start with original image
        result = image.copy()
        
        # Draw current path if available
        if current_path:
            result = self.draw_path(result, current_path, color=(0, 255, 0), thickness=2)
        
        # Draw alternative path if available
        if alternative_path:
            result = self.draw_path(result, alternative_path, color=(0, 165, 255), thickness=2)
        
        # Draw risk assessment for obstacles
        result = self.draw_risk_assessment(result, obstacles)
        
        # Draw collision warning
        result = self.draw_collision_warning(result, response)
        
        # Add title and timestamp
        height, width = result.shape[:2]
        title = "ADAS Collision Detection System"
        cv2.putText(result, title, (width // 2 - 150, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def save_visualization(self, image, filename='output.png'):
        """Save visualization to file
        
        Args:
            image: Visualization image
            filename: Output filename
        """
        cv2.imwrite(filename, image)
        
    def display_visualization(self, image, title="ADAS Visualization"):
        """Display visualization using matplotlib
        
        Args:
            image: Visualization image
            title: Plot title
        """
        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()