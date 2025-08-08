import cv2
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

from models.motion.motion_features import MotionFeature
from models.behavior.behavior_recognition import BehaviorAnalysisResult


def calculate_magnitude_and_angle(motion_vector: np.ndarray) -> Tuple[float, float]:
    """
    Calculate magnitude and angle from a motion vector.
    
    Args:
        motion_vector: Motion vector as [dx, dy]
        
    Returns:
        Tuple of (magnitude, angle_in_degrees)
    """
    dx, dy = motion_vector
    magnitude = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx) * 180 / np.pi
    
    return magnitude, angle


def filter_features_by_type(features: List[MotionFeature], feature_type: str) -> List[MotionFeature]:
    """
    Filter motion features by type.
    
    Args:
        features: List of motion features
        feature_type: Feature type to filter for
        
    Returns:
        Filtered list of motion features
    """
    return [f for f in features if f.type == feature_type]


def filter_features_by_object_id(features: List[MotionFeature], object_id: int) -> List[MotionFeature]:
    """
    Filter motion features by object ID.
    
    Args:
        features: List of motion features
        object_id: Object ID to filter for
        
    Returns:
        Filtered list of motion features
    """
    return [f for f in features if f.object_id == object_id]


def create_motion_heatmap(features: List[MotionFeature], frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a motion heatmap from motion features.
    
    Args:
        features: List of motion features
        frame_shape: Shape of the frame as (height, width)
        
    Returns:
        Motion heatmap as a numpy array
    """
    height, width = frame_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    for feature in features:
        x, y = feature.position
        
        # Skip if position is outside the frame
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        
        # Calculate magnitude for optical flow features
        if feature.type == 'optical_flow' or feature.type == 'sparse_flow' or feature.type == 'object_flow':
            vx, vy = feature.data
            magnitude = np.sqrt(vx*vx + vy*vy)
            
            # Add magnitude to heatmap with Gaussian weighting
            sigma = 10.0  # Standard deviation of Gaussian
            for i in range(max(0, int(y-3*sigma)), min(height, int(y+3*sigma))):
                for j in range(max(0, int(x-3*sigma)), min(width, int(x+3*sigma))):
                    # Calculate Gaussian weight
                    weight = np.exp(-((i-y)**2 + (j-x)**2) / (2*sigma*sigma))
                    
                    # Add weighted magnitude to heatmap
                    heatmap[i, j] += magnitude * weight * feature.confidence
        
        # Handle motion history features
        elif feature.type == 'motion_history':
            vx, vy, area = feature.data
            
            # Calculate radius based on area
            radius = int(np.sqrt(area) / 10)
            radius = max(5, min(radius, 50))  # Constrain radius
            
            # Add value to heatmap with circular weighting
            for i in range(max(0, int(y-radius)), min(height, int(y+radius))):
                for j in range(max(0, int(x-radius)), min(width, int(x+radius))):
                    # Calculate distance from center
                    distance = np.sqrt((i-y)**2 + (j-x)**2)
                    
                    if distance <= radius:
                        # Linear falloff from center
                        weight = 1.0 - (distance / radius)
                        
                        # Add weighted value to heatmap
                        heatmap[i, j] += weight * feature.confidence
    
    # Normalize to [0, 1]
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """
    Apply a colormap to a heatmap.
    
    Args:
        heatmap: Heatmap as a numpy array
        
    Returns:
        Colored heatmap as BGR image
    """
    # Ensure heatmap is in [0, 1]
    heatmap = np.clip(heatmap, 0, 1)
    
    # Convert to 8-bit
    heatmap_8bit = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    return colored_heatmap


def blend_heatmap_with_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend a heatmap with a frame.
    
    Args:
        frame: Frame as BGR image
        heatmap: Colored heatmap as BGR image
        alpha: Blend factor (0.0 for frame only, 1.0 for heatmap only)
        
    Returns:
        Blended image
    """
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)


def create_motion_field_visualization(frame: np.ndarray, features: List[MotionFeature], 
                                      scale: float = 3.0, thickness: int = 1, 
                                      grid_size: int = 20) -> np.ndarray:
    """
    Create a visualization of motion vectors as a flow field.
    
    Args:
        frame: Frame as BGR image
        features: List of motion features
        scale: Scale factor for vector visualization
        thickness: Thickness of lines
        grid_size: Size of grid cells for summarizing vectors
        
    Returns:
        Frame with motion field visualization
    """
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # Create grid for summary vectors
    grid_x = np.arange(grid_size // 2, w, grid_size)
    grid_y = np.arange(grid_size // 2, h, grid_size)
    
    grid_vectors = {}  # (grid_x, grid_y) -> (sum_vx, sum_vy, count)
    
    # Accumulate vectors in grid cells
    for feature in features:
        if feature.type == 'optical_flow' or feature.type == 'sparse_flow' or feature.type == 'object_flow':
            # Get position and vector
            x, y = feature.position
            vx, vy = feature.data
            
            # Skip if position is outside the frame
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            # Find grid cell
            gx = grid_size * (x // grid_size) + grid_size // 2
            gy = grid_size * (y // grid_size) + grid_size // 2
            
            # Accumulate vector
            if (gx, gy) in grid_vectors:
                sum_vx, sum_vy, count = grid_vectors[(gx, gy)]
                grid_vectors[(gx, gy)] = (sum_vx + vx, sum_vy + vy, count + 1)
            else:
                grid_vectors[(gx, gy)] = (vx, vy, 1)
    
    # Draw summarized vectors
    for (gx, gy), (sum_vx, sum_vy, count) in grid_vectors.items():
        # Calculate average vector
        avg_vx = sum_vx / count
        avg_vy = sum_vy / count
        
        # Calculate magnitude and skip if too small
        magnitude = np.sqrt(avg_vx*avg_vx + avg_vy*avg_vy)
        if magnitude < 0.5:
            continue
        
        # Calculate end point
        end_x = int(gx + avg_vx * scale)
        end_y = int(gy + avg_vy * scale)
        
        # Calculate color based on direction (HSV color space)
        angle = np.arctan2(avg_vy, avg_vx) * 180 / np.pi
        angle = (angle + 180) / 360.0  # Normalize to [0, 1]
        
        # Convert HSV color to BGR
        color = tuple(int(x * 255) for x in colorsys_hsv_to_rgb(angle, 1.0, 1.0))
        
        # Draw arrow
        cv2.arrowedLine(result, (int(gx), int(gy)), (end_x, end_y), color, thickness, tipLength=0.3)
    
    return result


def colorsys_hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB.
    
    Args:
        h: Hue [0, 1]
        s: Saturation [0, 1]
        v: Value [0, 1]
        
    Returns:
        RGB tuple
    """
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    if i % 6 == 0:
        return v, t, p
    elif i % 6 == 1:
        return q, v, p
    elif i % 6 == 2:
        return p, v, t
    elif i % 6 == 3:
        return p, q, v
    elif i % 6 == 4:
        return t, p, v
    else:
        return v, p, q


def visualize_trajectory(frame: np.ndarray, positions: List[Tuple[float, float, int]], 
                         color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2,
                         show_points: bool = True, point_radius: int = 3) -> np.ndarray:
    """
    Visualize a trajectory on a frame.
    
    Args:
        frame: Frame as BGR image
        positions: List of positions as [(x, y, frame_idx), ...]
        color: Color as BGR tuple
        thickness: Thickness of lines
        show_points: Whether to show points
        point_radius: Radius of points
        
    Returns:
        Frame with trajectory visualization
    """
    result = frame.copy()
    
    # Extract x, y coordinates
    points = np.array([(int(x), int(y)) for x, y, _ in positions])
    
    # Draw trajectory lines
    if len(points) > 1:
        cv2.polylines(result, [points], False, color, thickness)
    
    # Draw points
    if show_points:
        for point in points:
            cv2.circle(result, tuple(point), point_radius, color, -1)
    
    return result


def save_motion_features(features: List[MotionFeature], filename: str):
    """
    Save motion features to a JSON file.
    
    Args:
        features: List of motion features
        filename: Output filename
    """
    # Convert features to dictionaries
    feature_dicts = []
    
    for feature in features:
        feature_dict = {
            'type': feature.type,
            'data': feature.data.tolist(),
            'position': feature.position,
            'frame_idx': feature.frame_idx,
            'object_id': feature.object_id,
            'confidence': feature.confidence
        }
        
        feature_dicts.append(feature_dict)
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(feature_dicts, f, indent=2)


def save_behavior_results(results: List[BehaviorAnalysisResult], filename: str):
    """
    Save behavior analysis results to a JSON file.
    
    Args:
        results: List of behavior analysis results
        filename: Output filename
    """
    # Convert results to dictionaries
    result_dicts = [result.to_dict() for result in results]
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(result_dicts, f, indent=2)


def plot_trajectory_features(trajectory_data: Dict[str, Any], output_file: str = None):
    """
    Plot trajectory features for analysis.
    
    Args:
        trajectory_data: Dictionary with trajectory data
        output_file: Output file path (if None, will display the plot)
    """
    # Extract data
    positions = trajectory_data.get('positions', [])
    velocities = trajectory_data.get('velocities', [])
    
    if not positions or not velocities:
        return
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot trajectory
    x_values = [p[0] for p in positions]
    y_values = [p[1] for p in positions]
    frames = [p[2] for p in positions]
    
    axs[0, 0].plot(x_values, y_values, 'b-')
    axs[0, 0].scatter(x_values, y_values, c=frames, cmap='viridis')
    axs[0, 0].set_title('Trajectory (colored by frame)')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].grid(True)
    
    # Plot velocity over time
    vx_values = [v[0] for v in velocities]
    vy_values = [v[1] for v in velocities]
    v_frames = [v[2] for v in velocities]
    
    magnitude = [np.sqrt(vx*vx + vy*vy) for vx, vy in zip(vx_values, vy_values)]
    
    axs[0, 1].plot(v_frames, magnitude, 'r-')
    axs[0, 1].set_title('Speed over time')
    axs[0, 1].set_xlabel('Frame')
    axs[0, 1].set_ylabel('Speed')
    axs[0, 1].grid(True)
    
    # Plot velocity components
    axs[1, 0].plot(v_frames, vx_values, 'g-', label='VX')
    axs[1, 0].plot(v_frames, vy_values, 'b-', label='VY')
    axs[1, 0].set_title('Velocity components')
    axs[1, 0].set_xlabel('Frame')
    axs[1, 0].set_ylabel('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot velocity direction
    angles = [np.arctan2(vy, vx) * 180 / np.pi for vx, vy in zip(vx_values, vy_values)]
    
    axs[1, 1].plot(v_frames, angles, 'c-')
    axs[1, 1].set_title('Direction over time')
    axs[1, 1].set_xlabel('Frame')
    axs[1, 1].set_ylabel('Angle (degrees)')
    axs[1, 1].set_ylim(-180, 180)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()


def extract_relevant_frames(video_path: str, behavior_results: List[BehaviorAnalysisResult], 
                            output_dir: str, margin_frames: int = 15):
    """
    Extract relevant frames around detected behaviors from a video.
    
    Args:
        video_path: Path to video file
        behavior_results: List of behavior analysis results
        output_dir: Output directory for frame images
        margin_frames: Number of frames to extract before and after the behavior
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Group results by behavior type and frame
    behaviors_by_frame = {}
    
    for result in behavior_results:
        frame_idx = result.frame_idx
        
        if frame_idx in behaviors_by_frame:
            behaviors_by_frame[frame_idx].append(result)
        else:
            behaviors_by_frame[frame_idx] = [result]
    
    # Extract frames
    for frame_idx, results in behaviors_by_frame.items():
        # Calculate frame range to extract
        start_frame = max(0, frame_idx - margin_frames)
        end_frame = min(total_frames - 1, frame_idx + margin_frames)
        
        # Get behavior types
        behavior_types = [result.behavior_type.name for result in results]
        behavior_str = "_".join(behavior_types)
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame with current timestamp and behavior type
            frame_filename = f"{i:06d}_{behavior_str}.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            
            # Add frame number overlay
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add behavior type overlay if this is the key frame
            if i == frame_idx:
                y_pos = 70
                for behavior in behavior_types:
                    cv2.putText(frame, f"Behavior: {behavior}", (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y_pos += 40
            
            cv2.imwrite(output_path, frame)
    
    # Release video
    cap.release() 
