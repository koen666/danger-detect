import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime

from models.video_capture import VideoCaptureManager
from models.qwen_vl import QwenVLFeatureExtractor
from models.rga import SceneGraphBuilder, RelationGraphBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Graph Relationship Modeling Test')
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='graph_results',
        help='Directory to save output results'
    )
    parser.add_argument(
        '--model_version',
        type=str,
        default='Qwen/Qwen-VL-Chat',
        help='Qwen-VL model version to use'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Process every N-th frame'
    )
    parser.add_argument(
        '--num_entities',
        type=int,
        default=10,
        help='Maximum number of entities in the scene graph'
    )
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=1024,
        help='Dimension of features'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for relationship detection'
    )
    return parser.parse_args()


class RGADemo:
    """Interactive demo for RGA graph relationship modeling."""
    
    def __init__(self, args):
        """
        Initialize the demo.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.source = int(args.source) if args.source.isdigit() else args.source
        self.output_dir = args.output_dir
        self.interval = args.interval
        self.num_entities = args.num_entities
        self.threshold = args.threshold
        self.feature_dim = args.feature_dim
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.capture_manager = VideoCaptureManager()
        
        # Initialize Qwen-VL model
        print(f"Initializing Qwen-VL model ({args.model_version})...")
        self.feature_extractor = QwenVLFeatureExtractor(
            model_version=args.model_version,
            device=None
        )
        
        if not self.feature_extractor.is_initialized:
            print("Failed to initialize Qwen-VL model")
            exit(1)
        
        # Initialize RelationGraphBuilder
        print("Initializing RGA model...")
        self.relation_model = RelationGraphBuilder(
            feature_dim=self.feature_dim, 
            threshold=self.threshold
        )
        
        # Initialize SceneGraphBuilder
        self.graph_builder = SceneGraphBuilder(
            relation_model=self.relation_model,
            num_detections=self.num_entities
        )
        
        # UI state
        self.is_paused = False
        self.show_help = False
        self.frame_count = 0
        self.last_frame = None
        self.last_caption = None
        self.last_graph = None
        self.last_graph_vis = None
        self.processing_time = 0
        self.is_processing = False
    
    def connect_video_source(self):
        """Connect to the video source."""
        print(f"Connecting to video source: {self.source}")
        return self.capture_manager.open_source(self.source)
    
    def process_frame(self, frame):
        """
        Process a frame using RGA.
        
        Args:
            frame: Frame to process
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Generate caption with Qwen-VL
        caption = self.feature_extractor.generate_caption(frame)
        
        if caption:
            # Extract features
            features = self.feature_extractor.extract_features(frame)
            
            # Build graph from caption
            graph = self.graph_builder.build_scene_graph_from_caption(
                frame, caption, self.feature_extractor
            )
            
            # Visualize graph
            if graph:
                graph_vis = self.graph_builder.visualize_scene_graph(graph)
            else:
                graph_vis = None
        else:
            graph = None
            graph_vis = None
        
        # Calculate processing time
        self.processing_time = time.time() - start_time
        
        return {
            'caption': caption,
            'graph': graph,
            'graph_vis': graph_vis
        }
    
    def run_demo(self):
        """Run the RGA demo."""
        if not self.connect_video_source():
            print("Failed to connect to video source. Exiting.")
            return
        
        print(f"\nRunning RGA Demo with interval: {self.interval} frames")
        print("Controls:")
        print("  'q': Quit")
        print("  'p': Pause/Resume")
        print("  'h': Toggle help overlay")
        print("  's': Save current graph")
        print("  'space': Process current frame")
        
        # Create windows
        cv2.namedWindow('RGA Demo', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Scene Graph', cv2.WINDOW_NORMAL)
        
        # Main loop
        for _, (success, frame, frame_num) in enumerate(self.capture_manager.read_frames(validate=False)):
            if not success:
                print("Failed to read frame. Exiting.")
                break
            
            self.frame_count += 1
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Handle pause state
            if self.is_paused:
                key = self.handle_keypress(cv2.waitKey(50))
                if key == 27:  # ESC or 'q' to exit
                    break
                
                # Show the last processed result
                if self.last_caption is not None:
                    self.draw_result(display_frame, self.last_caption)
                
                # Display the frame and graph
                cv2.imshow('RGA Demo', display_frame)
                if self.last_graph_vis is not None:
                    cv2.imshow('Scene Graph', self.last_graph_vis)
                continue
            
            # Process frames at specified interval or when space is pressed
            process_this_frame = (self.frame_count % self.interval == 0) and not self.is_processing
            
            if process_this_frame:
                self.is_processing = True
                print(f"Processing frame #{frame_num}...")
                
                # Store a copy of the frame for saving
                self.last_frame = frame.copy()
                
                # Process the frame
                try:
                    result = self.process_frame(frame)
                    self.last_caption = result['caption']
                    self.last_graph = result['graph']
                    self.last_graph_vis = result['graph_vis']
                    
                    print(f"Caption: {self.last_caption}")
                    if self.last_graph:
                        print(f"Graph nodes: {len(self.last_graph['nodes'])}")
                        print(f"Graph edges: {len(self.last_graph['edges'])}")
                    print(f"Processing time: {self.processing_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    self.last_caption = f"Error: {str(e)}"
                    self.last_graph = None
                    self.last_graph_vis = None
                
                self.is_processing = False
            
            # Draw result on display frame
            if self.last_caption is not None:
                self.draw_result(display_frame, self.last_caption)
            
            # Display the frame and graph
            cv2.imshow('RGA Demo', display_frame)
            if self.last_graph_vis is not None:
                cv2.imshow('Scene Graph', self.last_graph_vis)
            
            # Handle keypress
            key = self.handle_keypress(cv2.waitKey(1))
            if key == 27:  # ESC or 'q' to exit
                break
        
        # Clean up
        self.cleanup()
    
    def draw_result(self, frame, caption):
        """
        Draw the processing result on the frame.
        
        Args:
            frame: Frame to draw on
            caption: Generated caption
        """
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add processing time
        cv2.putText(
            frame, 
            f"RGA Demo | Time: {self.processing_time:.2f}s", 
            (10, h - 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
        
        # Add caption
        if caption:
            # Wrap the text
            lines = self.wrap_text(caption, w - 20, font_scale=0.6)
            for i, line in enumerate(lines[:2]):  # Limit to 2 lines
                y = h - 40 + i * 25
                if y < h - 10:  # Prevent drawing outside frame
                    cv2.putText(
                        frame, line, (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
        
        # Processing indicator
        if self.is_processing:
            cv2.putText(
                frame,
                "PROCESSING...",
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Add help overlay if enabled
        if self.show_help:
            self.add_help_overlay(frame)
    
    def wrap_text(self, text, max_width, font_scale=0.6, thickness=1):
        """
        Wrap text to fit within a given width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font_scale: Font scale
            thickness: Line thickness
            
        Returns:
            List of wrapped text lines
        """
        if not text:
            return []
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Try adding the word to the current line
            test_line = ' '.join(current_line + [word])
            size = cv2.getTextSize(
                test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0]
            
            # If it fits, add it to the current line
            if size[0] <= max_width:
                current_line.append(word)
            # Otherwise, start a new line
            else:
                if current_line:  # Avoid empty lines
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def add_help_overlay(self, frame):
        """
        Add a help overlay to the frame.
        
        Args:
            frame: Frame to draw on
        """
        h, w = frame.shape[:2]
        
        # Create a semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add help text
        help_text = [
            "CONTROLS:",
            "q/ESC: Quit",
            "p: Pause/Resume",
            "h: Toggle help",
            "s: Save current graph",
            "space: Process current frame",
            f"Processing time: {self.processing_time:.2f}s"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(
                frame, 
                text, 
                (20, 40 + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                1
            )
    
    def save_current_result(self):
        """Save the current frame, caption, and graph."""
        if self.last_frame is None or self.last_caption is None or self.last_graph is None:
            print("No processed frame, caption, or graph to save")
            return
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame
        frame_path = f"{self.output_dir}/frame_{timestamp}.jpg"
        cv2.imwrite(frame_path, self.last_frame)
        
        # Save graph visualization if available
        if self.last_graph_vis is not None:
            graph_vis_path = f"{self.output_dir}/graph_{timestamp}.jpg"
            cv2.imwrite(graph_vis_path, self.last_graph_vis)
        
        # Save result to JSON file
        import json
        result_path = f"{self.output_dir}/result_{timestamp}.json"
        
        # Convert any numpy arrays to lists for JSON serialization
        graph_copy = {}
        if self.last_graph:
            graph_copy = {
                'nodes': self.last_graph['nodes'],
                'edges': self.last_graph['edges'],
                'caption': self.last_graph['caption']
            }
        
        with open(result_path, 'w') as f:
            json.dump({
                'caption': self.last_caption,
                'graph': graph_copy,
                'processing_time': self.processing_time
            }, f, indent=2)
        
        print(f"Saved frame to {frame_path}")
        if self.last_graph_vis is not None:
            print(f"Saved graph visualization to {graph_vis_path}")
        print(f"Saved result to {result_path}")
    
    def handle_keypress(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code
        
        Returns:
            27 to exit, otherwise the key code
        """
        if key == -1:  # No key pressed
            return key
        
        key &= 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            return 27
        
        elif key == ord('p'):  # Pause/Resume
            self.is_paused = not self.is_paused
            print("Playback " + ("paused" if self.is_paused else "resumed"))
        
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
        
        elif key == ord('s'):  # Save result
            self.save_current_result()
        
        elif key == 32:  # Space - process current frame
            if not self.is_processing:
                # Get current frame
                _, frame, _ = next(self.capture_manager.read_frames())
                self.last_frame = frame.copy()
                
                # Process the frame
                print("Processing current frame...")
                self.is_processing = True
                try:
                    result = self.process_frame(frame)
                    self.last_caption = result['caption']
                    self.last_graph = result['graph']
                    self.last_graph_vis = result['graph_vis']
                    
                    print(f"Caption: {self.last_caption}")
                    if self.last_graph:
                        print(f"Graph nodes: {len(self.last_graph['nodes'])}")
                        print(f"Graph edges: {len(self.last_graph['edges'])}")
                    print(f"Processing time: {self.processing_time:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    self.last_caption = f"Error: {str(e)}"
                    self.last_graph = None
                    self.last_graph_vis = None
                
                self.is_processing = False
        
        return key
    
    def cleanup(self):
        """Clean up resources."""
        self.capture_manager.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point for the demo."""
    args = parse_args()
    demo = RGADemo(args)
    demo.run_demo()


if __name__ == "__main__":
    main() 