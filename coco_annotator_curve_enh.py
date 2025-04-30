import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from PIL import Image, ImageTk
import numpy as np
from enum import Enum

class AnnotationMode(Enum):
    KEYPOINT = 1
    CURVE = 2
    SMOOTH_CURVE = 3  # New mode for smooth curves
    BBOX = 4

class CocoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("COCO Dataset Annotator")
        self.root.geometry("1200x800")
        
        self.dataset_path = None
        self.images = []
        self.current_image_index = -1
        self.current_image = None
        self.current_image_data = None
        self.current_annotations = {}
        
        self.annotation_mode = AnnotationMode.KEYPOINT
        self.drawing = False
        self.curve_points = []
        self.bbox_start = None
        
        self.keypoints = []
        self.curves = []
        self.smooth_curves = []  # New list for smooth curves
        self.bboxes = []
        
        # Smooth curve parameters
        self.smoothness = 0.3  # Controls the curve smoothness (0.0 to 1.0)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        # Load dataset button
        load_btn = ttk.Button(control_frame, text="Load Dataset", command=self.load_dataset)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=20)
        
        prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_image)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_counter = ttk.Label(nav_frame, text="0/0")
        self.image_counter.pack(side=tk.LEFT, padx=10)
        
        next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_image)
        next_btn.pack(side=tk.LEFT, padx=5)
        
        # Annotation mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Annotation Mode")
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        self.mode_var = tk.StringVar(value="keypoint")
        keypoint_rb = ttk.Radiobutton(mode_frame, text="Keypoint", variable=self.mode_var, 
                                      value="keypoint", command=self.set_annotation_mode)
        keypoint_rb.pack(side=tk.LEFT, padx=5)
        
        curve_rb = ttk.Radiobutton(mode_frame, text="Polyline", variable=self.mode_var, 
                                   value="curve", command=self.set_annotation_mode)
        curve_rb.pack(side=tk.LEFT, padx=5)
        
        # New smooth curve option
        smooth_curve_rb = ttk.Radiobutton(mode_frame, text="Smooth Curve", variable=self.mode_var, 
                                          value="smooth_curve", command=self.set_annotation_mode)
        smooth_curve_rb.pack(side=tk.LEFT, padx=5)
        
        bbox_rb = ttk.Radiobutton(mode_frame, text="Bounding Box", variable=self.mode_var, 
                                  value="bbox", command=self.set_annotation_mode)
        bbox_rb.pack(side=tk.LEFT, padx=5)
        
        # Curve smoothness control (only visible when smooth curve mode is active)
        self.smoothness_frame = ttk.Frame(control_frame)
        self.smoothness_frame.pack(side=tk.LEFT, padx=20)
        
        smoothness_label = ttk.Label(self.smoothness_frame, text="Smoothness:")
        smoothness_label.pack(side=tk.LEFT, padx=5)
        
        self.smoothness_var = tk.DoubleVar(value=0.3)
        smoothness_scale = ttk.Scale(self.smoothness_frame, from_=0.0, to=1.0, 
                                    variable=self.smoothness_var, length=100)
        smoothness_scale.pack(side=tk.LEFT, padx=5)
        smoothness_scale.bind("<ButtonRelease-1>", self.update_smoothness)
        
        # Save button
        save_btn = ttk.Button(control_frame, text="Save Annotations", command=self.save_annotations)
        save_btn.pack(side=tk.RIGHT, padx=5)

        # Remove YOLO export buttons
        # Canvas for image display and annotation
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Annotation list panel
        annotation_frame = ttk.LabelFrame(main_frame, text="Annotations")
        annotation_frame.pack(fill=tk.BOTH, side=tk.BOTTOM, expand=False, pady=5)
        
        # Create tabs for different annotation types
        self.annotation_tabs = ttk.Notebook(annotation_frame)
        self.annotation_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Keypoints tab
        keypoints_tab = ttk.Frame(self.annotation_tabs)
        self.annotation_tabs.add(keypoints_tab, text="Keypoints")
        
        self.keypoints_tree = ttk.Treeview(keypoints_tab, columns=("id", "x", "y"), 
                                          show="headings", selectmode="browse")
        self.keypoints_tree.heading("id", text="ID")
        self.keypoints_tree.heading("x", text="X")
        self.keypoints_tree.heading("y", text="Y")
        self.keypoints_tree.column("id", width=50)
        self.keypoints_tree.column("x", width=100)
        self.keypoints_tree.column("y", width=100)
        self.keypoints_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        kp_scrollbar = ttk.Scrollbar(keypoints_tab, orient=tk.VERTICAL, command=self.keypoints_tree.yview)
        kp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.keypoints_tree.configure(yscrollcommand=kp_scrollbar.set)
        
        delete_kp_btn = ttk.Button(keypoints_tab, text="Delete Selected", 
                                  command=lambda: self.delete_annotation("keypoint"))
        delete_kp_btn.pack(pady=5)
        
        # Polyline curves tab
        curves_tab = ttk.Frame(self.annotation_tabs)
        self.annotation_tabs.add(curves_tab, text="Polylines")
        
        self.curves_tree = ttk.Treeview(curves_tab, columns=("id", "points"), 
                                       show="headings", selectmode="browse")
        self.curves_tree.heading("id", text="ID")
        self.curves_tree.heading("points", text="Points")
        self.curves_tree.column("id", width=50)
        self.curves_tree.column("points", width=250)
        self.curves_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        curves_scrollbar = ttk.Scrollbar(curves_tab, orient=tk.VERTICAL, command=self.curves_tree.yview)
        curves_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.curves_tree.configure(yscrollcommand=curves_scrollbar.set)
        
        delete_curve_btn = ttk.Button(curves_tab, text="Delete Selected", 
                                     command=lambda: self.delete_annotation("curve"))
        delete_curve_btn.pack(pady=5)
        
        # Smooth curves tab
        smooth_curves_tab = ttk.Frame(self.annotation_tabs)
        self.annotation_tabs.add(smooth_curves_tab, text="Smooth Curves")
        
        self.smooth_curves_tree = ttk.Treeview(smooth_curves_tab, columns=("id", "points"), 
                                              show="headings", selectmode="browse")
        self.smooth_curves_tree.heading("id", text="ID")
        self.smooth_curves_tree.heading("points", text="Points")
        self.smooth_curves_tree.column("id", width=50)
        self.smooth_curves_tree.column("points", width=250)
        self.smooth_curves_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        smooth_curves_scrollbar = ttk.Scrollbar(smooth_curves_tab, orient=tk.VERTICAL, 
                                               command=self.smooth_curves_tree.yview)
        smooth_curves_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.smooth_curves_tree.configure(yscrollcommand=smooth_curves_scrollbar.set)
        
        delete_smooth_curve_btn = ttk.Button(smooth_curves_tab, text="Delete Selected", 
                                            command=lambda: self.delete_annotation("smooth_curve"))
        delete_smooth_curve_btn.pack(pady=5)
        
        # Bounding boxes tab
        bbox_tab = ttk.Frame(self.annotation_tabs)
        self.annotation_tabs.add(bbox_tab, text="Bounding Boxes")
        
        self.bbox_tree = ttk.Treeview(bbox_tab, columns=("id", "x1", "y1", "x2", "y2"), 
                                     show="headings", selectmode="browse")
        self.bbox_tree.heading("id", text="ID")
        self.bbox_tree.heading("x1", text="X1")
        self.bbox_tree.heading("y1", text="Y1")
        self.bbox_tree.heading("x2", text="X2")
        self.bbox_tree.heading("y2", text="Y2")
        self.bbox_tree.column("id", width=50)
        self.bbox_tree.column("x1", width=75)
        self.bbox_tree.column("y1", width=75)
        self.bbox_tree.column("x2", width=75)
        self.bbox_tree.column("y2", width=75)
        self.bbox_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        bbox_scrollbar = ttk.Scrollbar(bbox_tab, orient=tk.VERTICAL, command=self.bbox_tree.yview)
        bbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bbox_tree.configure(yscrollcommand=bbox_scrollbar.set)
        
        delete_bbox_btn = ttk.Button(bbox_tab, text="Delete Selected", 
                                    command=lambda: self.delete_annotation("bbox"))
        delete_bbox_btn.pack(pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initially hide smoothness control
        self.smoothness_frame.pack_forget()
    
    def load_dataset(self):
        """Load COCO dataset from a directory"""
        self.dataset_path = filedialog.askdirectory(title="Select COCO dataset directory")
        if not self.dataset_path:
            return
        
        # Look for images directory and annotations
        images_dir = os.path.join(self.dataset_path, "images")
        if not os.path.exists(images_dir):
            images_dir = self.dataset_path  # Fallback to selected directory
            
        # Get all image files
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.images.extend(
                [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                 if f.lower().endswith(ext)]
            )
        
        if not self.images:
            messagebox.showerror("Error", "No images found in the selected directory")
            return
        
        # Create annotation directory if it doesn't exist
        self.annotations_dir = os.path.join(self.dataset_path, "annotations")
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        self.images.sort()
        self.current_image_index = 0
        self.load_image()
        self.update_status(f"Loaded {len(self.images)} images")
    
    def load_image(self):
        """Load and display the current image"""
        if not self.images or self.current_image_index < 0:
            return
        
        # Clear existing annotations
        self.keypoints = []
        self.curves = []
        self.smooth_curves = []
        self.bboxes = []
        self.clear_annotation_lists()
        
        # Load image
        img_path = self.images[self.current_image_index]
        self.current_image_data = Image.open(img_path)
        
        # Resize if necessary to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 10 and canvas_height > 10:  # Ensure canvas has been rendered
            img_width, img_height = self.current_image_data.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            self.current_image_data = self.current_image_data.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter image and display
        self.current_image = ImageTk.PhotoImage(self.current_image_data)
        self.canvas.config(width=self.current_image_data.width, height=self.current_image_data.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
        # Update image counter
        self.image_counter.config(text=f"{self.current_image_index + 1}/{len(self.images)}")
        
        # Try to load existing annotations for this image
        self.load_annotations()
    
    def prev_image(self):
        """Go to the previous image"""
        if self.current_image_index > 0:
            self.prompt_save_annotations()
            self.current_image_index -= 1
            self.load_image()
    
    def next_image(self):
        """Go to the next image"""
        if self.current_image_index < len(self.images) - 1:
            self.prompt_save_annotations()
            self.current_image_index += 1
            self.load_image()
    
    def set_annotation_mode(self):
        """Set the current annotation mode based on radio button selection"""
        mode = self.mode_var.get()
        if mode == "keypoint":
            self.annotation_mode = AnnotationMode.KEYPOINT
            self.smoothness_frame.pack_forget()
        elif mode == "curve":
            self.annotation_mode = AnnotationMode.CURVE
            self.smoothness_frame.pack_forget()
        elif mode == "smooth_curve":
            self.annotation_mode = AnnotationMode.SMOOTH_CURVE
            self.smoothness_frame.pack(side=tk.LEFT, padx=20) # Show smoothness control
        elif mode == "bbox":
            self.annotation_mode = AnnotationMode.BBOX
            self.smoothness_frame.pack_forget()
        
        self.update_status(f"Annotation mode: {mode}")
    
    def update_smoothness(self, event=None):
        """Update the smoothness parameter and redraw temporary curves"""
        self.smoothness = self.smoothness_var.get()
        
        # If we're currently drawing a curve, update the preview
        if self.drawing and self.annotation_mode == AnnotationMode.SMOOTH_CURVE and len(self.curve_points) > 1:
            self.canvas.delete("temp_curve")
            # Draw temporary control points
            for x, y in self.curve_points:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="green", tags="temp_curve")
            
            # Draw temporary smooth curve
            if len(self.curve_points) >= 2:
                self.draw_smooth_curve_preview()
    
    def on_canvas_click(self, event):
        """Handle mouse click on the canvas"""
        if self.current_image is None:
            return
        
        x, y = event.x, event.y
        
        if self.annotation_mode == AnnotationMode.KEYPOINT:
            # Add keypoint
            keypoint_id = len(self.keypoints) + 1
            
            # Store in normalized coordinates
            img_width = self.current_image_data.width
            img_height = self.current_image_data.height
            x_norm = x / img_width
            y_norm = y / img_height
            
            # Save as (id, x_norm, y_norm, x_pixel, y_pixel)
            # We store both normalized and pixel coordinates for easy display
            keypoint = (keypoint_id, x_norm, y_norm, x, y)
            self.keypoints.append(keypoint)
            self.draw_keypoint(keypoint)
            self.update_keypoint_list()
            
        elif self.annotation_mode == AnnotationMode.CURVE:
            # Start or continue a polyline curve
            if not self.drawing:
                self.drawing = True
                self.curve_points = [(x, y)]
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="green", tags="temp_curve")
            else:
                self.curve_points.append((x, y))
                if len(self.curve_points) > 1:
                    p1 = self.curve_points[-2]
                    p2 = (x, y)
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], 
                                          fill="green", width=2, tags="temp_curve")
                    self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="green", tags="temp_curve")
        
        elif self.annotation_mode == AnnotationMode.SMOOTH_CURVE:
            # Start or continue a smooth curve
            if not self.drawing:
                self.drawing = True
                self.curve_points = [(x, y)]
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="purple", tags="temp_curve")
            else:
                self.curve_points.append((x, y))
                # Draw control point
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="purple", tags="temp_curve")
                
                # Draw smooth curve preview if we have at least 2 points
                if len(self.curve_points) >= 2:
                    self.draw_smooth_curve_preview()
        
        elif self.annotation_mode == AnnotationMode.BBOX:
            # Start bounding box
            self.drawing = True
            self.bbox_start = (x, y)
            
    def draw_smooth_curve_preview(self):
        """Draw a preview of the smooth curve while drawing"""
        if len(self.curve_points) < 2:
            return
        
        # Generate smooth curve points
        smooth_points = self.generate_smooth_curve(self.curve_points, self.smoothness)
        
        # Draw smooth curve segments
        for i in range(len(smooth_points) - 1):
            p1 = smooth_points[i]
            p2 = smooth_points[i+1]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], 
                                   fill="purple", width=2, tags="temp_curve")
    
    def on_canvas_drag(self, event):
        """Handle mouse drag on the canvas"""
        if not self.drawing or self.current_image is None:
            return
        
        x, y = event.x, event.y
        
        if self.annotation_mode == AnnotationMode.BBOX and self.bbox_start:
            # Update bounding box preview
            self.canvas.delete("temp_bbox")
            self.canvas.create_rectangle(
                self.bbox_start[0], self.bbox_start[1], x, y,
                outline="red", width=2, dash=(5, 5), tags="temp_bbox"
            )
    
    def on_canvas_release(self, event):
        """Handle mouse release on the canvas"""
        if not self.drawing or self.current_image is None:
            return
        
        x, y = event.x, event.y
        img_width = self.current_image_data.width
        img_height = self.current_image_data.height
        
        if self.annotation_mode == AnnotationMode.CURVE:
            # Double-click to end the curve
            if len(self.curve_points) > 1 and abs(x - self.curve_points[0][0]) < 10 and abs(y - self.curve_points[0][1]) < 10:
                # Close the curve
                self.canvas.delete("temp_curve")
                curve_id = len(self.curves) + 1
                
                # Convert all points to normalized coordinates
                # Store as (x_norm, y_norm, x_pixel, y_pixel) for each point
                normalized_points = []
                for px, py in self.curve_points:
                    x_norm = px / img_width
                    y_norm = py / img_height
                    normalized_points.append((x_norm, y_norm, px, py))
                
                # Add to curves list
                self.curves.append((curve_id, normalized_points))
                self.draw_curve((curve_id, normalized_points))
                self.update_curve_list()
                self.drawing = False
                self.curve_points = []
                
        elif self.annotation_mode == AnnotationMode.SMOOTH_CURVE:
            # Double-click to end the smooth curve
            if len(self.curve_points) > 1 and abs(x - self.curve_points[0][0]) < 10 and abs(y - self.curve_points[0][1]) < 10:
                # Close the smooth curve
                self.canvas.delete("temp_curve")
                curve_id = len(self.smooth_curves) + 1
                
                # Convert all points to normalized coordinates
                # Store as (x_norm, y_norm, x_pixel, y_pixel) for each point
                normalized_points = []
                for px, py in self.curve_points:
                    x_norm = px / img_width
                    y_norm = py / img_height
                    normalized_points.append((x_norm, y_norm, px, py))
                
                # Save both control points and smoothness parameter
                curve_data = (curve_id, normalized_points, self.smoothness)
                self.smooth_curves.append(curve_data)
                
                # Draw the final smooth curve
                self.draw_smooth_curve(curve_data)
                self.update_smooth_curve_list()
                self.drawing = False
                self.curve_points = []
        
        elif self.annotation_mode == AnnotationMode.BBOX and self.bbox_start:
            # Finish drawing the bounding box
            self.drawing = False
            self.canvas.delete("temp_bbox")
            
            # Ensure x1,y1 is top-left and x2,y2 is bottom-right
            x1 = min(self.bbox_start[0], x)
            y1 = min(self.bbox_start[1], y)
            x2 = max(self.bbox_start[0], x)
            y2 = max(self.bbox_start[1], y)
            
            # Ignore very small boxes (probably misclicks)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                bbox_id = len(self.bboxes) + 1
                
                # Convert to YOLO format (normalized coordinates)
                img_width = self.current_image_data.width
                img_height = self.current_image_data.height
                
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Store in YOLO format: (id, x_center, y_center, width, height)
                bbox = (bbox_id, x_center, y_center, width, height)
                self.bboxes.append(bbox)
                
                # Still draw in pixel coordinates for display
                self.draw_bbox(bbox, img_width, img_height)
                self.update_bbox_list()
            
            self.bbox_start = None
    
    def generate_smooth_curve(self, points, smoothness):
        """Generate points for a smooth curve using Catmull-Rom spline"""
        if len(points) < 2:
            return points
        
        # If we only have 2 points, return them
        if len(points) == 2:
            return points
        
        # For closed curves, add the first point at the end and the last point at the beginning
        if abs(points[0][0] - points[-1][0]) < 10 and abs(points[0][1] - points[-1][1]) < 10:
            # Close the curve by making the first and last points the same
            control_points = points[:-1]  # Remove the last point since it's a duplicate
            control_points.append(control_points[0])  # Add the first point to close the loop
            
            # Add extra control points for a closed curve
            extended_points = [control_points[-2]] + control_points + [control_points[1]]
        else:
            # For an open curve, duplicate the first and last points
            extended_points = [points[0]] + points + [points[-1]]
        
        # Number of segments to generate between each pair of control points
        num_segments = 10
        
        # Generate smooth curve points
        curve_points = []
        for i in range(1, len(extended_points) - 2):
            p0 = np.array(extended_points[i - 1])
            p1 = np.array(extended_points[i])
            p2 = np.array(extended_points[i + 1])
            p3 = np.array(extended_points[i + 2])
            
            # Generate points for this segment
            for t in range(num_segments + 1):
                t_normalized = t / num_segments
                
                # Catmull-Rom spline formula
                t2 = t_normalized * t_normalized
                t3 = t2 * t_normalized
                
                # Adjust tensioning (smoothness)
                s = 1.0 - smoothness
                
                # Calculate position using Catmull-Rom spline
                pos = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * s * t_normalized +
                    (2*p0 - 5*p1 + 4*p2 - p3) * s * t2 +
                    (-p0 + 3*p1 - 3*p2 + p3) * s * t3
                )
                
                # Add point to curve
                curve_points.append((int(pos[0]), int(pos[1])))
        
        return curve_points
    
    def draw_keypoint(self, keypoint):
        """Draw a keypoint on the canvas"""
        _, _, _, x, y = keypoint  # Use pixel coordinates for drawing
        keypoint_id = f"kp_{keypoint[0]}"
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", tags=keypoint_id)
        self.canvas.create_text(x, y-15, text=str(keypoint[0]), tags=keypoint_id)
    
    def draw_curve(self, curve):
        """Draw a polyline curve on the canvas"""
        curve_id, points = curve
        tag = f"curve_{curve_id}"
        
        if len(points) < 2:
            return
            
        # Draw line segments - use pixel coordinates for drawing
        for i in range(len(points) - 1):
            p1 = (points[i][2], points[i][3])  # Get pixel coordinates (x_pixel, y_pixel)
            p2 = (points[i+1][2], points[i+1][3])
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="green", width=2, tags=tag)
        
        # Close the curve
        p1 = (points[-1][2], points[-1][3])
        p2 = (points[0][2], points[0][3])
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="green", width=2, tags=tag)
        
        # Draw control points
        for point in points:
            x, y = point[2], point[3]  # Pixel coordinates for display
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="green", tags=tag)
        
        # Draw curve ID
        if points:
            self.canvas.create_text(points[0][2], points[0][3]-15, text=str(curve_id), tags=tag)
    
    def draw_smooth_curve(self, curve_data):
        """Draw a smooth curve on the canvas"""
        curve_id, control_points, smoothness = curve_data
        tag = f"smooth_curve_{curve_id}"
        
        if len(control_points) < 2:
            return
        
        # Draw control points - use pixel coordinates for display
        for point in control_points:
            x, y = point[2], point[3]  # Get pixel coordinates
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="purple", tags=tag)
        
        # Generate pixel coordinates for drawing
        pixel_points = [(p[2], p[3]) for p in control_points]
        
        # Generate and draw the smooth curve
        smooth_points = self.generate_smooth_curve(pixel_points, smoothness)
        
        for i in range(len(smooth_points) - 1):
            p1 = smooth_points[i]
            p2 = smooth_points[i+1]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="purple", width=2, tags=tag)
            
        # Draw curve ID
        if control_points:
            x, y = control_points[0][2], control_points[0][3]  # Pixel coordinates
            self.canvas.create_text(x, y-15, text=str(curve_id), tags=tag)
    
    def draw_bbox(self, bbox, img_width=None, img_height=None):
        """Draw a bounding box on the canvas"""
        bbox_id, x_center, y_center, width, height = bbox
        tag = f"bbox_{bbox_id}"
        
        # If image dimensions not provided, get them from current image
        if img_width is None:
            img_width = self.current_image_data.width
        if img_height is None:
            img_height = self.current_image_data.height
        
        # Convert from YOLO format to pixel coordinates for drawing
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags=tag)
        self.canvas.create_text(x1, y1-10, text=str(bbox_id), tags=tag)
    
    def update_keypoint_list(self):
        """Update the keypoints list in the UI"""
        # Clear previous entries
        for item in self.keypoints_tree.get_children():
            self.keypoints_tree.delete(item)
        
        # Add keypoints to the tree
        for kp_id, x_norm, y_norm, x, y in self.keypoints:
            # Display pixel coordinates to the user
            self.keypoints_tree.insert("", "end", values=(kp_id, x, y))
    
    def update_curve_list(self):
        """Update the curves list in the UI"""
        # Clear previous entries
        for item in self.curves_tree.get_children():
            self.curves_tree.delete(item)
        
        # Add curves to the tree
        for curve_id, points in self.curves:
            points_str = f"{len(points)} points"
            self.curves_tree.insert("", "end", values=(curve_id, points_str))
    
    def update_smooth_curve_list(self):
        """Update the smooth curves list in the UI"""
        # Clear previous entries
        for item in self.smooth_curves_tree.get_children():
            self.smooth_curves_tree.delete(item)
        
        # Add curves to the tree
        for curve_id, points, smoothness in self.smooth_curves:
            points_str = f"{len(points)} points (smoothness: {smoothness:.2f})"
            self.smooth_curves_tree.insert("", "end", values=(curve_id, points_str))
    
    def update_bbox_list(self):
        """Update the bounding boxes list in the UI"""
        # Clear previous entries
        for item in self.bbox_tree.get_children():
            self.bbox_tree.delete(item)
        
        img_width = self.current_image_data.width
        img_height = self.current_image_data.height
        
        # Add bounding boxes to the tree - show in pixel coordinates for user readability
        for bbox_id, x_center, y_center, width, height in self.bboxes:
            # Convert normalized coordinates to pixel coordinates for display
            x1 = int((x_center - width/2) * img_width)
            y1 = int((y_center - height/2) * img_height)
            x2 = int((x_center + width/2) * img_width)
            y2 = int((y_center + height/2) * img_height)
            
            self.bbox_tree.insert("", "end", values=(bbox_id, x1, y1, x2, y2))
    
    def clear_annotation_lists(self):
        """Clear all annotation lists in the UI"""
        for item in self.keypoints_tree.get_children():
            self.keypoints_tree.delete(item)
        for item in self.curves_tree.get_children():
            self.curves_tree.delete(item)
        for item in self.smooth_curves_tree.get_children():
            self.smooth_curves_tree.delete(item)
        for item in self.bbox_tree.get_children():
            self.bbox_tree.delete(item)
    
    def delete_annotation(self, annotation_type):
        """Delete the selected annotation"""
        if annotation_type == "keypoint":
            selected = self.keypoints_tree.selection()
            if selected:
                idx = self.keypoints_tree.index(selected[0])
                if 0 <= idx < len(self.keypoints):
                    kp_id = self.keypoints[idx][0]
                    self.canvas.delete(f"kp_{kp_id}")
                    self.keypoints.pop(idx)
                    self.update_keypoint_list()
        
        elif annotation_type == "curve":
            selected = self.curves_tree.selection()
            if selected:
                idx = self.curves_tree.index(selected[0])
                if 0 <= idx < len(self.curves):
                    curve_id = self.curves[idx][0]
                    self.canvas.delete(f"curve_{curve_id}")
                    self.curves.pop(idx)
                    self.update_curve_list()
        
        elif annotation_type == "smooth_curve":
            selected = self.smooth_curves_tree.selection()
            if selected:
                idx = self.smooth_curves_tree.index(selected[0])
                if 0 <= idx < len(self.smooth_curves):
                    curve_id = self.smooth_curves[idx][0]
                    self.canvas.delete(f"smooth_curve_{curve_id}")
                    self.smooth_curves.pop(idx)
                    self.update_smooth_curve_list()
        
        elif annotation_type == "bbox":
            selected = self.bbox_tree.selection()
            if selected:
                idx = self.bbox_tree.index(selected[0])
                if 0 <= idx < len(self.bboxes):
                    bbox_id = self.bboxes[idx][0]
                    self.canvas.delete(f"bbox_{bbox_id}")
                    self.bboxes.pop(idx)
                    self.update_bbox_list()
    
    def load_annotations(self):
        """Load annotations for the current image"""
        if not self.current_image_index >= 0 or not self.images:
            return
            
        img_path = self.images[self.current_image_index]
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]
        
        # Check for annotations in the annotations directory
        json_filename = f"{base_filename}_annotations.json"
        json_path = os.path.join(self.annotations_dir, json_filename)
        
        # If not found in annotations directory, check original location
        if not os.path.exists(json_path):
            old_json_path = os.path.join(os.path.dirname(img_path), json_filename)
            if os.path.exists(old_json_path):
                # Move to the new location
                import shutil
                shutil.copy2(old_json_path, json_path)
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                img_width = self.current_image_data.width
                img_height = self.current_image_data.height
                
                # Load keypoints
                self.keypoints = []
                for kp in data.get('keypoints', []):
                    # Handle both old and new format
                    if 'x_norm' in kp:
                        # New format with normalized coordinates
                        kp_id = kp['id']
                        x_norm = kp['x_norm']
                        y_norm = kp['y_norm']
                        # Convert to pixel coordinates for display
                        x = int(x_norm * img_width)
                        y = int(y_norm * img_height)
                        keypoint = (kp_id, x_norm, y_norm, x, y)
                    else:
                        # Old format with pixel coordinates
                        kp_id = kp['id']
                        x = kp['x']
                        y = kp['y']
                        # Calculate normalized coordinates
                        x_norm = x / img_width
                        y_norm = y / img_height
                        keypoint = (kp_id, x_norm, y_norm, x, y)
                        
                    self.keypoints.append(keypoint)
                    self.draw_keypoint(keypoint)
                
                # Load curves
                self.curves = []
                for c in data.get('curves', []):
                    curve_id = c['id']
                    
                    if 'normalized_points' in c:
                        # New format with normalized points
                        norm_points = c['normalized_points']
                        points = []
                        for p in norm_points:
                            x_norm = p['x_norm']
                            y_norm = p['y_norm']
                            # Convert to pixel coordinates for display
                            x = int(x_norm * img_width)
                            y = int(y_norm * img_height)
                            points.append((x_norm, y_norm, x, y))
                    else:
                        # Old format with pixel points
                        points = []
                        for p in c['points']:
                            x = p['x']
                            y = p['y']
                            # Calculate normalized coordinates
                            x_norm = x / img_width
                            y_norm = y / img_height
                            points.append((x_norm, y_norm, x, y))
                            
                    curve_data = (curve_id, points)
                    self.curves.append(curve_data)
                    self.draw_curve(curve_data)
                
                # Load smooth curves
                self.smooth_curves = []
                for sc in data.get('smooth_curves', []):
                    curve_id = sc['id']
                    smoothness = sc['smoothness']
                    
                    if 'normalized_points' in sc:
                        # New format with normalized points
                        norm_points = sc['normalized_points']
                        points = []
                        for p in norm_points:
                            x_norm = p['x_norm']
                            y_norm = p['y_norm']
                            # Convert to pixel coordinates for display
                            x = int(x_norm * img_width)
                            y = int(y_norm * img_height)
                            points.append((x_norm, y_norm, x, y))
                    else:
                        # Old format with pixel points
                        points = []
                        for p in sc['points']:
                            x = p['x']
                            y = p['y']
                            # Calculate normalized coordinates
                            x_norm = x / img_width
                            y_norm = y / img_height
                            points.append((x_norm, y_norm, x, y))
                    
                    curve_data = (curve_id, points, smoothness)
                    self.smooth_curves.append(curve_data)
                    self.draw_smooth_curve(curve_data)
                
                # Load bounding boxes (support both old format and YOLO format)
                self.bboxes = []
                img_width = self.current_image_data.width
                img_height = self.current_image_data.height
                
                for bb in data.get('bboxes', []):
                    if 'x1' in bb:  # Old format with pixel coordinates
                        # Convert to YOLO format
                        x1, y1 = bb['x1'], bb['y1']
                        x2, y2 = bb['x2'], bb['y2']
                        x_center = (x1 + x2) / (2 * img_width)
                        y_center = (y1 + y2) / (2 * img_height)
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        bbox_data = (bb['id'], x_center, y_center, width, height)
                    else:  # New YOLO format
                        bbox_data = (bb['id'], bb['x_center'], bb['y_center'], bb['width'], bb['height'])
                    
                    self.bboxes.append(bbox_data)
                    self.draw_bbox(bbox_data, img_width, img_height)
                
                # Update UI lists
                self.update_keypoint_list()
                self.update_curve_list()
                self.update_smooth_curve_list()
                self.update_bbox_list()
                
                self.update_status(f"Loaded annotations from {json_filename}")
                
            except Exception as e:
                self.update_status(f"Error loading annotations: {str(e)}")
    
    def save_annotations(self):
        """Save annotations for the current image"""
        if not self.current_image_index >= 0 or not self.images:
            return
            
        img_path = self.images[self.current_image_index]
        img_filename = os.path.basename(img_path)
        base_filename = os.path.splitext(img_filename)[0]
        
        # Save JSON format in the annotations directory
        json_filename = f"{base_filename}_annotations.json"
        json_path = os.path.join(self.annotations_dir, json_filename)
        
        # Prepare data to save
        data = {
            'image_filename': img_filename,
            'keypoints': [{'id': kp[0], 'x_norm': kp[1], 'y_norm': kp[2]} for kp in self.keypoints],
            'curves': [{'id': c[0], 
                       'normalized_points': [{'x_norm': p[0], 'y_norm': p[1]} for p in c[1]]} 
                      for c in self.curves],
            'smooth_curves': [{'id': sc[0], 
                              'normalized_points': [{'x_norm': p[0], 'y_norm': p[1]} for p in sc[1]], 
                              'smoothness': sc[2]} 
                             for sc in self.smooth_curves],
            'bboxes': [{'id': bb[0], 'x_center': bb[1], 'y_center': bb[2], 'width': bb[3], 'height': bb[4]} 
                      for bb in self.bboxes]
        }
        
        try:
            # Save JSON with all annotations
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.update_status(f"Saved annotations to {json_filename}")
            return True
        
        except Exception as e:
            self.update_status(f"Error saving annotations: {str(e)}")
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
            return False
    
    def prompt_save_annotations(self):
        """Prompt the user to save annotations before moving to another image"""
        if any([self.keypoints, self.curves, self.smooth_curves, self.bboxes]):
            if messagebox.askyesno("Save Annotations", "Save annotations for the current image?"):
                self.save_annotations()
    
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_bar.config(text=message)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CocoAnnotator(root)
    root.mainloop()