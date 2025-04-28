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
    BBOX = 3

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
        self.bboxes = []
        
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
        
        curve_rb = ttk.Radiobutton(mode_frame, text="Curve", variable=self.mode_var, 
                                   value="curve", command=self.set_annotation_mode)
        curve_rb.pack(side=tk.LEFT, padx=5)
        
        bbox_rb = ttk.Radiobutton(mode_frame, text="Bounding Box", variable=self.mode_var, 
                                  value="bbox", command=self.set_annotation_mode)
        bbox_rb.pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_btn = ttk.Button(control_frame, text="Save Annotations", command=self.save_annotations)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
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
        
        # Curves tab
        curves_tab = ttk.Frame(self.annotation_tabs)
        self.annotation_tabs.add(curves_tab, text="Curves")
        
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
        elif mode == "curve":
            self.annotation_mode = AnnotationMode.CURVE
        elif mode == "bbox":
            self.annotation_mode = AnnotationMode.BBOX
        
        self.update_status(f"Annotation mode: {mode}")
    
    def on_canvas_click(self, event):
        """Handle mouse click on the canvas"""
        if self.current_image is None:
            return
        
        x, y = event.x, event.y
        
        if self.annotation_mode == AnnotationMode.KEYPOINT:
            # Add keypoint
            keypoint_id = len(self.keypoints) + 1
            keypoint = (keypoint_id, x, y)
            self.keypoints.append(keypoint)
            self.draw_keypoint(keypoint)
            self.update_keypoint_list()
            
        elif self.annotation_mode == AnnotationMode.CURVE:
            # Start or continue a curve
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
        
        elif self.annotation_mode == AnnotationMode.BBOX:
            # Start bounding box
            self.drawing = True
            self.bbox_start = (x, y)
            
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
        
        if self.annotation_mode == AnnotationMode.CURVE:
            # Double-click to end the curve
            if len(self.curve_points) > 1 and abs(x - self.curve_points[0][0]) < 10 and abs(y - self.curve_points[0][1]) < 10:
                # Close the curve
                self.canvas.delete("temp_curve")
                curve_id = len(self.curves) + 1
                self.curves.append((curve_id, self.curve_points[:]))
                self.draw_curve((curve_id, self.curve_points))
                self.update_curve_list()
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
                bbox = (bbox_id, x1, y1, x2, y2)
                self.bboxes.append(bbox)
                self.draw_bbox(bbox)
                self.update_bbox_list()
            
            self.bbox_start = None
    
    def draw_keypoint(self, keypoint):
        """Draw a keypoint on the canvas"""
        _, x, y = keypoint
        keypoint_id = f"kp_{keypoint[0]}"
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", tags=keypoint_id)
        self.canvas.create_text(x, y-15, text=str(keypoint[0]), tags=keypoint_id)
    
    def draw_curve(self, curve):
        """Draw a curve on the canvas"""
        curve_id, points = curve
        tag = f"curve_{curve_id}"
        
        if len(points) < 2:
            return
            
        # Draw line segments
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="green", width=2, tags=tag)
        
        # Close the curve
        p1 = points[-1]
        p2 = points[0]
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="green", width=2, tags=tag)
        
        # Draw curve ID
        if points:
            self.canvas.create_text(points[0][0], points[0][1]-15, text=str(curve_id), tags=tag)
    
    def draw_bbox(self, bbox):
        """Draw a bounding box on the canvas"""
        bbox_id, x1, y1, x2, y2 = bbox
        tag = f"bbox_{bbox_id}"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, tags=tag)
        self.canvas.create_text(x1, y1-10, text=str(bbox_id), tags=tag, anchor="w")
    
    def update_keypoint_list(self):
        """Update the keypoints in the treeview"""
        self.keypoints_tree.delete(*self.keypoints_tree.get_children())
        for kp in self.keypoints:
            kp_id, x, y = kp
            self.keypoints_tree.insert("", "end", values=(kp_id, x, y))
    
    def update_curve_list(self):
        """Update the curves in the treeview"""
        self.curves_tree.delete(*self.curves_tree.get_children())
        for curve in self.curves:
            curve_id, points = curve
            points_str = f"{len(points)} points"
            self.curves_tree.insert("", "end", values=(curve_id, points_str))
    
    def update_bbox_list(self):
        """Update the bounding boxes in the treeview"""
        self.bbox_tree.delete(*self.bbox_tree.get_children())
        for bbox in self.bboxes:
            bbox_id, x1, y1, x2, y2 = bbox
            self.bbox_tree.insert("", "end", values=(bbox_id, x1, y1, x2, y2))
    
    def clear_annotation_lists(self):
        """Clear all annotation lists"""
        self.keypoints_tree.delete(*self.keypoints_tree.get_children())
        self.curves_tree.delete(*self.curves_tree.get_children())
        self.bbox_tree.delete(*self.bbox_tree.get_children())
    
    def delete_annotation(self, annotation_type):
        """Delete the selected annotation"""
        if annotation_type == "keypoint":
            selected = self.keypoints_tree.selection()
            if selected:
                idx = self.keypoints_tree.index(selected[0])
                if 0 <= idx < len(self.keypoints):
                    kp_id = self.keypoints[idx][0]
                    self.canvas.delete(f"kp_{kp_id}")
                    del self.keypoints[idx]
                    self.update_keypoint_list()
        
        elif annotation_type == "curve":
            selected = self.curves_tree.selection()
            if selected:
                idx = self.curves_tree.index(selected[0])
                if 0 <= idx < len(self.curves):
                    curve_id = self.curves[idx][0]
                    self.canvas.delete(f"curve_{curve_id}")
                    del self.curves[idx]
                    self.update_curve_list()
        
        elif annotation_type == "bbox":
            selected = self.bbox_tree.selection()
            if selected:
                idx = self.bbox_tree.index(selected[0])
                if 0 <= idx < len(self.bboxes):
                    bbox_id = self.bboxes[idx][0]
                    self.canvas.delete(f"bbox_{bbox_id}")
                    del self.bboxes[idx]
                    self.update_bbox_list()
    
    def get_annotation_filename(self):
        """Get the annotation filename for the current image"""
        if not self.current_image_index >= 0 or not self.images:
            return None
            
        img_path = self.images[self.current_image_index]
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Create annotations directory if it doesn't exist
        annotations_dir = os.path.join(self.dataset_path, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        return os.path.join(annotations_dir, f"{base_name}.json")
    
    def load_annotations(self):
        """Try to load existing annotations for the current image"""
        annotation_file = self.get_annotation_filename()
        if not annotation_file or not os.path.exists(annotation_file):
            return
            
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                
            # Load keypoints
            if 'keypoints' in data:
                self.keypoints = []
                for kp in data['keypoints']:
                    keypoint = (kp['id'], kp['x'], kp['y'])
                    self.keypoints.append(keypoint)
                    self.draw_keypoint(keypoint)
                self.update_keypoint_list()
                
            # Load curves
            if 'curves' in data:
                self.curves = []
                for curve in data['curves']:
                    curve_data = (curve['id'], [(p['x'], p['y']) for p in curve['points']])
                    self.curves.append(curve_data)
                    self.draw_curve(curve_data)
                self.update_curve_list()
                
            # Load bounding boxes
            if 'bboxes' in data:
                self.bboxes = []
                for bbox in data['bboxes']:
                    bbox_data = (bbox['id'], bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
                    self.bboxes.append(bbox_data)
                    self.draw_bbox(bbox_data)
                self.update_bbox_list()
                
            self.update_status(f"Loaded annotations for {os.path.basename(self.images[self.current_image_index])}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotations: {str(e)}")
    
    def save_annotations(self):
        """Save annotations for the current image"""
        if self.current_image_index < 0 or not self.images:
            messagebox.showinfo("Info", "No image loaded")
            return
            
        annotation_data = {
            'image': os.path.basename(self.images[self.current_image_index]),
            'keypoints': [{'id': kp[0], 'x': kp[1], 'y': kp[2]} for kp in self.keypoints],
            'curves': [{'id': c[0], 'points': [{'x': p[0], 'y': p[1]} for p in c[1]]} for c in self.curves],
            'bboxes': [{'id': bb[0], 'x1': bb[1], 'y1': bb[2], 'x2': bb[3], 'y2': bb[4]} for bb in self.bboxes]
        }
        
        annotation_file = self.get_annotation_filename()
        if not annotation_file:
            return
            
        try:
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
                
            self.update_status(f"Saved annotations to {os.path.basename(annotation_file)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def prompt_save_annotations(self):
        """Prompt user to save annotations before switching images"""
        if self.keypoints or self.curves or self.bboxes:
            if messagebox.askyesno("Save Annotations", 
                                  "Do you want to save the current annotations before continuing?"):
                self.save_annotations()
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_bar.config(text=message)

def main():
    root = tk.Tk()
    app = CocoAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
