import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pywt
from datetime import datetime

class DualImageViewer:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Dual Image Viewer")
        self.top.geometry("1200x750")
        
        # Register for two images
        self.images = [None, None]  # [left, right]
        self.image_paths = [None, None]
        self.wavelet = None
        self.level = None
        self.available_paths = {}
        self.path_list = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Toolbar in multiple rows
        toolbar_frame = tk.Frame(self.top)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Row 1: Main operations
        row1 = tk.Frame(toolbar_frame)
        row1.pack(fill=tk.X, pady=2)
        
        tk.Label(row1, text="Operations:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row1, text="A - B → A", command=lambda: self.compute_difference('AminusB'),
                 bg="lightcoral", width=12).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1, text="B - A → B", command=lambda: self.compute_difference('BminusA'),
                 bg="lightcoral", width=12).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1, text="Swap A ↔ B", command=self.swap_images,
                 bg="lightblue", width=12).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1, text="Clear All", command=self.clear_all,
                 bg="lightgray", width=12).pack(side=tk.LEFT, padx=2)
        
        # Row 2: Transformations
        row2 = tk.Frame(toolbar_frame)
        row2.pack(fill=tk.X, pady=2)
        
        tk.Label(row2, text="Transform:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row2, text="↻ 90°", command=lambda: self.transform_image('rotate_90'),
                 bg="lightyellow", width=8).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="↺ 90°", command=lambda: self.transform_image('rotate_270'),
                 bg="lightyellow", width=8).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="180°", command=lambda: self.transform_image('rotate_180'),
                 bg="lightyellow", width=8).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="↔ Mirror H", command=lambda: self.transform_image('mirror_h'),
                 bg="lightgreen", width=10).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="↕ Mirror V", command=lambda: self.transform_image('mirror_v'),
                 bg="lightgreen", width=10).pack(side=tk.LEFT, padx=2)
        
        # Row 3: Shift controls and target selection
        row3 = tk.Frame(toolbar_frame)
        row3.pack(fill=tk.X, pady=2)
        
        tk.Label(row3, text="Shift:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.shift_entry = tk.Entry(row3, width=3)
        self.shift_entry.pack(side=tk.LEFT, padx=2)
        self.shift_entry.insert(0, "1")
        
        tk.Button(row3, text="←", command=lambda: self.transform_image('shift_left'),
                 bg="lightpink", width=3).pack(side=tk.LEFT, padx=1)
        
        tk.Button(row3, text="→", command=lambda: self.transform_image('shift_right'),
                 bg="lightpink", width=3).pack(side=tk.LEFT, padx=1)
        
        tk.Button(row3, text="↑", command=lambda: self.transform_image('shift_up'),
                 bg="lightpink", width=3).pack(side=tk.LEFT, padx=1)
        
        tk.Button(row3, text="↓", command=lambda: self.transform_image('shift_down'),
                 bg="lightpink", width=3).pack(side=tk.LEFT, padx=1)
        
        tk.Label(row3, text="  Apply to:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20,2))
        
        self.target_var = tk.StringVar(value="A")
        tk.Radiobutton(row3, text="A", variable=self.target_var, value="A").pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(row3, text="B", variable=self.target_var, value="B").pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(row3, text="Both", variable=self.target_var, value="Both").pack(side=tk.LEFT, padx=2)
        
        # Row 4: New row for grid button
        row4 = tk.Frame(toolbar_frame)
        row4.pack(fill=tk.X, pady=2)
        
        tk.Label(row4, text="Grid:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.grid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Show grid", variable=self.grid_var, 
                      command=self.toggle_grid, bg="lightgreen").pack(side=tk.LEFT, padx=2)
        
        # Row 5: Load controls (бывшая Row 4)
        row5 = tk.Frame(toolbar_frame)
        row5.pack(fill=tk.X, pady=5)
        
        tk.Label(row5, text="Load Image #:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.load_entry = tk.Entry(row5, width=6)
        self.load_entry.pack(side=tk.LEFT, padx=2)
        self.load_entry.bind('<Return>', self.on_load_enter)
        
        tk.Button(row5, text="→ A", command=lambda: self.load_image_to_slot(0),
                 bg="lightblue", width=8).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row5, text="→ B", command=lambda: self.load_image_to_slot(1),
                 bg="lightblue", width=8).pack(side=tk.LEFT, padx=2)
        
        self.load_info = tk.Label(row5, text="", font=("Arial", 9), fg="green")
        self.load_info.pack(side=tk.LEFT, padx=10)
        
        # Main content area
        main_frame = tk.Frame(self.top)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left and right panels
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Left image frame
        left_frame = tk.LabelFrame(left_panel, text="Image A", font=("Arial", 12, "bold"))
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        self.left_figure = Figure(figsize=(5, 4), dpi=100)
        self.left_canvas = FigureCanvasTkAgg(self.left_figure, master=left_frame)
        self.left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add navigation toolbar for each canvas
        left_toolbar = NavigationToolbar2Tk(self.left_canvas, left_frame)
        left_toolbar.update()
        
        left_info = tk.Frame(left_frame)
        left_info.pack(fill=tk.X, padx=5, pady=2)
        
        self.left_path_label = tk.Label(left_info, text="Path: None", font=("Arial", 8))
        self.left_path_label.pack(side=tk.LEFT)
        
        self.left_stats_label = tk.Label(left_info, text="", font=("Arial", 8), fg="gray")
        self.left_stats_label.pack(side=tk.RIGHT)
        
        # Right image frame
        right_frame = tk.LabelFrame(right_panel, text="Image B", font=("Arial", 12, "bold"))
        right_frame.pack(fill=tk.BOTH, expand=True)
        
        self.right_figure = Figure(figsize=(5, 4), dpi=100)
        self.right_canvas = FigureCanvasTkAgg(self.right_figure, master=right_frame)
        self.right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_toolbar = NavigationToolbar2Tk(self.right_canvas, right_frame)
        right_toolbar.update()
        
        right_info = tk.Frame(right_frame)
        right_info.pack(fill=tk.X, padx=5, pady=2)
        
        self.right_path_label = tk.Label(right_info, text="Path: None", font=("Arial", 8))
        self.right_path_label.pack(side=tk.LEFT)
        
        self.right_stats_label = tk.Label(right_info, text="", font=("Arial", 8), fg="gray")
        self.right_stats_label.pack(side=tk.RIGHT)
        
        # Status bar at bottom
        self.status_label = tk.Label(self.top, text="Enter image number and click →A or →B to load",
                                     font=("Arial", 9), fg="blue", bg="lightyellow", pady=2)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_grid(self):
        """Toggle grid display for both images"""
        self.update_display(0)
        self.update_display(1)

    def draw_grid(self, ax, matrix):
        """Draw green grid on the image (straight cross and diagonal cross)"""
        if not self.grid_var.get():
            return
        
        rows, cols = matrix.shape
        
        # Draw straight cross (horizontal and vertical lines through center)
        center_row = rows // 2
        center_col = cols // 2
        
        # Horizontal line
        ax.axhline(y=center_row - 0.5, xmin=0, xmax=1, color='green', linewidth=1, alpha=0.7)
        # Vertical line
        ax.axvline(x=center_col - 0.5, ymin=0, ymax=1, color='green', linewidth=1, alpha=0.7)
        
        # Draw diagonal cross
        # Main diagonal (top-left to bottom-right)
        ax.plot([-0.5, cols - 0.5], [-0.5, rows - 0.5], 
                color='green', linewidth=1, alpha=0.7, linestyle='-')
        
        # Anti-diagonal (top-right to bottom-left)
        ax.plot([cols - 0.5, -0.5], [-0.5, rows - 0.5], 
                color='green', linewidth=1, alpha=0.7, linestyle='-')
       
    def set_paths(self, paths):
        """Set available paths from main window"""
        self.available_paths = paths
        self.path_list = list(paths.keys())
        self.status_label.config(text=f"Available: 1-{len(self.path_list)} images. Enter number and load.")
        
    def on_load_enter(self, event):
        """Handle Enter key in load entry"""
        self.load_image_to_slot(0)  # Default to A on Enter
        
    def load_image_to_slot(self, slot):
        """Load image number to specified slot"""
        try:
            num = int(self.load_entry.get())
            if 1 <= num <= len(self.path_list):
                path = self.path_list[num-1]
                matrix = self.available_paths[path]
                
                self.images[slot] = matrix.copy()
                self.image_paths[slot] = path
                self.update_display(slot)
                
                slot_name = "A" if slot == 0 else "B"
                self.load_info.config(text=f"Loaded #{num} → {slot_name}", fg="green")
                self.load_entry.delete(0, tk.END)
            else:
                self.load_info.config(text=f"Number must be 1-{len(self.path_list)}", fg="red")
        except ValueError:
            self.load_info.config(text="Invalid number", fg="red")
            
    def update_display(self, slot):
        """Update display for specified slot with grid if enabled"""
        if slot == 0:
            fig = self.left_figure
            canvas = self.left_canvas
            path_label = self.left_path_label
            stats_label = self.left_stats_label
            title = "Image A"
        else:
            fig = self.right_figure
            canvas = self.right_canvas
            path_label = self.right_path_label
            stats_label = self.right_stats_label
            title = "Image B"
            
        fig.clear()
        
        if self.images[slot] is not None:
            ax = fig.add_subplot(111)
            im = ax.imshow(self.images[slot], cmap='gray', aspect='equal')
            
            # Draw grid if enabled
            self.draw_grid(ax, self.images[slot])
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}: {self.image_paths[slot]}", fontsize=9)
            
            max_val = np.max(np.abs(self.images[slot]))
            if max_val > 0:
                im.set_clim(-max_val, max_val)
            
            fig.colorbar(im, ax=ax)
            
            # Update stats
            stats = f"Min: {np.min(self.images[slot]):.3f} | Max: {np.max(self.images[slot]):.3f} | Mean: {np.mean(self.images[slot]):.3f}"
            stats_label.config(text=stats)
            path_label.config(text=f"Path: {self.image_paths[slot]}")
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            stats_label.config(text="")
            path_label.config(text="Path: None")
            
        canvas.draw()
        
    def transform_image(self, transform_type):
        """Apply transformation to selected target(s)"""
        target = self.target_var.get()
        
        try:
            steps = int(self.shift_entry.get())
        except:
            steps = 1
        
        slots = []
        if target == "A":
            slots = [0]
        elif target == "B":
            slots = [1]
        else:  # Both
            slots = [0, 1]
            
        for slot in slots:
            if self.images[slot] is None:
                continue
                
            if transform_type == 'rotate_90':
                self.images[slot] = np.rot90(self.images[slot], k=-1)
            elif transform_type == 'rotate_270':
                self.images[slot] = np.rot90(self.images[slot], k=1)
            elif transform_type == 'rotate_180':
                self.images[slot] = np.rot90(self.images[slot], k=2)
            elif transform_type == 'mirror_h':
                self.images[slot] = np.fliplr(self.images[slot])
            elif transform_type == 'mirror_v':
                self.images[slot] = np.flipud(self.images[slot])
            elif transform_type == 'shift_left':
                self.images[slot] = np.roll(self.images[slot], shift=-steps, axis=1)
            elif transform_type == 'shift_right':
                self.images[slot] = np.roll(self.images[slot], shift=steps, axis=1)
            elif transform_type == 'shift_up':
                self.images[slot] = np.roll(self.images[slot], shift=-steps, axis=0)
            elif transform_type == 'shift_down':
                self.images[slot] = np.roll(self.images[slot], shift=steps, axis=0)
                
            self.update_display(slot)
        
    def compute_difference(self, mode):
        """Compute difference between images"""
        if self.images[0] is None or self.images[1] is None:
            messagebox.showwarning("Warning", "Both images must be loaded")
            return
            
        if mode == 'AminusB':
            # A - B → A
            self.images[0] = self.images[0] - self.images[1]
            self.image_paths[0] = f"({self.image_paths[0]}) - ({self.image_paths[1]})"
            self.update_display(0)
        else:
            # B - A → B
            self.images[1] = self.images[1] - self.images[0]
            self.image_paths[1] = f"({self.image_paths[1]}) - ({self.image_paths[0]})"
            self.update_display(1)
            
    def swap_images(self):
        """Swap images between slots"""
        self.images[0], self.images[1] = self.images[1], self.images[0]
        self.image_paths[0], self.image_paths[1] = self.image_paths[1], self.image_paths[0]
        self.update_display(0)
        self.update_display(1)
        
    def clear_all(self):
        """Clear both slots"""
        self.images = [None, None]
        self.image_paths = [None, None]
        self.update_display(0)
        self.update_display(1)
        self.load_info.config(text="")

class WaveletPathExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Wavelet Reconstruction Explorer")
        self.root.geometry("1200x800")

        self.current_matrix = np.zeros((8, 8), dtype=float)
        self.current_level = 1
        self.current_wavelet = 'db1'
        self.update_job = None
        self.dual_viewer = None
        self.current_results = {}
        
        self.create_widgets()
        self.root.after(100, self.realtime_update)

    def create_widgets(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Matrix input frame
        matrix_frame = tk.LabelFrame(left_frame, text="Seed 8x8", font=("Arial", 10, "bold"))
        matrix_frame.pack(pady=10)

        self.buttons = []
        for i in range(8):
            row = []
            for j in range(8):
                btn = tk.Button(
                    matrix_frame,
                    text="0",
                    width=3,
                    height=1,
                    bg="white",
                    relief="raised",
                    command=lambda i=i, j=j: self.toggle_button(i, j)
                )
                btn.grid(row=i, column=j, padx=1, pady=1)
                row.append(btn)
            self.buttons.append(row)

        # Settings frame
        control_frame = tk.LabelFrame(left_frame, text="Settings", font=("Arial", 10, "bold"))
        control_frame.pack(pady=10, fill=tk.X)

        # Level selection
        level_frame = tk.Frame(control_frame)
        level_frame.pack(pady=5, fill=tk.X)
        tk.Label(level_frame, text="Level:", width=8, anchor=tk.W).pack(side=tk.LEFT)

        self.level_var = tk.StringVar(value="1")
        level_combo = ttk.Combobox(level_frame, textvariable=self.level_var,
                                   values=["1", "2", "3"], state="readonly", width=8)
        level_combo.pack(side=tk.LEFT, padx=5)
        level_combo.bind('<<ComboboxSelected>>', self.on_level_change)

        # Wavelet selection
        wavelet_frame = tk.Frame(control_frame)
        wavelet_frame.pack(pady=5, fill=tk.X)
        tk.Label(wavelet_frame, text="Wavelet:", width=8, anchor=tk.W).pack(side=tk.LEFT)

        wavelet_families = {
            'Haar': 'haar',
            'Daubechies': ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10'],
            'Symlets': ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8'],
            'Coiflets': ['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6'],
            'Biorthogonal': ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 
                            'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9',
                            'bior4.4', 'bior5.5', 'bior6.8'],
            'Reverse biorthogonal': ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6',
                                     'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9',
                                     'rbio4.4', 'rbio5.5', 'rbio6.8']
        }

        all_wavelets = []
        for family, wavelets in wavelet_families.items():
            if family == 'Haar':
                all_wavelets.append(('haar', 'haar'))
            else:
                for w in wavelets:
                    all_wavelets.append((w, w))

        self.wavelet_var = tk.StringVar(value="haar")
        wavelet_combo = ttk.Combobox(wavelet_frame, textvariable=self.wavelet_var,
                                     values=[w[0] for w in all_wavelets], width=15)
        wavelet_combo.pack(side=tk.LEFT, padx=5)
        wavelet_combo.bind('<<ComboboxSelected>>', self.on_wavelet_change)

        self.wavelet_display_to_real = {display: real for display, real in all_wavelets}

        # Control buttons
        tk.Button(control_frame, text="Init seed", command=self.reset_matrix,
                 bg="lightcoral").pack(pady=5, fill=tk.X)
        
        tk.Button(control_frame, text="Inverse seed", command=self.invert_matrix,
                 bg="lightblue").pack(pady=5, fill=tk.X)

        tk.Button(control_frame, text="Random seed", command=self.random_matrix,
                 bg="lightyellow").pack(pady=5, fill=tk.X)

        tk.Button(control_frame, text="🖼️ Open Dual Viewer", command=self.open_dual_viewer,
                 bg="lightgreen", font=("Arial", 9, "bold")).pack(pady=5, fill=tk.X)

        # Export buttons
        export_frame = tk.Frame(control_frame)
        export_frame.pack(pady=5, fill=tk.X)
        
        tk.Button(export_frame, text="💾 PNG", command=self.save_as_png,
                 bg="lightgreen", width=10).pack(side=tk.LEFT, padx=2)
        
        tk.Button(export_frame, text="📄 TXT", command=self.export_to_txt,
                 bg="lightyellow", width=10).pack(side=tk.LEFT, padx=2)

        # Wavelet info
        self.wavelet_info = tk.Label(control_frame, text="", justify=tk.LEFT,
                                     font=("Arial", 8), fg="gray")
        self.wavelet_info.pack(pady=5, fill=tk.X)
        self.update_wavelet_info()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(left_frame, textvariable=self.status_var,
                               bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Plot frame
        self.plot_frame = tk.Frame(right_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = None
        self.fig = None

    def open_dual_viewer(self):
        """Open or focus the dual viewer"""
        if self.dual_viewer is None or not self.dual_viewer.top.winfo_exists():
            self.dual_viewer = DualImageViewer(self.root)
            if self.current_results:
                self.dual_viewer.set_paths(self.current_results)
        else:
            self.dual_viewer.top.lift()

    def update_wavelet_info(self):
        wavelet_name = self.current_wavelet
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            info_text = f"📊 {wavelet_name}\n"
            info_text += f"Filter length: {len(wavelet.dec_lo)}\n"
            info_text += f"Vanishing moments: {wavelet.vanishing_moments_phi}"
            self.wavelet_info.config(text=info_text)
        except:
            self.wavelet_info.config(text=f"📊 {wavelet_name}")

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def toggle_button(self, i, j):
        if self.current_matrix[i, j] == 0:
            self.current_matrix[i, j] = 1
            self.buttons[i][j].config(text="1", bg="white", fg="black")
        else:
            self.current_matrix[i, j] = 0
            self.buttons[i][j].config(text="0", bg="white", fg="black")
        self.schedule_update()

    def on_level_change(self, event=None):
        self.current_level = int(self.level_var.get())
        self.schedule_update()

    def on_wavelet_change(self, event=None):
        display_name = self.wavelet_var.get()
        self.current_wavelet = self.wavelet_display_to_real.get(display_name, 'haar')
        self.update_wavelet_info()
        self.schedule_update()

    def schedule_update(self):
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(300, self.realtime_update)

    def reset_matrix(self):
        self.current_matrix = np.zeros((8, 8), dtype=float)
        for i in range(8):
            for j in range(8):
                self.buttons[i][j].config(text="0", bg="white", fg="black")
        self.schedule_update()
    
    def invert_matrix(self):
        for i in range(8):
            for j in range(8):
                if self.current_matrix[i, j] == 0:
                    self.current_matrix[i, j] = 1
                    self.buttons[i][j].config(text="1", bg="white", fg="black")
                else:
                    self.current_matrix[i, j] = 0
                    self.buttons[i][j].config(text="0", bg="white", fg="black")
        self.schedule_update()

    def random_matrix(self):
        self.current_matrix = np.random.choice([0, 1], size=(8, 8))
        for i in range(8):
            for j in range(8):
                self.buttons[i][j].config(text=str(int(self.current_matrix[i, j])),
                                        bg="white", fg="black")
        self.schedule_update()

    def single_step_transform(self, input_matrix, coeff_choice):
        zero_matrix = np.zeros_like(input_matrix)
        coeffs_dict = {
            'cA': zero_matrix.copy(),
            'cH': zero_matrix.copy(),
            'cV': zero_matrix.copy(),
            'cD': zero_matrix.copy()
        }

        coeffs_dict[coeff_choice] = input_matrix.copy()

        try:
            reconstructed = pywt.waverec2(
                (coeffs_dict['cA'], (coeffs_dict['cH'], coeffs_dict['cV'], coeffs_dict['cD'])),
                self.current_wavelet,
                mode='periodization'
            )
            return reconstructed
        except Exception as e:
            print(f"Error transforming with wavelet {self.current_wavelet}: {e}")
            return None

    def generate_all_paths(self, level=1):
        results = {}
        coeffs = ['cA', 'cH', 'cV', 'cD']

        def recursive_transform(current_matrix, path, current_level):
            if current_level == 0:
                results['→'.join(path)] = current_matrix
                return

            for coeff in coeffs:
                next_matrix = self.single_step_transform(current_matrix, coeff)
                if next_matrix is not None:
                    recursive_transform(next_matrix, path + [coeff], current_level - 1)

        recursive_transform(self.current_matrix, [], level)
        return results

    def get_grid_layout(self, level):
        layouts = {
            1: (2, 2),
            2: (4, 4),
            3: (8, 8)
        }
        return layouts.get(level, (2, 2))

    def realtime_update(self):
        """Update display in real time"""
        try:
            if np.sum(self.current_matrix) == 0:
                self.update_status("ℹ️ Matrix is empty - nothing to display")
                if self.canvas:
                    self.canvas.get_tk_widget().destroy()
                    self.canvas = None
                return

            self.update_status(f"🔄 Calculating paths ({self.current_wavelet})...")
            self.root.update_idletasks()
            
            self.current_results = self.generate_all_paths(self.current_level)
            
            # Update dual viewer if it exists
            if self.dual_viewer and self.dual_viewer.top.winfo_exists():
                self.dual_viewer.set_paths(self.current_results)
            
            n_rows, n_cols = self.get_grid_layout(self.current_level)
            
            # Adjust figure size
            if self.current_level == 1:
                figsize = (10, 8)
            elif self.current_level == 2:
                figsize = (12, 10)
            else:
                figsize = (14, 12)

            self.fig = Figure(figsize=figsize)
            self.fig.suptitle(f'Wavelet: {self.current_wavelet}, Level: L={self.current_level} ({len(self.current_results)} paths)',
                              fontsize=12, fontweight='bold')

            paths = list(self.current_results.keys())
            matrices = list(self.current_results.values())

            for idx, (path, matrix) in enumerate(zip(paths, matrices)):
                ax = self.fig.add_subplot(n_rows, n_cols, idx + 1)
                im = ax.imshow(matrix, cmap='gray', aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Show number for all levels
                ax.set_title(f"{idx+1}", fontsize=6 if self.current_level > 1 else 8, 
                           pad=1 if self.current_level > 1 else 2, fontweight='bold')
                
                max_val = np.max(np.abs(matrix))
                if max_val > 0:
                    im.set_clim(-max_val, max_val)

            self.fig.tight_layout()

            # Update canvas
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.update_status(f"✅ Displayed {len(self.current_results)} paths (wavelet: {self.current_wavelet})")

        except Exception as e:
            self.update_status(f"❌ Error: {str(e)[:50]}...")
            print(f"Detailed error: {str(e)}")

    def matrix_to_string(self, matrix):
        rows = []
        for i in range(matrix.shape[0]):
            row_str = '  '.join([f"{x:10.6f}" for x in matrix[i, :]])
            rows.append(f"  [{row_str}]")
        return '\n'.join(rows)

    def export_to_txt(self):
        try:
            if np.sum(self.current_matrix) == 0:
                messagebox.showwarning("Warning", "Seed is empty! Set at least one element.")
                return

            self.update_status("📄 Exporting to TXT...")
            
            results = self.generate_all_paths(self.current_level)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wavelet_paths_{self.current_wavelet}_L{self.current_level}_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("WAVELET RECONSTRUCTION PATHS EXPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write("PARAMETERS:\n")
                f.write(f"  Wavelet: {self.current_wavelet}\n")
                f.write(f"  Level: L={self.current_level}\n")
                f.write(f"  Seed size: {self.current_matrix.shape}\n")
                f.write(f"  Number of paths: {len(results)}\n")
                f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("SEED:\n")
                f.write(self.matrix_to_string(self.current_matrix))
                f.write("\n\n")
                
                f.write("RECONSTRUCTION PATHS:\n")
                f.write("-"*80 + "\n\n")
                
                for idx, (path, matrix) in enumerate(sorted(results.items()), 1):
                    f.write(f"Path {idx}: {path}\n")
                    f.write(f"Size: {matrix.shape}\n")
                    f.write("Matrix:\n")
                    f.write(self.matrix_to_string(matrix))
                    f.write("\n\n")
                    f.write("-"*40 + "\n\n")
                
                f.write("STATISTICS:\n")
                f.write("-"*40 + "\n")
                
                all_values = []
                for matrix in results.values():
                    all_values.extend(matrix.flatten())
                
                all_values = np.array(all_values)
                f.write(f"  Min: {np.min(all_values):.6f}\n")
                f.write(f"  Max: {np.max(all_values):.6f}\n")
                f.write(f"  Mean: {np.mean(all_values):.6f}\n")
                f.write(f"  STD: {np.std(all_values):.6f}\n")
                
                f.write("\nPATH ENERGIES (sum of squares):\n")
                for path, matrix in sorted(results.items()):
                    energy = np.sum(matrix**2)
                    f.write(f"  {path}: {energy:.6f}\n")
            
            self.update_status(f"✅ Export complete: {filename}")
            messagebox.showinfo("Complete", f"File saved successfully:\n{filename}\n\nTotal paths: {len(results)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export error:\n{str(e)}")
            self.update_status("❌ Export error")

    def save_as_png(self):
        try:
            if np.sum(self.current_matrix) == 0:
                messagebox.showwarning("Warning", "Matrix is empty.\nAdd at least one element.")
                return

            self.update_status("💾 Saving...")
            
            results = self.generate_all_paths(self.current_level)
            
            n_rows, n_cols = self.get_grid_layout(self.current_level)
            
            if self.current_level == 1:
                figsize = (10, 8)
            elif self.current_level == 2:
                figsize = (12, 10)
            else:
                figsize = (14, 12)

            fig = Figure(figsize=figsize)
            fig.suptitle(f'Wavelet: {self.current_wavelet}, Level: L={self.current_level} ({len(results)} paths)',
                         fontsize=12, fontweight='bold')

            paths = list(results.keys())
            matrices = list(results.values())

            for idx, (path, matrix) in enumerate(zip(paths, matrices)):
                ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                im = ax.imshow(matrix, cmap='gray', aspect='equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Show number for all levels
                ax.set_title(f"{idx+1}", fontsize=6 if self.current_level > 1 else 8, 
                           pad=1 if self.current_level > 1 else 2)

                max_val = np.max(np.abs(matrix))
                if max_val > 0:
                    im.set_clim(-max_val, max_val)

            fig.tight_layout()

            filename = f"wavelet_paths_{self.current_wavelet}_L{self.current_level}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            messagebox.showinfo("Complete", f"File saved: {filename}")
            self.update_status(f"✅ File saved: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving file:\n{str(e)}")
            self.update_status("❌ Save error")

if __name__ == "__main__":
    root = tk.Tk()
    app = WaveletPathExplorer(root)
    root.mainloop()