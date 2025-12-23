import cv2
import numpy as np
from skimage.color import rgb2gray
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ---------------- Image Processing Functions ----------------

kernel_size_default = 3
radius_default = 30
Q_default = 1.5
butterworth_order_default = 2

def sobelx_func(frame, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    gray = rgb2gray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelx_abs = np.absolute(sobelx)
    if np.max(sobelx_abs) > 0:
        return np.uint8(np.clip(sobelx_abs / np.max(sobelx_abs) * 255, 0, 255))
    return np.zeros_like(gray, dtype=np.uint8)

def sobely_func(frame, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    gray = rgb2gray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobely_abs = np.absolute(sobely)
    if np.max(sobely_abs) > 0:
        return np.uint8(np.clip(sobely_abs / np.max(sobely_abs) * 255, 0, 255))
    return np.zeros_like(gray, dtype=np.uint8)

def gradient_magnitude_func(sobelx, sobely):
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude.astype(np.uint8)

def Canny_edge_detection(frame, ksize, low_thresh, high_thresh):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    
    low_thresh = max(0, min(255, float(low_thresh)))
    high_thresh = max(0, min(255, float(high_thresh)))
    
    # Ensure high > low
    if high_thresh <= low_thresh:
        high_thresh = low_thresh + 10
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_mag = np.hypot(gx, gy)
    
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max() * 255
    
    grad_angle = np.arctan2(gy, gx)
    M, N = grad_mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = grad_angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = r = 255
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q, r = grad_mag[i, j+1], grad_mag[i, j-1]
            elif 22.5 <= angle[i,j] < 67.5:
                q, r = grad_mag[i+1, j-1], grad_mag[i-1, j+1]
            elif 67.5 <= angle[i,j] < 112.5:
                q, r = grad_mag[i+1, j], grad_mag[i-1, j]
            elif 112.5 <= angle[i,j] < 157.5:
                q, r = grad_mag[i-1, j-1], grad_mag[i+1, j+1]
            if grad_mag[i,j] >= q and grad_mag[i,j] >= r:
                Z[i,j] = grad_mag[i,j]

    strong, weak = 255, 75
    res = np.zeros_like(Z, dtype=np.uint8)
    strong_i, strong_j = np.where(Z >= high_thresh)
    weak_i, weak_j = np.where((Z < high_thresh) & (Z >= low_thresh))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    result = np.copy(res)

    for i in range(1, M-1):
        for j in range(1, N-1):
            if result[i,j] == weak:
                if any(result[i+di, j+dj] == strong
                       for di in [-1, 0, 1]
                       for dj in [-1, 0, 1]
                       if not (di == 0 and dj == 0)):
                    result[i,j] = strong
                else:
                    result[i,j] = 0
    return result

# ---------------- Frequency Filters ----------------

def DFT_and_reconstruct(gray_img, filter_mask=None):
    img = np.float32(gray_img)
    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    c = 255.0 / (np.log(1 + np.max(magnitude)) + 1e-9)
    magnitude_spectrum = c * np.log(magnitude + 1)

    if filter_mask is not None:
        fshift = dft_shift * filter_mask
    else:
        fshift = dft_shift

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    if img_back.max() != 0:
        img_back = img_back / img_back.max() * 255
    img_back = img_back.astype(np.uint8)
    mag_display = np.uint8(np.clip(magnitude_spectrum, 0, 255))
    return img_back, mag_display

def ILPF(frame, r):
    r = max(1, int(r))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    mask = np.zeros((rows, cols, 2), np.float32)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r*r
    mask[mask_area] = 1
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

def GLPF(frame, r):
    r = max(1, float(r))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    D0 = float(r)
    mask = np.zeros((rows, cols, 2), np.float32)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    H = np.exp(-(D**2)/(2*D0*D0))
    mask[:,:,0] = H
    mask[:,:,1] = H
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

def BLPF(frame, r, n=2):
    r = max(1, float(r))
    n = max(1, int(n))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    D0 = float(r)
    mask = np.zeros((rows, cols, 2), np.float32)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol)**2 + (y - crow)**2)
    H = 1.0 / (1.0 + (D / (D0 + 1e-9))**(2*n))
    mask[:,:,0] = H
    mask[:,:,1] = H
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

def IHPF(frame, r):
    r = max(1, int(r))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    mask = np.ones((rows, cols, 2), np.float32)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
    mask[mask_area] = 0
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

def GHPF(frame, r):
    r = max(1, float(r))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    D0 = float(r)
    mask = np.zeros((rows, cols, 2), np.float32)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol)**2 + (y - crow)**2)
    H = 1.0 - np.exp(-(D**2)/(2*D0*D0))
    mask[:,:,0] = H
    mask[:,:,1] = H
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

def BHPF(frame, r, n=2):
    r = max(1, float(r))
    n = max(1, int(n))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    D0 = float(r)
    mask = np.zeros((rows, cols, 2), np.float32)
    crow, ccol = rows//2, cols//2
    y, x = np.ogrid[:rows, :cols]
    D = np.sqrt((x - ccol)**2 + (y - crow)**2)
    H = 1.0 - 1.0 / (1.0 + (D/(D0 + 1e-9))**(2*n))
    mask[:,:,0] = H
    mask[:,:,1] = H
    img_back, _ = DFT_and_reconstruct(gray, mask)
    return img_back

# ---------------- Mean Filters ----------------

def arithmetic_mean_filter(frame, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)
    return cv2.filter2D(gray, -1, kernel)

def geometric_mean_filter(frame, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)+1
    log_img = np.log(gray)
    kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)
    log_mean = cv2.filter2D(log_img, -1, kernel)
    geo_mean = np.exp(log_mean)
    geo_mean = np.clip(geo_mean, 0, 255)
    return geo_mean.astype(np.uint8)

def harmonic_mean_filter(frame, ksize):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)+1
    inv = 1.0/gray
    kernel = np.ones((ksize, ksize), np.float32)
    inv_sum = cv2.filter2D(inv, -1, kernel)
    h_mean = (ksize*ksize)/(inv_sum + 1e-9)
    h_mean = np.clip(h_mean, 0, 255)
    return h_mean.astype(np.uint8)

def contraharmonic_mean_filter(frame, ksize, Q=1.5):
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)
    Q = float(Q)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1e-9
    numerator = cv2.filter2D(np.power(gray, Q+1), -1, np.ones((ksize, ksize)))
    denominator = cv2.filter2D(np.power(gray, Q), -1, np.ones((ksize, ksize))) + 1e-9
    ch_mean = numerator/denominator
    ch_mean = np.clip(ch_mean, 0, 255)
    return ch_mean.astype(np.uint8)

# ---------------- Professional GUI ----------------

class VideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Professional Camera Filters - Full Control System")
        self.window.geometry("1400x800")
        self.window.configure(bg='#1e1e1e')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Segoe UI', 10))
        
        self.cap = cv2.VideoCapture(0)
        self.mode = None
        self.kernel_size = kernel_size_default
        self.radius = radius_default
        self.Q = Q_default
        self.butterworth_order = butterworth_order_default
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.button_refs = {}
        self.sliders = {}
        
        # Main container
        main_container = ttk.Frame(window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video frame with border
        video_frame = tk.Frame(left_panel, bg='#0a0a0a', bd=3, relief=tk.GROOVE)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='#0a0a0a')
        self.video_label.pack(padx=5, pady=5)
        
        # Status bar with FPS
        status_frame = tk.Frame(left_panel, bg='#2a2a2a')
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = tk.Label(status_frame, text="● Status: Ready", 
                                     bg='#2a2a2a', fg='#00ff88', 
                                     font=('Consolas', 10, 'bold'), anchor=tk.W, padx=15, pady=8)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_label = tk.Label(status_frame, text="FPS: --", 
                                 bg='#2a2a2a', fg='#00d4ff', 
                                 font=('Consolas', 10, 'bold'), anchor=tk.E, padx=15)
        self.fps_label.pack(side=tk.RIGHT)
        
        # Right panel - Controls with scrollbar
        right_container = ttk.Frame(main_container, width=450)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, padx=0)
        right_container.pack_propagate(False)
        
        # Canvas and scrollbar for scrolling
        canvas = tk.Canvas(right_container, bg='#1e1e1e', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas, orient="vertical", command=canvas.yview, 
                                bg='#2a2a2a', troughcolor='#1e1e1e', 
                                activebackground='#00d4ff', width=14)
        right_panel = ttk.Frame(canvas)
        
        right_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=right_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        title_frame = tk.Frame(right_panel, bg='#0078d4', bd=0)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(title_frame, text="⚙ FILTER CONTROL CENTER", 
                              bg='#0078d4', fg='white', 
                              font=('Segoe UI', 16, 'bold'), pady=12)
        title_label.pack()
        
        # Edge Detection Filters
        self.create_filter_group(right_panel, "🔍 Edge Detection", [
            ("Sobel X", "sobelx", "x"),
            ("Sobel Y", "sobely", "y"),
            ("Gradient Magnitude", "gradient", "s"),
            ("Canny Edge", "canny", "c")
        ])
        
        # Low Pass Filters
        self.create_filter_group(right_panel, "🔽 Low Pass Filters", [
            ("Ideal LPF", "ilpf", "1"),
            ("Gaussian LPF", "glpf", "2"),
            ("Butterworth LPF", "blpf", "3")
        ])
        
        # High Pass Filters
        self.create_filter_group(right_panel, "🔼 High Pass Filters", [
            ("Ideal HPF", "ihpf", "4"),
            ("Gaussian HPF", "ghpf", "5"),
            ("Butterworth HPF", "bhpf", "6")
        ])
        
        # Mean Filters
        self.create_filter_group(right_panel, "📊 Mean Filters", [
            ("Arithmetic Mean", "arith_mean", "a"),
            ("Geometric Mean", "geo_mean", "G"),
            ("Harmonic Mean", "harm_mean", "h"),
            ("Contraharmonic Mean", "contra_mean", "m")
        ])
        
        # ==================== PARAMETERS SECTION ====================
        
        params_title_frame = tk.Frame(right_panel, bg='#ff6b35', bd=0)
        params_title_frame.pack(fill=tk.X, pady=(15, 0))
        
        params_title = tk.Label(params_title_frame, text="🎛️ PARAMETER CONTROLS", 
                               bg='#ff6b35', fg='white', 
                               font=('Segoe UI', 14, 'bold'), pady=10)
        params_title.pack()
        
        # Kernel Size Control
        self.create_slider_control(
            right_panel, 
            "Kernel Size", 
            1, 31, 2,
            self.kernel_size,
            'kernel_size',
            "📐 Sobel X/Y, Gradient, Canny, Mean Filters"
        )
        
        # Radius Control
        self.create_slider_control(
            right_panel, 
            "Radius", 
            1, 200, 5,
            self.radius,
            'radius',
            "⭕ All Frequency Filters (LPF/HPF)"
        )
        
        # Q Parameter Control
        self.create_slider_control(
            right_panel, 
            "Q Parameter", 
            -5.0, 5.0, 0.1,
            self.Q,
            'Q',
            "🎚️ Contraharmonic Mean Filter",
            is_float=True
        )
        
        # Butterworth Order Control
        self.create_slider_control(
            right_panel, 
            "Butterworth Order", 
            1, 10, 1,
            self.butterworth_order,
            'butterworth_order',
            "📈 Butterworth LPF/HPF Sharpness"
        )
        
        # Canny Low Threshold Control
        self.create_slider_control(
            right_panel, 
            "Canny Low Threshold", 
            0, 255, 1,
            self.canny_low_threshold,
            'canny_low_threshold',
            "🔻 Canny Edge - Weak Edge Detection"
        )
        
        # Canny High Threshold Control
        self.create_slider_control(
            right_panel, 
            "Canny High Threshold", 
            0, 255, 1,
            self.canny_high_threshold,
            'canny_high_threshold',
            "🔺 Canny Edge - Strong Edge Detection"
        )
        
        # Control buttons
        button_frame = ttk.Frame(right_panel)
        button_frame.pack(fill=tk.X, pady=15, padx=10)
        
        reset_btn = tk.Button(button_frame, text="🔄 RESET ALL", command=self.reset_all,
                             bg='#ff4444', fg='white', font=('Segoe UI', 11, 'bold'),
                             padx=20, pady=12, relief=tk.FLAT, cursor='hand2')
        reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="⭕ NO FILTER", command=self.reset_filter,
                             bg='#6c757d', fg='white', font=('Segoe UI', 11, 'bold'),
                             padx=20, pady=12, relief=tk.FLAT, cursor='hand2')
        clear_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Keyboard shortcuts info
        shortcuts_frame = tk.LabelFrame(right_panel, text="⌨️ Keyboard Shortcuts", 
                                       bg='#2a2a2a', fg='#00d4ff', 
                                       font=('Segoe UI', 10, 'bold'), bd=2, relief=tk.GROOVE)
        shortcuts_frame.pack(fill=tk.X, pady=10, padx=10)
        
        shortcuts_text = (
            "x: Sobel X  |  y: Sobel Y  |  s: Gradient  |  c: Canny\n"
            "1-6: Frequency Filters  |  a, G, h, m: Mean Filters\n"
            "n: No Filter  |  q: Quit Application"
        )
        shortcuts_label = tk.Label(shortcuts_frame, text=shortcuts_text, 
                                  bg='#2a2a2a', fg='#cccccc', 
                                  font=('Consolas', 9), justify=tk.LEFT)
        shortcuts_label.pack(padx=10, pady=8)
        
        # Keyboard bindings
        window.bind("<Key>", self.key_handler)
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = cv2.getTickCount()
        
        self.update_frame()
    
    def create_filter_group(self, parent, title, filters):
        group_frame = tk.LabelFrame(parent, text=title, bg='#2a2a2a', fg='#00d4ff', 
                                   font=('Segoe UI', 11, 'bold'), bd=2, relief=tk.GROOVE)
        group_frame.pack(fill=tk.X, pady=8, padx=10)
        
        for label, mode, key in filters:
            btn = tk.Button(group_frame, text=f"{label} [{key}]", 
                          command=lambda m=mode: self.set_mode(m),
                          bg='#3a3a3a', fg='white', font=('Segoe UI', 10),
                          activebackground='#0078d4', activeforeground='white',
                          relief=tk.FLAT, padx=15, pady=8, cursor='hand2',
                          borderwidth=0)
            btn.pack(fill=tk.X, padx=8, pady=3)
            self.button_refs[mode] = btn
    
    def create_slider_control(self, parent, name, min_val, max_val, step, 
                             default_val, attr_name, description, is_float=False):
        """Create a comprehensive slider control with all features"""
        frame = tk.LabelFrame(parent, text=f"  {name}  ", bg='#2a2a2a', fg='white', 
                             font=('Segoe UI', 10, 'bold'), bd=2, relief=tk.GROOVE)
        frame.pack(fill=tk.X, pady=8, padx=10)
        
        # Description
        desc_label = tk.Label(frame, text=description, 
                            bg='#2a2a2a', fg='#aaaaaa', 
                            font=('Segoe UI', 8), wraplength=400, justify=tk.LEFT)
        desc_label.pack(padx=10, pady=(5, 2))
        
        # Value display frame
        value_frame = tk.Frame(frame, bg='#1a1a1a', bd=2, relief=tk.SUNKEN)
        value_frame.pack(fill=tk.X, padx=10, pady=8)
        
        if is_float:
            value_text = f"{default_val:.1f}"
        else:
            value_text = str(int(default_val))
        
        value_label = tk.Label(value_frame, text=value_text, 
                              bg='#1a1a1a', fg='#00ff88', 
                              font=('Consolas', 16, 'bold'))
        value_label.pack(pady=8)
        
        # Range display
        range_label = tk.Label(frame, text=f"Range: {min_val} → {max_val}", 
                              bg='#2a2a2a', fg='#888888', 
                              font=('Segoe UI', 8))
        range_label.pack(pady=(0, 5))
        
        # Slider with custom styling
        slider_frame = tk.Frame(frame, bg='#2a2a2a')
        slider_frame.pack(fill=tk.X, padx=15, pady=8)
        
        slider = tk.Scale(slider_frame, from_=min_val, to=max_val, 
                         orient=tk.HORIZONTAL, bg='#2a2a2a', fg='#00d4ff',
                         troughcolor='#1a1a1a', highlightthickness=0,
                         showvalue=0, resolution=step if not is_float else 0.01,
                         length=350, width=20, sliderlength=30,
                         activebackground='#0078d4',
                         command=lambda v: self.update_slider_value(attr_name, value_label, float(v), is_float))
        slider.set(default_val)
        slider.pack(fill=tk.X)
        
        self.sliders[attr_name] = slider
        
        # Button controls
        btn_frame = tk.Frame(frame, bg='#2a2a2a')
        btn_frame.pack(fill=tk.X, padx=10, pady=8)
        
        def decrease():
            current = getattr(self, attr_name)
            new_val = max(min_val, current - step)
            setattr(self, attr_name, new_val)
            slider.set(new_val)
            self.update_slider_value(attr_name, value_label, new_val, is_float)
        
        def increase():
            current = getattr(self, attr_name)
            new_val = min(max_val, current + step)
            setattr(self, attr_name, new_val)
            slider.set(new_val)
            self.update_slider_value(attr_name, value_label, new_val, is_float)
        
        def reset():
            setattr(self, attr_name, default_val)
            slider.set(default_val)
            self.update_slider_value(attr_name, value_label, default_val, is_float)
        
        tk.Button(btn_frame, text="− −", command=decrease, 
                 bg='#dc3545', fg='white', width=8, font=('Segoe UI', 10, 'bold'),
                 relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="RESET", command=reset, 
                 bg='#ffc107', fg='black', width=10, font=('Segoe UI', 10, 'bold'),
                 relief=tk.FLAT, cursor='hand2').pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        tk.Button(btn_frame, text="+ +", command=increase, 
                 bg='#28a745', fg='white', width=8, font=('Segoe UI', 10, 'bold'),
                 relief=tk.FLAT, cursor='hand2').pack(side=tk.RIGHT, padx=2)
    
    def update_slider_value(self, attr_name, label, value, is_float):
        # Ensure kernel size is always odd
        if attr_name == 'kernel_size':
            value = int(value)
            if value % 2 == 0:
                value += 1
            value = max(1, min(31, value))
        
        # Ensure valid ranges
        if attr_name == 'radius':
            value = max(1, min(200, int(value)))
        elif attr_name == 'butterworth_order':
            value = max(1, min(10, int(value)))
        elif attr_name == 'canny_low_threshold':
            value = max(0, min(255, int(value)))
        elif attr_name == 'canny_high_threshold':
            value = max(0, min(255, int(value)))
        elif attr_name == 'Q':
            value = max(-5.0, min(5.0, float(value)))
        
        setattr(self, attr_name, value)
        if is_float:
            label.config(text=f"{value:.1f}")
        else:
            label.config(text=str(int(value)))
    
    def set_mode(self, mode):
        self.mode = mode
        self.update_button_states()
        mode_names = {
            'sobelx': 'Sobel X Edge Detection', 
            'sobely': 'Sobel Y Edge Detection', 
            'gradient': 'Gradient Magnitude',
            'canny': 'Canny Edge Detection', 
            'ilpf': 'Ideal Low Pass Filter', 
            'glpf': 'Gaussian Low Pass Filter',
            'blpf': 'Butterworth Low Pass Filter', 
            'ihpf': 'Ideal High Pass Filter', 
            'ghpf': 'Gaussian High Pass Filter',
            'bhpf': 'Butterworth High Pass Filter', 
            'arith_mean': 'Arithmetic Mean Filter',
            'geo_mean': 'Geometric Mean Filter', 
            'harm_mean': 'Harmonic Mean Filter',
            'contra_mean': 'Contraharmonic Mean Filter'
        }
        self.status_label.config(text=f"● Active: {mode_names.get(mode, 'Unknown')}")
    
    def reset_filter(self):
        self.mode = None
        self.update_button_states()
        self.status_label.config(text="● Status: No Filter Active")
    
    def reset_all(self):
        """Reset all parameters to default values"""
        self.mode = None
        self.kernel_size = kernel_size_default
        self.radius = radius_default
        self.Q = Q_default
        self.butterworth_order = butterworth_order_default
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        
        # Update all sliders
        for attr_name, slider in self.sliders.items():
            value = getattr(self, attr_name)
            slider.set(value)
        
        self.update_button_states()
        self.status_label.config(text="● Status: All Parameters Reset")
    
    def update_button_states(self):
        for mode, btn in self.button_refs.items():
            if mode == self.mode:
                btn.config(bg='#0078d4', fg='white', relief=tk.SUNKEN)
            else:
                btn.config(bg='#3a3a3a', fg='white', relief=tk.FLAT)
    
    def key_handler(self, event):
        key = event.char
        if key in 'xyscn123456aGhm':
            mode_map = {
                'x': 'sobelx', 'y': 'sobely', 's': 'gradient', 'c': 'canny', 'n': None,
                '1': 'ilpf', '2': 'glpf', '3': 'blpf', 
                '4': 'ihpf', '5': 'ghpf', '6': 'bhpf',
                'a': 'arith_mean', 'G': 'geo_mean', 'h': 'harm_mean', 'm': 'contra_mean'
            }
            self.mode = mode_map[key]
            if self.mode is None:
                self.reset_filter()
            else:
                self.set_mode(self.mode)
        elif key == 'q':
            self.cap.release()
            self.window.destroy()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            display = frame.copy()
            
            try:
                # Ensure kernel_size is odd and in valid range
                kernel_size = int(self.kernel_size)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = max(1, min(31, kernel_size))
                
                # Ensure radius is valid
                radius = max(1, int(self.radius))
                
                # Ensure butterworth order is valid
                butterworth_order = max(1, int(self.butterworth_order))
                
                # Ensure thresholds are valid
                low_thresh = max(0, min(255, int(self.canny_low_threshold)))
                high_thresh = max(0, min(255, int(self.canny_high_threshold)))
                if high_thresh <= low_thresh:
                    high_thresh = low_thresh + 10
                
                # Ensure Q is valid
                Q = float(self.Q)
                
                if self.mode == 'sobelx':
                    out = sobelx_func(frame, kernel_size)
                    display = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                elif self.mode == 'sobely':
                    out = sobely_func(frame, kernel_size)
                    display = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                elif self.mode == 'gradient':
                    sx = sobelx_func(frame, kernel_size)
                    sy = sobely_func(frame, kernel_size)
                    out = gradient_magnitude_func(sx, sy)
                    display = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                elif self.mode == 'canny':
                    out = Canny_edge_detection(frame, kernel_size, low_thresh, high_thresh)
                    display = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                elif self.mode == 'ilpf':
                    display = cv2.cvtColor(ILPF(frame, radius), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'glpf':
                    display = cv2.cvtColor(GLPF(frame, radius), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'blpf':
                    display = cv2.cvtColor(BLPF(frame, radius, butterworth_order), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'ihpf':
                    display = cv2.cvtColor(IHPF(frame, radius), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'ghpf':
                    display = cv2.cvtColor(GHPF(frame, radius), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'bhpf':
                    display = cv2.cvtColor(BHPF(frame, radius, butterworth_order), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'arith_mean':
                    display = cv2.cvtColor(arithmetic_mean_filter(frame, kernel_size), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'geo_mean':
                    display = cv2.cvtColor(geometric_mean_filter(frame, kernel_size), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'harm_mean':
                    display = cv2.cvtColor(harmonic_mean_filter(frame, kernel_size), cv2.COLOR_GRAY2BGR)
                elif self.mode == 'contra_mean':
                    display = cv2.cvtColor(contraharmonic_mean_filter(frame, kernel_size, Q), cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"Processing error ({self.mode}): {e}")
                import traceback
                traceback.print_exc()
                display = frame.copy()
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = cv2.getTickCount()
                time_diff = (current_time - self.start_time) / cv2.getTickFrequency()
                fps = 30 / time_diff
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.start_time = current_time
            
            # Convert and resize for display
            img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (800, 600))
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)
        
        self.window.after(10, self.update_frame)
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

root = tk.Tk()
app = VideoApp(root)
root.mainloop()