"""
OsteoScan AI — Medical Imaging Intelligence
Production-ready Streamlit application.

Fixes applied:
  - DICOM (H, W, 1) shape crash
  - weights_only torch.load fallback (UnpicklingError)
  - XSS via uploaded_file.name
  - Single forward pass (classification + Grad-CAM)
  - DICOM detection via magic bytes (extension-independent)
  - Optional dependency guards (pydicom / cv2 / pytorch-grad-cam)
  - Safe error display (no traceback leak)
  - MONOCHROME1 case-insensitive
"""

import os
import io
import html
import warnings

import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION (must be first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="OsteoScan AI | Medical Imaging Intelligence",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DEPENDENCY IMPORTS (with guards)
# ============================================================

# --- Core (required) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image, ImageOps
    from torchvision import models, transforms
except ImportError as err:
    st.error(
        f"Missing core dependency: `{err.name}`. "
        "Run: `pip install -r requirements.txt`"
    )
    st.stop()


# --- Optional: DICOM ---
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import (
        apply_modality_lut,
        apply_voi_lut,
    )
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    pydicom = None


# --- Optional: OpenCV ---
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


# --- Optional: Grad-CAM ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# Grad-CAM is core to the app — hard stop if missing
if not HAS_CV2 or not HAS_GRADCAM:
    st.error(
        "Grad-CAM dependencies missing. Install:\n\n"
        "`pip install opencv-python pytorch-grad-cam`"
    )
    st.stop()

if not HAS_PYDICOM:
    st.warning(
        "⚠️ `pydicom` not installed — DICOM uploads disabled. "
        "Install with: `pip install pydicom`"
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

MODEL_PATH = "best_lumos_densenet.pth"

CLASS_NAMES = ["Normal", "Osteopenia", "Osteoporosis"]

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_DETAILS = {
    "Normal": {
        "color": "#16A085",
        "description": "The model classified this image under the Normal category.",
    },
    "Osteopenia": {
        "color": "#D69E36",
        "description": "The model classified this image under the Osteopenia category.",
    },
    "Osteoporosis": {
        "color": "#D65A62",
        "description": "The model classified this image under the Osteoporosis category.",
    },
}


# ============================================================
# 4. LOAD CUSTOM CSS
# ============================================================

def local_css(file_name: str) -> None:
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(
            "`app.css` not found. Keep app.css in the same folder as app.py."
        )


local_css("app.css")


# ============================================================
# 5. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMethod.BILINEAR
        if hasattr(transforms, "InterpolationMethod")
        else transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# 6. SAFE TORCH CHECKPOINT LOADER
# ============================================================

def safe_torch_load(path: str, device):
    """
    Load a checkpoint with weights_only=True, falling back safely.

    Cases handled:
      1. Modern PyTorch + tensor-only checkpoint → works directly.
      2. Old PyTorch (no weights_only arg) → TypeError → plain load.
      3. Modern PyTorch + checkpoint has non-tensor content
         (e.g. numpy arrays, custom classes) → UnpicklingError
         → fall back to weights_only=False (trusted file only).
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)

    except TypeError:
        # PyTorch < 1.13 — weights_only arg doesn't exist yet
        return torch.load(path, map_location=device)

    except Exception as err:
        warnings.warn(
            f"weights_only load failed ({type(err).__name__}: {err}). "
            "Falling back to full pickle load — "
            "ONLY use with trusted checkpoints.",
            RuntimeWarning,
        )
        return torch.load(path, map_location=device, weights_only=False)


# ============================================================
# 7. LOAD TRAINED DENSENET121
# ============================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = models.densenet121(weights=None)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(CLASS_NAMES))

    checkpoint = safe_torch_load(MODEL_PATH, DEVICE)

    # Unwrap common checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    # Strip DataParallel "module." prefix
    if checkpoint and all(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {
            k.replace("module.", "", 1): v
            for k, v in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# 8. INTENSITY NORMALIZATION
# ============================================================

def normalize_to_uint8(image_array) -> np.ndarray:
    """Robust percentile-based normalization to uint8."""
    arr = np.asarray(image_array)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    low = float(np.percentile(arr, 1))
    high = float(np.percentile(arr, 99))

    if high <= low:
        raise ValueError("Invalid image intensity range.")

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low) * 255.0

    return arr.astype(np.uint8)


# ============================================================
# 9. DICOM DETECTION
# ============================================================

def looks_like_dicom(file_bytes: bytes, filename: str) -> bool:
    """
    Detect DICOM by:
      - '.dcm' extension (fast path)
      - 'DICM' magic bytes at offset 128 (standard preamble)
    Fallback to extension-only for preambless DICOMs.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".dcm":
        return True
    if len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM":
        return True
    return False


# ============================================================
# 10. DICOM IMAGE PROCESSING
# ============================================================

def process_dicom(file_bytes: bytes) -> np.ndarray:
    if not HAS_PYDICOM:
        raise ValueError(
            "DICOM file detected but `pydicom` is not installed."
        )

    ds = pydicom.dcmread(io.BytesIO(file_bytes))

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM file does not contain pixel data.")

    pixel_array = ds.pixel_array

    # ---- Frame / channel normalization ----
    if pixel_array.ndim == 4:
        # Multi-frame, multi-channel — take first frame
        pixel_array = pixel_array[0]

    elif pixel_array.ndim == 3:
        last_dim = pixel_array.shape[-1]
        if last_dim in (3, 4):
            pass  # RGB / RGBA — keep
        elif last_dim == 1:
            # (H, W, 1) → (H, W)
            pixel_array = pixel_array[:, :, 0]
        else:
            # Multi-frame (N, H, W) — first frame
            pixel_array = pixel_array[0]

    # ---- Modality LUT ----
    try:
        pixel_array = apply_modality_lut(pixel_array, ds)
    except Exception:
        pass

    # ---- VOI LUT / Windowing ----
    try:
        if hasattr(ds, "WindowCenter") or hasattr(ds, "VOILUTSequence"):
            pixel_array = apply_voi_lut(pixel_array, ds)
    except Exception:
        pass

    # ---- RGB / RGBA DICOM ----
    if pixel_array.ndim == 3 and pixel_array.shape[-1] in (3, 4):
        if pixel_array.shape[-1] == 4:
            pixel_array = pixel_array[:, :, :3]

        rgb = np.asarray(pixel_array, dtype=np.float32)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

        low = np.percentile(rgb, 1)
        high = np.percentile(rgb, 99)
        if high <= low:
            raise ValueError("Invalid RGB DICOM intensity range.")

        rgb = np.clip(rgb, low, high)
        rgb = ((rgb - low) / (high - low) * 255).astype(np.uint8)

        gray = Image.fromarray(rgb).convert("L")
        return normalize_to_uint8(np.array(gray))

    # ---- Grayscale DICOM ----
    img_array = normalize_to_uint8(pixel_array)

    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if photometric == "MONOCHROME1":
        img_array = 255 - img_array

    return img_array


# ============================================================
# 11. UNIFIED IMAGE PROCESSOR
# ============================================================

def process_uploaded_file(uploaded_file):
    """
    Returns (grayscale_uint8_array, RGB_PIL_display).
    """
    file_bytes = uploaded_file.getvalue()

    if looks_like_dicom(file_bytes, uploaded_file.name):
        img_array = process_dicom(file_bytes)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        img_array = normalize_to_uint8(np.array(image))

    if img_array is None or img_array.size == 0:
        raise ValueError("Unable to process the uploaded image.")

    img_display = Image.fromarray(img_array).convert("RGB")
    return img_array, img_display


# ============================================================
# 12. PREDICTION (single forward pass)
# ============================================================

def predict(model, image_array):
    """
    Single forward pass: GradCAM internally does forward+backward,
    and exposes raw model outputs via cam.outputs.
    Softmax on those gives us classification AND heatmap —
    no need for a separate torch.no_grad() forward.

    Falls back to classification-only if Grad-CAM fails.
    """
    img_tensor = transform(image_array).unsqueeze(0).to(DEVICE)
    target_layers = [model.features[-1]]

    cam_image = None
    cam_error = None

    try:
        with GradCAM(model=model, target_layers=target_layers) as cam:
            # targets=None → cam automatically uses argmax (predicted class)
            grayscale_cam = cam(input_tensor=img_tensor, targets=None)
            outputs = cam.outputs  # raw logits from the same forward pass

        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class_idx = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx].item())
        all_probabilities = probabilities.detach().cpu().numpy()

        # Build overlay
        grayscale_cam = grayscale_cam[0, :]
        rgb = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    except Exception as err:
        cam_error = f"{type(err).__name__}: {err}"

        # Fallback: classification only
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_class_idx = int(torch.argmax(probabilities).item())
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx].item())
        all_probabilities = probabilities.detach().cpu().numpy()

    return (
        predicted_class,
        confidence,
        all_probabilities,
        cam_image,
        cam_error,
    )



# ============================================================
# 13. CLINICAL PREMIUM UI + FUNCTIONAL NAVIGATION
# ============================================================

def section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<p class="section-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-header"><h3>{html.escape(title)}</h3>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_probability_card(class_name: str, probability: float) -> None:
    percentage = float(probability) * 100
    color = CLASS_DETAILS[class_name]["color"]
    st.markdown(
        f"""
        <div class="probability-item">
            <div class="probability-top">
                <span class="probability-name">{html.escape(class_name)}</span>
                <span class="probability-value">{percentage:.2f}%</span>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width:{percentage:.2f}%;background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to_analysis() -> None:
    st.session_state["active_page"] = "AI Analysis"


# ============================================================
# 14. SIDEBAR / NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">✚</div>
            <div>
                <div class="brand-name">OsteoScan AI</div>
                <div class="brand-subtitle">MEDICAL IMAGING INTELLIGENCE</div>
            </div>
        </div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-label">WORKSPACE NAVIGATION</div>
        """,
        unsafe_allow_html=True,
    )

    active_page = st.radio(
        "Workspace navigation",
        ["AI Analysis", "Imaging Workspace", "Model Information"],
        key="active_page",
        label_visibility="collapsed",
        format_func=lambda value: {
            "AI Analysis": "◈  AI Analysis",
            "Imaging Workspace": "▧  Imaging Workspace",
            "Model Information": "ⓘ  Model Information",
        }[value],
    )

    st.markdown(
        """
        <div class="sidebar-divider"></div>
        <div class="sidebar-label">MODEL CONFIGURATION</div>
        <div class="sidebar-info"><span>Architecture</span><strong>DenseNet121</strong></div>
        <div class="sidebar-info"><span>Input size</span><strong>224 × 224</strong></div>
        <div class="sidebar-info"><span>Classes</span><strong>03</strong></div>
        <div class="sidebar-info"><span>Explainability</span><strong>Grad-CAM</strong></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-info"><span>Compute device</span><strong>{str(DEVICE).upper()}</strong></div>
        <div class="sidebar-footer">
            <div class="sidebar-status"><span class="status-dot"></span>Research prototype</div>
            <p>AI-assisted bone imaging research. Not intended for clinical diagnosis.</p>
            <div class="sidebar-version">OsteoScan AI · v2.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 15. SHARED HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div>
            <div class="eyebrow">CLINICAL IMAGING • AI RESEARCH PLATFORM</div>
            <h1>Osteoporosis Analysis</h1>
            <p>Explore lumbar spine X-ray images with DenseNet121 classification and Grad-CAM explainability.</p>
        </div>
        <div class="header-status"><span class="status-dot"></span> SYSTEM READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 16. MODEL INFORMATION PAGE
# ============================================================

if active_page == "Model Information":
    section_header(
        "Model Information",
        "Architecture, supported categories and interpretation guidance.",
    )
    metric_cols = st.columns(3)
    metrics = [
        ("MODEL ARCHITECTURE", "DenseNet121", "Deep convolutional network"),
        ("INPUT RESOLUTION", "224 × 224", "Preprocessing pipeline preserved"),
        ("EXPLAINABILITY", "Grad-CAM", "Visual model-activation overlay"),
    ]
    for col, (label, value, detail) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f'<div class="info-card"><div><div class="info-label">{label}</div>'
                f'<div class="info-value">{value}</div><div class="metric-detail">{detail}</div></div></div>',
                unsafe_allow_html=True,
            )

    section_header("Classification Categories", "The model returns one of these three trained labels.")
    class_cols = st.columns(3)
    class_descriptions = {
        "Normal": "The model assigned the image to the Normal class.",
        "Osteopenia": "The model assigned the image to the Osteopenia class.",
        "Osteoporosis": "The model assigned the image to the Osteoporosis class.",
    }
    for col, name in zip(class_cols, CLASS_NAMES):
        with col:
            color = CLASS_DETAILS[name]["color"]
            st.markdown(
                f'<div class="class-info-card"><span class="class-indicator" style="background:{color}"></span>'
                f'<h4>{name}</h4><p>{class_descriptions[name]}</p></div>',
                unsafe_allow_html=True,
            )

    section_header("How to use this research interface")
    st.markdown(
        """
        <div class="guide-card">
            <div class="guide-row"><span>01 · Select an image</span><strong>DICOM / JPG / JPEG / PNG</strong></div>
            <div class="guide-row"><span>02 · Run the analysis</span><strong>DenseNet121 inference</strong></div>
            <div class="guide-row"><span>03 · Review model outputs</span><strong>Class scores + Grad-CAM</strong></div>
            <div class="guide-row"><span>04 · Interpret cautiously</span><strong>Research use only</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "The displayed softmax scores are model outputs, not clinically calibrated probabilities. "
        "This research prototype is not a validated diagnostic device."
    )


# ============================================================
# 17. IMAGING WORKSPACE / UPLOAD
# ============================================================

elif active_page == "Imaging Workspace":
    section_header(
        "Imaging Workspace",
        "Add a lumbar spine X-ray to inspect the source image before analysis.",
    )
    upload_col, guide_col = st.columns([1.45, 1], gap="large")

    with upload_col:
        st.markdown(
            '<div class="upload-heading"><span class="upload-heading-icon">↑</span><div>'
            '<div class="upload-title">Upload medical image</div>'
            '<div class="upload-description">Choose a supported X-ray file from your device.</div></div></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=["dcm", "jpg", "jpeg", "png"],
            help="Supported formats: DICOM, JPG, JPEG and PNG.",
            label_visibility="collapsed",
            key="medical_image_uploader",
        )

    with guide_col:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">Image requirements</div>
                <div class="guide-row"><span>Supported formats</span><strong>DICOM / JPG / PNG</strong></div>
                <div class="guide-row"><span>Image modality</span><strong>Lumbar spine X-ray</strong></div>
                <div class="guide-row"><span>Model input</span><strong>224 × 224 px</strong></div>
                <div class="guide-row"><span>Model</span><strong>DenseNet121</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">＋</div>
                <h3>Your imaging workspace is ready</h3>
                <p>Upload a supported image to preview it here. You can then move to AI Analysis to run classification and Grad-CAM.</p>
                <div class="empty-tags"><span>DICOM</span><span>JPEG</span><span>PNG</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            img_array, img_display = process_uploaded_file(uploaded_file)
            preview_col, details_col = st.columns([1.2, 1], gap="large")
            with preview_col:
                section_header("Image preview", "Review the imported image.")
                st.image(img_display, use_container_width=True)
            with details_col:
                section_header("File details")
                safe_name = html.escape(uploaded_file.name)
                st.markdown(
                    f'<div class="guide-card"><div class="guide-row"><span>Filename</span><strong>{safe_name}</strong></div>'
                    f'<div class="guide-row"><span>Image dimensions</span><strong>{img_array.shape[1]} × {img_array.shape[0]}</strong></div>'
                    f'<div class="guide-row"><span>Processed format</span><strong>Grayscale</strong></div>'
                    f'<div class="guide-row"><span>Model input</span><strong>224 × 224</strong></div></div>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "Continue to AI Analysis →",
                    type="primary",
                    use_container_width=True,
                    on_click=go_to_analysis,
                )
        except Exception as err:
            st.error(f"Image could not be previewed: {err}")


# ============================================================
# 18. AI ANALYSIS PAGE
# ============================================================

else:
    section_header(
        "AI Analysis",
        "Upload a lumbar spine X-ray and review the model classification and visual explanation.",
    )

    upload_col, guide_col = st.columns([1.45, 1], gap="large")
    with upload_col:
        st.markdown(
            '<div class="upload-heading"><span class="upload-heading-icon">↑</span><div>'
            '<div class="upload-title">Upload medical image</div>'
            '<div class="upload-description">DICOM, JPG, JPEG or PNG files are supported.</div></div></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=["dcm", "jpg", "jpeg", "png"],
            help="Supported formats: DICOM, JPG, JPEG and PNG.",
            label_visibility="collapsed",
            key="medical_image_uploader",
        )
    with guide_col:
        st.markdown(
            """
            <div class="guide-card">
                <div class="guide-title">Analysis pipeline</div>
                <div class="guide-row"><span>Image processing</span><strong>Grayscale · 224²</strong></div>
                <div class="guide-row"><span>Classification</span><strong>DenseNet121</strong></div>
                <div class="guide-row"><span>Explainability</span><strong>Grad-CAM</strong></div>
                <div class="guide-row"><span>Output classes</span><strong>3 categories</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">⌁</div>
                <h3>Ready for image analysis</h3>
                <p>Upload a lumbar spine X-ray to generate the model prediction, class scores and Grad-CAM visualization.</p>
                <div class="empty-tags"><span>DenseNet121</span><span>3 classes</span><span>Grad-CAM</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)
        try:
            img_array, img_display = process_uploaded_file(uploaded_file)
            model = load_model()

            with st.spinner("Analyzing image and generating Grad-CAM..."):
                predicted_class, confidence, all_probs, cam_image, cam_error = predict(model, img_array)

            details = CLASS_DETAILS[predicted_class]
            image_col, result_col = st.columns([1, 1], gap="large")

            with image_col:
                section_header("Uploaded X-ray", "Processed image supplied to the model.")
                st.image(img_display, use_container_width=True)
                st.markdown(
                    f'<div class="image-meta"><span>FILE</span><strong>{html.escape(uploaded_file.name)}</strong></div>',
                    unsafe_allow_html=True,
                )

            with result_col:
                section_header("Model prediction", "DenseNet121 classification output.")
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-overline">PREDICTED CLASS</div>
                        <div class="result-class" style="color:{details['color']};">{html.escape(predicted_class)}</div>
                        <div class="result-description">{html.escape(details['description'])}</div>
                        <div class="confidence-divider"></div>
                        <div class="confidence-row"><span>Top-class model score</span><strong>{confidence * 100:.2f}%</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)
            section_header("Classification breakdown", "Softmax output scores across the three target classes.")
            prob_col, note_col = st.columns([1.4, 1], gap="large")
            with prob_col:
                st.markdown('<div class="probability-card">', unsafe_allow_html=True)
                for class_name, probability in zip(CLASS_NAMES, all_probs):
                    render_probability_card(class_name, probability)
                st.markdown("</div>", unsafe_allow_html=True)
            with note_col:
                st.markdown(
                    """
                    <div class="interpretation-card">
                        <div class="interpretation-title">Understanding the scores</div>
                        <p>Percentages represent the model's softmax output scores for the three trained categories.</p>
                        <div class="interpretation-note">These values are not clinically calibrated probabilities and must not be interpreted as a confirmed diagnosis.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)
            section_header("Model explainability", "Compare the source image with the Grad-CAM activation overlay.")
            if cam_image is not None:
                original_col, cam_col = st.columns(2, gap="large")
                with original_col:
                    st.markdown('<div class="visual-heading"><h4>Original X-ray</h4><span>INPUT IMAGE</span></div>', unsafe_allow_html=True)
                    st.image(img_display, use_container_width=True)
                with cam_col:
                    st.markdown('<div class="visual-heading"><h4>Grad-CAM heatmap</h4><span>ACTIVATION OVERLAY</span></div>', unsafe_allow_html=True)
                    st.image(cam_image, use_container_width=True)
                st.markdown(
                    '<div class="gradcam-note"><strong>How to interpret:</strong> The heatmap highlights image regions associated with the model output. It does not establish the presence or absence of disease.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Prediction was generated, but Grad-CAM visualization could not be produced.")
                if cam_error:
                    with st.expander("Technical details"):
                        st.code(cam_error)

            st.markdown('<div class="content-spacer"></div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="medical-disclaimer">
                    <div class="disclaimer-icon">ⓘ</div>
                    <div><strong>Medical research disclaimer</strong>
                    <p>OsteoScan AI is an experimental research prototype for AI-assisted medical image classification. It is not a clinically validated diagnostic system. Predictions must not be used for treatment decisions. Consult a qualified healthcare professional for medical evaluation.</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except FileNotFoundError as err:
            st.error(str(err))
            st.info("Place `best_lumos_densenet.pth` in the same folder as `app.py`.")
        except ValueError as err:
            st.error(f"Image could not be processed: {err}")
        except Exception as err:
            st.error("Unable to complete image analysis.")
            with st.expander("View technical error"):
                st.code(f"{type(err).__name__}: {err}")
                st.caption("Full traceback is available in the server logs.")


# ============================================================
# 19. FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <div class="footer-brand">OsteoScan AI<span>Clinical Premium · Medical Imaging Research</span></div>
        <div class="footer-copy">DenseNet121 · Grad-CAM · Research prototype</div>
    </div>
    """,
    unsafe_allow_html=True,
)
