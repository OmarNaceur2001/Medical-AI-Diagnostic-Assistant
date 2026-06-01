import io, base64, torch, torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helper: load any checkpoint ──────────────────────────────────────────────
# ── Helper: load any checkpoint ──────────────────────────────────────────────
def load_model(path, num_classes, hidden_units=None):
    ckpt = torch.load(path, map_location=DEVICE)
    m    = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if hidden_units:
        # Multi-layer classifier: 1280 → hidden → num_classes
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, hidden_units),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, num_classes),
        )
    else:
        # Simple classifier: 1280 → num_classes
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE).eval()
    return m, ckpt.get("threshold", 0.5)


# ── Load both models ──────────────────────────────────────────────────────────
breast_model, breast_threshold = load_model("model/final_model.pth", num_classes=2)
brain_model,  brain_threshold  = load_model("model/brain_model.pth",  num_classes=4, hidden_units=256)
# Brain dataset has 4 classes: glioma, meningioma, notumor, pituitary

BRAIN_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── Transforms (same for both) ────────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform = get_transform()

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, model, input_tensor, class_idx):
        output = model(input_tensor)
        model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

breast_gradcam = GradCAM(breast_model, breast_model.features[-1])
brain_gradcam  = GradCAM(brain_model,  brain_model.features[-1])

# ── GradCAM overlay image ─────────────────────────────────────────────────────
def make_gradcam_image(pil_img, cam):
    img_np  = np.array(pil_img.resize((160, 160)))
    heatmap = cv2.resize(cam, (160, 160))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(axes,
                               [img_np, heatmap, overlay],
                               ["Original", "GradCAM heatmap", "Overlay"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/predict/breast")
async def predict_breast(image: UploadFile = File(...)):
    contents = await image.read()
    pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    # Breast histopathology patches are always 50x50
    # Warn if image looks nothing like a patch
    w, h = pil_img.size
    if w > 150 or h > 150:
        return JSONResponse({
            "error": "Image dimensions suggest this may not be a histopathology patch. "
                     "Expected ~50×50px tissue patches."
        }, status_code=400)
    tensor   = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_()

    with torch.enable_grad():
        output         = breast_model(tensor)
        probs          = torch.softmax(output, dim=1)[0]
        prob_malignant = probs[1].item()

    pred_class  = int(prob_malignant > breast_threshold)
    cam         = breast_gradcam.generate(breast_model, tensor, pred_class)
    gradcam_b64 = make_gradcam_image(pil_img, cam)

    return JSONResponse({
        "prediction":     "Malignant" if pred_class == 1 else "Benign",
        "confidence":     round(prob_malignant * 100, 1),
        "threshold_used": round(breast_threshold, 2),
        "gradcam_image":  gradcam_b64
    })

@app.post("/predict/brain")
async def predict_brain(image: UploadFile = File(...)):
    contents = await image.read()
    pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    # Brain MRIs are typically much larger than 50x50
    w, h = pil_img.size
    if w < 100 or h < 100:
        return JSONResponse({
            "error": "Image too small to be an MRI scan. "
                     "Expected a standard brain MRI image."
        }, status_code=400)
    tensor   = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_()

    with torch.enable_grad():
        output     = brain_model(tensor)
        probs      = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    cam         = brain_gradcam.generate(brain_model, tensor, pred_class)
    gradcam_b64 = make_gradcam_image(pil_img, cam)

    return JSONResponse({
        "prediction":    BRAIN_CLASSES[pred_class],
        "confidence":    round(confidence * 100, 1),
        "all_probs":     {c: round(probs[i].item() * 100, 1) for i, c in enumerate(BRAIN_CLASSES)},
        "gradcam_image": gradcam_b64
    })