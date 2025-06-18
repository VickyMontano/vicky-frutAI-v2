import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
import torch.nn as nn
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from collections import defaultdict
import torchvision.ops as ops
import urllib.request
import os

# 🔧 Descarga archivos solo si no existen
def descargar_si_falta(path_local, url):
    if not os.path.exists(path_local):
        print(f"⏬ Descargando {path_local} desde la nube...")
        urllib.request.urlretrieve(url, path_local)

# 🧠 Carga del modelo EfficientNet
def cargar_modelo(path_modelo, device):
    class_names = ['Anana', 'Banana', 'Coco', 'Frutilla', 'Higo',
                   'Manzana', 'Mora', 'Naranja', 'Palta', 'Pera']

    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    model.load_state_dict(torch.load(path_modelo, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return model, transform, class_names

# 🧩 Segmentación con SAM
def segmentar_frutas(image_np, device):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    descargar_si_falta(sam_checkpoint,
        "https://huggingface.co/VickyMontano03/frutai-models/resolve/main/sam_vit_h_4b8939.pth")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device).eval()

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=12,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
        min_mask_region_area=10000,
        crop_n_layers=0
    )

    confidence_threshold = 0.9
    masks = mask_generator.generate(image_np)

    boxes, scores = [], []
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        x, y, w, h = cv2.boundingRect(seg)
        boxes.append([x, y, x + w, y + h])
        scores.append(mask["predicted_iou"])

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    masks = [masks[i] for i in indices]
    filtered_masks = [mask for mask in masks if mask["predicted_iou"] >= confidence_threshold]
    print(f"🔍 Después del filtro quedan {len(filtered_masks)} máscaras.")

    final_boxes = []
    for i in indices:
        if scores[i] >= confidence_threshold:
            x1, y1, x2, y2 = boxes[i]
            final_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    return final_boxes

# 🍓 Clasificación de frutas por bounding boxes
def clasificar_imagen(image_np, boxes, model, transform, class_names, device):
    detecciones = []

    for (x, y, w, h) in boxes:
        pad = int(0.1 * max(w, h))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, image_np.shape[1])
        y2 = min(y + h + pad, image_np.shape[0])

        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        crop_resized = cv2.resize(crop, (300, 300))
        pil_crop = Image.fromarray(crop_resized)
        tensor = transform(pil_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(1).item()
            conf = torch.softmax(output, dim=1)[0, pred].item()

        detecciones.append({
            "label": class_names[pred],
            "conf": conf,
            "bbox": (x1, y1, x2 - x1, y2 - y1)
        })

    filtradas = []
    if len(detecciones) > 0:
        agrupadas = defaultdict(list)
        for d in detecciones:
            agrupadas[d["label"]].append(d)

        for grupo in agrupadas.values():
            mejor = max(grupo, key=lambda d: d["conf"])
            if mejor["conf"] > 0.45:
                filtradas.append(mejor)

    return filtradas
