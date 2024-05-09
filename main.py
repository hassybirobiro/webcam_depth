import cv2

import torch
from torchvision import transforms
import numpy as np

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def estimate(img):
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = 255

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
    return out

def normalize_depth(depth, bits):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
    if bits == 1:
        return out.astype("uint8")
    elif bits == 2:
        return out.astype("uint16")


# ----------------


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1000, 600))
    out = estimate(frame)
    out = normalize_depth(out, bits=2)
    cv2.imshow("Web Camera movie", frame)
    cv2.imshow("Depth Estimation", out)
    
    i = cv2.waitKey(1)
    if i == 27 or i == 13:
        break

cap.release()
cv2.destroyAllWindows()
