import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ── 加载训练好的权重 ──
WEIGHT_PATH = r"F:\HYH_LocalFile\8307_GroupProject\weights\efficientnet_b7_deepfake.pth"
model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=1)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
model.eval()
print("✅ 权重加载成功")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── GradCAM 设置 ──
target_layer = model.blocks[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ── 对一批图像生成热力图 ──
def generate_heatmaps(folder_path, label, max_images=5):
    out_dir = f"F:\\HYH_LocalFile\\8307_GroupProject\\heatmaps\\{label}"
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_images]

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img_pil = Image.open(img_path).convert('RGB')
        tensor = transform(img_pil).unsqueeze(0)

        # 获取概率
        with torch.no_grad():
            real_prob = torch.sigmoid(model(tensor)).item()
        fake_prob = 1.0 - real_prob

        # 生成 CAM
        grayscale_cam = cam(input_tensor=tensor)[0]

        # 叠加原图
        rgb_img = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 保存，文件名含概率
        out_name = f"{fname.split('.')[0]}_fake{fake_prob:.2f}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, visualization[:, :, ::-1])
        print(f"  [{label}] {fname} → Fake概率: {fake_prob:.4f} → {out_path}")

# ── 分别对 real 和 fake 各生成 5 张 ──
base = r"F:\HYH_LocalFile\8307_GroupProject\test_images"
print("\n🔍 生成 REAL 图像热力图...")
generate_heatmaps(os.path.join(base, 'real'), 'REAL', max_images=5)

print("\n🔍 生成 FAKE 图像热力图...")
generate_heatmaps(os.path.join(base, 'fake'), 'FAKE', max_images=5)

print("\n✅ 所有热力图保存至 heatmaps/ 文件夹")