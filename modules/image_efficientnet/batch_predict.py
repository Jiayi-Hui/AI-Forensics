import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# ── 1. 加载训练好的权重 ──
WEIGHT_PATH = r"F:\HYH_LocalFile\8307_GroupProject\weights\efficientnet_b7_deepfake.pth"
model = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=1)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
model.eval()
print("✅ 已加载训练好的 Deepfake 检测权重")

# ── 2. 图像预处理 ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 3. 推理函数 ──
# 注意：类别映射 fake=0, real=1
# sigmoid 输出值越高 → 越像 Real；越低 → 越像 Fake
def predict_folder(folder_path, true_label):
    results = []
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 {true_label} 文件夹 → {len(files)} 张图像")

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                real_prob = torch.sigmoid(model(tensor)).item()
            fake_prob = 1.0 - real_prob   # 转换为 Fake 概率
            results.append({
                'file': fname,
                'true_label': true_label,
                'fake_prob': round(fake_prob, 4),
                'real_prob': round(real_prob, 4),
                'pred_label': 'FAKE' if fake_prob > 0.5 else 'REAL',
                'correct': (true_label == 'FAKE') == (fake_prob > 0.5)
            })
        except Exception as e:
            print(f"  ❌ {fname} 失败: {e}")
    return results

# ── 4. 执行推理 ──
base = r"F:\HYH_LocalFile\8307_GroupProject\test_images"
all_results = []
all_results += predict_folder(os.path.join(base, 'real'), true_label='REAL')
all_results += predict_folder(os.path.join(base, 'fake'), true_label='FAKE')

# ── 5. 保存结果 ──
os.makedirs('results', exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv(r'F:\HYH_LocalFile\8307_GroupProject\results\predictions.csv', index=False)

# ── 6. 统计报告 ──
print("\n📊 Fake 概率分布统计：")
print(df.groupby('true_label')['fake_prob'].describe().round(4))

print("\n🎯 分类准确率：")
for label in ['REAL', 'FAKE']:
    sub = df[df['true_label'] == label]
    acc = sub['correct'].mean()
    print(f"  {label}: {acc*100:.1f}%  ({sub['correct'].sum()}/{len(sub)} 正确)")

overall = df['correct'].mean()
print(f"  总体准确率: {overall*100:.1f}%")
print(f"\n✅ 结果已保存至 results/predictions.csv")