样本图片在 results/official_samples 目录下
图片原始标签：picked_gt0 为 real，picked_gt1 为 AI-generated
置信度已标注：对于 gt0 来说，conf > 0.76 才能被判定为 real；对于 gt1 来说，conf > 0.24 被判定为 AI-generated
在图片判别正确的前提下，选取了 **置信度最高** 的和 **在 threshold 边缘（置信度较低）** 的一些图片作为参考，conf 已在图片文件名中标记