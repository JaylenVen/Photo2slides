# photo2slide

把会议现场拍摄的幻灯片照片转换为分层、可继续编辑的 PPTX。默认流程只使用本地视觉语言模型 PaddleOCR‑VL 1.6 做页面理解，不使用 DeepSeek，也不需要 API Key。VLM 与版面模型统一运行在官方 Transformers/PyTorch 后端。

## 默认流程

1. YOLO 分割模型定位照片中的投影屏幕。
2. 根据分割掩膜拟合四边形，完成透视校正、颜色与清晰度增强。
3. PaddleOCR‑VL 1.6 直接读取校正后的整页图像，识别标题、正文、图片、图表、表格和公式区域。
4. 高置信文本重建为原生 PPT 文本框；简单矩形重建为原生形状。
5. 图表、公式、照片和复杂示意图保留为独立可移动的图片对象。
6. 从背景中移除已重建文字，再由 Artifact Tool 写入 PPTX，并逐页生成预览和布局检查文件。

这是一种“可编辑文字 + 保真视觉块”的混合重建。它不会把被人物、讲台或弹窗完全挡住的像素伪造为真实内容。

## 目录

```text
p2p/
├─ data/
│  ├─ input/                 # 原始照片
│  ├─ cache/                 # 首次运行下载的官方模型缓存
│  ├─ output/                # 最终 PPTX 与验收报告
│  ├─ work/latest/           # 最近一次运行的中间结果和逐页预览
│  └─ archive/               # 旧结果归档
├─ src/                      # 主流程与 PPTX 构建器
├─ tests/                    # 单元测试
└─ requirements-vision.txt
```

## 安装

以下命令在仓库根目录执行：

```powershell
D:\Anaconda\envs\p2p\python.exe -m pip install -r p2p\requirements-vision.txt
```

默认流程依赖 PyTorch。使用 NVIDIA GPU 时，可先确认当前 PyTorch 能看到 CUDA：

```powershell
D:\Anaconda\envs\p2p\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出末尾为 `True` 即可使用 GPU；否则仍可加 `--analysis-device cpu` 运行，但速度会明显变慢。默认 VLM 路径不需要安装 PaddlePaddle。

安装 PPTX 写入依赖：

```powershell
Set-Location p2p\src
npm install
Set-Location ..\..
```

PaddleOCR‑VL 和版面模型会在第一次运行时从官方 ModelScope 仓库自动下载到 `p2p\data\cache\paddlex`，后续运行直接复用。

## 手动跑完整流程

1. 把照片放入 `p2p\data\input`。文件名按自然数字顺序排序，例如 `1.jpg`、`2.png`、`10.jpg`。
2. 确认 YOLO 权重存在：`yolo26\runs\segment\slide_seg_9_1\weights\best.pt`。
3. 在仓库根目录运行：

```powershell
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py `
  --analysis-engine vlm `
  --analysis-device auto `
  --ppt-mode editable
```

其中 `vlm`、`auto` 和 `editable` 都是默认值，因此也可以直接运行：

```powershell
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py
```

成功后查看：

- 最终文件：`p2p\data\output\photo2slide-vlm.pptx`
- 验收报告：`p2p\data\output\qa-report.txt`
- 最终渲染：`p2p\data\output\qa\rendered`
- 逐页预览：`p2p\data\work\latest\reconstruction\preview`
- 页面识别结果：`p2p\data\work\latest\reconstruction\slide-XX\ocr.json`
- 整体统计：`p2p\data\work\latest\reconstruction\reconstruction-summary.json`

## 常用参数

```powershell
# 指定 GPU；若 PyTorch 无法访问 CUDA，此命令会明确报错
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py --analysis-device gpu

# 只生成整页图片型 PPT，用于快速检查分割和透视校正
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py --ppt-mode image

# 指定输入、输出和中间目录
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py `
  --input-dir D:\photos `
  --output D:\results\slides.pptx `
  --work-dir D:\results\work
```

## 可编辑范围与真实性边界

- 标题、正文、页码等文本：原生文本框。
- 高置信简单矩形：原生形状。
- 图表、公式、复杂示意图和照片：独立图片对象，可移动、裁剪、缩放或替换。
- 完全遮挡的区域：无法从单张照片真实恢复；项目不会把模型猜测伪装成原稿。
- 超分辨率只能改善观感和模型输入，不能保证恢复真实小字、图表数值或原始高清素材。

## 验证

```powershell
D:\Anaconda\envs\p2p\python.exe -m unittest discover -s p2p\tests -v
```

PPTX 生成后还应检查 `preview` 中的每一页，并运行 Presentations 工具自带的越界检查；不能只凭程序成功退出判断版式合格。
