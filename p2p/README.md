# photo2slide

把会议现场拍摄的幻灯片照片转换为分层、可继续编辑的 PPTX。默认流程只使用本地视觉语言模型 PaddleOCR‑VL 1.6 做页面理解，不需要 API Key；需要更强的小字、样式和遮挡判断时，可切换到 OpenAI 高精度视觉模型。

## 默认流程

1. YOLO 分割模型定位照片中的投影屏幕。
2. 根据分割掩膜拟合四边形，完成透视校正。
3. 修正投影光照与色偏，抑制密集横/竖向摩尔纹，并保守修补人物、讲台等边缘遮挡。
4. 视觉模型读取恢复后的整页图像，识别标题、正文、图片、图表、表格、公式和前景遮挡。
5. 高置信文本按实际逐行字形框重建，恢复位置、字号、颜色、粗斜体和对齐；简单矩形重建为原生形状。
6. 图表、公式、照片和复杂示意图保留为独立可移动的图片对象。
7. 从背景中完整移除已重建文字；低置信文字以原图块保留，避免背景残字和错误转写同时出现。
8. 由 Node.js 写入 PPTX；普通环境使用 `pptxgenjs`，检测到 Codex presentations artifact runtime 时会额外生成逐页预览和布局检查文件。

这是一种“可编辑文字 + 保真视觉块”的混合重建。它不会把被人物、讲台或弹窗完全挡住的像素伪造为真实内容。

## 目录

```text
p2p/
├─ data/
│  ├─ input/                 # 原始照片
│  ├─ final/                 # 所有历次最终 PPTX
│  └─ intermediate/
│     ├─ cache/              # 模型与下载缓存
│     ├─ runs/latest/        # 默认中间结果
│     ├─ qa/                 # 成品渲染与审查报告
│     └─ archive/            # 旧版中间资料
├─ models/                   # 运行时 YOLO 推理权重
├─ src/                      # 主流程与 PPTX 构建器
├─ tests/                    # 本地测试（Git 忽略，不上传）
├─ training/                 # 本地训练资料（Git 忽略，不参与日常运行）
└─ requirements.txt
```

## 安装

以下命令在仓库根目录执行：

```powershell
D:\Anaconda\envs\p2p\python.exe -m pip install -r p2p\requirements.txt
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

PaddleOCR‑VL 和版面模型会在第一次运行时下载到 `p2p\data\intermediate\cache`，后续运行直接复用。

## 手动跑完整流程

1. 把照片放入 `p2p\data\input`。文件名按自然数字顺序排序，例如 `1.jpg`、`2.png`、`10.jpg`。
2. 确认默认 YOLO 权重存在：`p2p\models\slide-seg.pt`。
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

如需使用高精度 OpenAI 视觉理解：

```powershell
$env:OPENAI_API_KEY = "..."
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py `
  --analysis-engine openai `
  --vision-model gpt-5.6
```

OpenAI 模式会上传透视校正后的幻灯片图像，并产生 API 费用。敏感材料请继续使用默认本地模式。

成功后查看：

- 最终文件：`p2p\data\final\photo2slide-vlm.pptx`
- 页面识别结果：`p2p\data\intermediate\runs\latest\reconstruction\slide-XX\ocr.json`
- 整体统计：`p2p\data\intermediate\runs\latest\reconstruction\reconstruction-summary.json`
- 构建信息：`p2p\data\intermediate\runs\latest\reconstruction\artifact-workspace\build-manifest.json`

普通环境不会自动生成验收报告。仅在检测到 Codex presentations artifact runtime 时，才会额外生成 `preview`、`layout` 和 `inspect.ndjson`。

## 常用参数

```powershell
# 指定 GPU；若 PyTorch 无法访问 CUDA，此命令会明确报错
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py --analysis-device gpu

# 只生成整页图片型 PPT，用于快速检查分割和透视校正
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py --ppt-mode image

# 调整转为可编辑文字的最低置信度；默认 0.70
D:\Anaconda\envs\p2p\python.exe p2p\src\photo_to_pptx.py --ocr-conf 0.75

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
- 人物或讲台遮住的纯背景：根据邻近颜色和纹理近似补齐。
- 完全遮住的文字、图表数值或图像细节：无法从单张照片真实恢复；项目不会把模型猜测伪装成原稿。
- 模糊小字：视觉模型可给出最佳可见读法；低置信或不确定内容保留为图片，并在 `ocr.json` 中标为 `needsReview`。
- 未知或未安装的定制字体：使用最接近的已安装字体，因此无法保证字面宽度绝对一致。

## 本地维护验证

```powershell
D:\Anaconda\envs\p2p\python.exe -m unittest discover -s p2p\tests -v
```

`p2p\tests` 按项目要求只保留在维护者本地工作区，不上传到 GitHub。PPTX 生成后应在 PowerPoint 中逐页检查；若当前环境生成了 `preview`，也应检查每一张预览，不能只凭程序成功退出判断版式合格。

面向首次使用者的完整安装、运行、产物检查和故障排查说明见仓库根目录的 `README.md`。
