# Photo2slides

把会议现场拍摄的投影照片恢复为校正后的、可继续编辑的 PowerPoint。

Photo2slides 先用自训练的 YOLO 分割模型定位屏幕并完成透视校正，再修正投影光照、颜色偏移、密集摩尔纹和边缘遮挡，最后用视觉模型理解版面。标题、正文等内容会按实际字形行框重建为原生文本框，图表、公式和复杂示意图则保留为可移动的独立图片。默认使用本地 PaddleOCR-VL 1.6，不需要 API Key；也可选择 OpenAI 高精度视觉模型。

## 效果

`原始照片 → 屏幕分割定位 → 透视校正 → 图像增强 → 可编辑幻灯片重建`

| 处理流程 | 实际处理结果 |
| --- | --- |
| **① 原始现场照片** | ![已模糊人脸和讲台文字的第 28 张原始现场照片](docs/images/example-28-step-01-original-redacted.png) |
| **↓ ② YOLO 屏幕分割定位**<br>识别投影屏幕区域并拟合边界。 | ![已模糊人脸和讲台文字的 YOLO 屏幕分割定位结果](docs/images/example-28-step-02-segmentation-redacted.png) |
| **↓ ③ 透视校正**<br>按照屏幕四角恢复为 16:9 正视图。 | ![透视校正结果](docs/images/example-28-step-03-rectified.png) |
| **↓ ④ 图像增强**<br>调整颜色、对比度和清晰度，为页面理解提供输入。 | ![图像增强结果](docs/images/example-28-step-04-enhanced.png) |
| **↓ ⑤ 幻灯片重建**<br>重建文本与视觉元素并写入 PPTX。 | ![最终幻灯片重建结果](docs/images/example-28-step-05-reconstructed.png) |

## 功能

- YOLO 屏幕分割、四边形拟合与透视校正
- 自动修正投影光照、颜色渐变和密集横/竖向摩尔纹
- 检测人物、讲台等边缘遮挡，并以邻近背景保守修补
- PaddleOCR-VL 页面理解与中英文文本识别
- 按逐行像素几何恢复位置、字号、颜色、粗体、斜体和对齐
- 清除已重建文字的原图字样；低置信内容保留为图片，避免误写
- `editable`、`hybrid`、`image` 三种 PPTX 输出模式
- 输出中间过程图、逐页识别 JSON 和整体构建统计

## 运行环境

下面的命令以 Windows PowerShell 为例，并且都要在仓库根目录执行。

| 依赖 | 要求 |
| --- | --- |
| Python | 64 位 Python 3.11 或 3.12 |
| Node.js | 18 或更高版本，需同时提供 npm |
| 磁盘 | 首次下载本地视觉模型时需预留数 GB 空间 |
| GPU | 可选；NVIDIA GPU 会明显加速本地页面理解，CPU 也能完成 |
| 网络 | 本地 VLM 首次下载模型时需要；OpenAI 模式每次识别都需要 |

仓库已包含推理所需的 `p2p\models\slide-seg.pt`，不需要下载 YOLO 权重。训练照片、标注和训练日志不在仓库中。

## 一次性安装

新环境可以这样安装：

```powershell
git clone https://github.com/JaylenVen/Photo2slides.git
Set-Location .\Photo2slides

py -3.12 -m venv .venv
$python = Join-Path $PWD ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r .\p2p\requirements.txt
npm install --prefix .\p2p\src
```

如果已经有可用的 Conda 或 Python 环境，不必再创建 `.venv`，只需把 `$python` 指向对应的 `python.exe`。安装后建议先检查：

```powershell
& $python --version
node --version
npm --version
Test-Path .\p2p\models\slide-seg.pt
```

最后一条必须输出 `True`。

## 准备输入照片

把同一套幻灯片的照片放入 `p2p\data\input`。支持 `.bmp`、`.jpeg`、`.jpg`、`.png`、`.tif`、`.tiff` 和 `.webp`。

- 文件会按自然数字顺序处理，例如 `1.jpg`、`2.png`、`10.jpg`。
- 不要同时使用同名不同扩展名的文件，例如 `1.jpg` 和 `1.png`，否则中间结果会互相覆盖。
- 每张照片应尽量包含完整投影屏幕；过度遮挡或完全过曝的真实内容无法从单张照片恢复。

## 运行一次完整流程

推荐为每次运行使用独立的输出名和工作目录，避免与旧结果混在一起：

```powershell
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$workDir = ".\p2p\data\intermediate\runs\$runId"
$output = ".\p2p\data\final\photo2slide-$runId.pptx"

& $python .\p2p\src\photo_to_pptx.py `
  --model .\p2p\models\slide-seg.pt `
  --input-dir .\p2p\data\input `
  --work-dir $workDir `
  --output $output `
  --device auto `
  --analysis-engine vlm `
  --analysis-device auto `
  --ppt-mode editable `
  --ocr-conf 0.70

if ($LASTEXITCODE -ne 0) {
  throw "Photo2slides 运行失败，请查看上方错误信息。"
}

Get-Item $output | Select-Object FullName, Length, LastWriteTime
```

这条命令会依次完成屏幕检测、透视校正、光照与摩尔纹修复、页面识别、可编辑元素重建和 PPTX 写入。第一次使用 `vlm` 时会下载 PaddleOCR-VL 相关模型，等待时间会明显长于后续运行。

如果不需要保留每次运行的独立目录，最短命令是：

```powershell
& $python .\p2p\src\photo_to_pptx.py
```

默认输出为 `p2p\data\final\photo2slide-vlm.pptx`，默认中间结果为 `p2p\data\intermediate\runs\latest`。

## 使用 OpenAI 高精度视觉模式

小字、复杂样式或遮挡判断要求更高时，可以改用 OpenAI 视觉模型：

```powershell
$env:OPENAI_API_KEY = "填写你自己的 API Key"

& $python .\p2p\src\photo_to_pptx.py `
  --input-dir .\p2p\data\input `
  --work-dir .\p2p\data\intermediate\runs\openai-latest `
  --output .\p2p\data\final\photo2slide-openai.pptx `
  --analysis-engine openai `
  --vision-model gpt-5.6 `
  --ppt-mode editable
```

OpenAI 模式会上传透视校正后的幻灯片图像，并产生 API 费用。不要把 API Key 写入项目文件或提交到 Git；敏感材料请使用默认本地模式。

## 如何确认 PPTX 已完整生成

程序成功结束时会打印 `PPTX：<输出路径>`、成功页数和跳过页数。随后检查：

1. 输出 `.pptx` 文件存在且大小不为 0。
2. 成功页数与预期照片数一致；如果有“跳过”，应先处理对应照片再重跑。
3. 打开 `reconstruction-summary.json`，重点查看每页的 `needsReviewTextCount` 和 `meanTextConfidence`。
4. 在 PowerPoint 中逐页检查文字、图表、公式和遮挡修复区域。

完整运行的主要产物如下：

```text
p2p/data/
├─ input/                               # 原始照片
├─ final/<输出文件>.pptx                # 所有历次最终 PPTX
└─ intermediate/
   ├─ cache/                            # 模型与下载缓存
   ├─ qa/                               # 成品渲染与审查报告
   ├─ archive/                          # 旧版中间资料
   └─ runs/<运行目录>/
      ├─ masks/                         # 屏幕分割调试图
      ├─ rectified/                     # 透视校正图
      ├─ enhanced/                      # 光照、颜色和摩尔纹修复图
      ├─ ocr/                           # 页面理解输入图
      └─ reconstruction/
         ├─ slide-XX/ocr.json           # 逐页识别结果和复核标记
         ├─ deck-manifest.json          # PPTX 构建清单
         ├─ reconstruction-summary.json # 整体统计
         └─ artifact-workspace/build-manifest.json
```

普通 Node.js 环境会使用 `pptxgenjs` 写入 PPTX。只有检测到 Codex presentations artifact runtime 时，才会额外生成 `preview`、`layout` 和 `inspect.ndjson`；这些预览文件不是判断普通运行成功的必要条件。

## 常用参数

```powershell
# 强制本地页面理解使用 CPU
& $python .\p2p\src\photo_to_pptx.py --analysis-device cpu

# 指定第一块 NVIDIA GPU
& $python .\p2p\src\photo_to_pptx.py --device 0 --analysis-device gpu

# 仅生成整页图片型 PPTX，用于快速检查屏幕检测和透视校正
& $python .\p2p\src\photo_to_pptx.py --ppt-mode image

# 提高转为可编辑文字的最低置信度；更保守，但更多文字会保留为图片
& $python .\p2p\src\photo_to_pptx.py --ocr-conf 0.75

# 查看全部参数
& $python .\p2p\src\photo_to_pptx.py --help
```

| 模式 | 用途 |
| --- | --- |
| `editable` | 推荐；原生文字/简单形状 + 独立复杂视觉块 |
| `hybrid` | 清理后的背景图 + 原生文字 |
| `image` | 整页图片，不进行页面文字重建 |

## 常见问题

- **提示模型文件不存在**：确认 `p2p\models\slide-seg.pt` 存在；这是默认权重路径。
- **Node.js 18+ is required**：安装 Node.js 后重新打开 PowerShell，再运行 `node --version`。
- **首次运行长时间没有结束**：本地 VLM 正在下载和初始化模型；保持联网并确认磁盘空间足够。
- **GPU 不可用或显存不足**：改用 `--analysis-device cpu`；YOLO 也可加 `--device cpu`。
- **某页被跳过**：该照片没有检测到足够可信的屏幕区域。可先检查 `masks`，必要时降低 `--conf`，例如 `--conf 0.15`。
- **OpenAI 模式提示缺少 Key**：仅在当前 PowerShell 会话设置 `OPENAI_API_KEY` 后重试。
- **输出中仍有不确定小字**：查看对应 `slide-XX\ocr.json` 中的 `needsReview`；低置信内容会保留为图片，不会伪装成准确文本。

## 能力边界

- 人物或讲台遮住的纯背景可按周围颜色近似补齐；被完全遮住的文字、图表数值和图像细节无法从单张照片真实恢复。
- 模糊小字会给出最佳可见读法；低于置信度阈值或被模型标为不确定的内容不会冒充准确文本，而会以原图块保留。
- 找不到原字体时会使用最接近的已安装字体；因此无法保证未知或定制字体的字面宽度完全一致。
- 复杂图表、公式和示意图仍以独立图片保留，以优先保证视觉真实性。
- 使用 OpenAI 模式会上传校正后的幻灯片图像，并产生 API 费用；敏感材料请使用默认本地模式。

## 本地维护验证

```powershell
& $python -m unittest discover -s .\p2p\tests -v
```

按项目要求，`p2p\tests` 只保留在维护者本地工作区，不上传到 GitHub；公开仓库使用者可先运行 `--help`，再用少量照片完成一次端到端冒烟验证。
