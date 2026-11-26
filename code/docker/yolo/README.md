# YOLOv8 训练与可视化平台

在本目录运行 Docker Compose。Docker Desktop 的 Containers 页面展开 `yolo`，可以看到 Vue 前端和 FastAPI 后端。网页可以发起训练，查看数据集、历史实验、ECharts 图表和测试错例，也可以导出 PNG、SVG 和原始数据。

默认从仓库根目录的 `DataSet/VarroaDataset-cls` 读取数据。换机器后，如果数据放在别处，将 `.env.example` 复制为 `.env`，把 `VARROA_DATASET_PATH` 改为包含 `train/`、`val/`、`test/` 的目录。Windows 绝对路径可以写成 `D:/datasets/VarroaDataset-cls`。

## 启动平台

从仓库根目录进入启动目录并执行以下命令。

```powershell
cd code/docker/yolo
docker compose up -d --build
```

打开 [http://127.0.0.1:8088](http://127.0.0.1:8088)。如需换端口，在 `.env` 配置 `YOLO_WEB_PORT`。Compose 项目名为 `yolo`，前端为 `yolo/frontend:local`，训练后端为 `yolo/backend:baseline`。需要 Docker Compose 2.30 或更高版本，以及支持 CUDA 12.8 的 NVIDIA 驱动。Windows 上使用 Docker Desktop 的 WSL 2 后端。[Docker Desktop GPU 支持](https://docs.docker.com/desktop/features/gpu/)

## 小样本试跑

可在网页的训练任务页勾选“小样本试跑”，将训练轮数设为 1。也可以在空闲时运行以下命令。

```powershell
docker compose exec backend python train_smoke.py
```

脚本会先执行一次 CUDA 矩阵运算，再从训练集和验证集各取每类最多 16 张 PNG，跑一轮 `YOLOv8n-cls`。原数据只读挂载，预训练权重首次运行时下载到 `runs/pretrained/`。首次试跑的结果在 `runs/smoke/` 下，重复运行会自动创建新目录。

- `results.csv` 保存每轮指标。
- `weights/best.pt` 保存训练产生的模型权重。

## 全量试跑

在同一个启动目录执行以下命令。

```powershell
docker compose exec backend python train.py --name full-trial --epochs 1 --batch 32 --imgsz 160
```

这个入口使用完整的训练集和验证集。训练前将两组图像暂存到容器内的可写目录，让 Ultralytics 可以生成校验缓存，原数据集保持只读。试跑结果保存在 `runs/full-trial/`，重跑时会自动增加编号。

除每轮指标和权重外，还会保存 `args.yaml` 和 `run_summary.json`。汇总文件记录样本数、运行参数、环境版本、复制数据耗时、训练与验证总耗时，以及 PyTorch 的峰值已分配与预留显存。测试集留到正式方案选定后评估。

## 预处理对照

两组使用同一份原始数据和预训练权重。默认组沿用 Ultralytics 的训练随机裁剪、验证中心裁剪。补边组把这些几何操作替换为等比例缩放，再用 RGB `(114, 114, 114)` 居中补边。训练增强链的其余设置保持相同。

```powershell
docker compose exec backend python train.py --name compare-default --preprocessing default --epochs 10 --batch 32 --imgsz 160 --workers 2 --seed 42
docker compose exec backend python train.py --name compare-pad --preprocessing pad --epochs 10 --batch 32 --imgsz 160 --workers 2 --seed 42
```

两组训练完成后，用各自的预处理和 Ultralytics 保存的 `weights/best.pt` 在验证集生成 `validation_metrics.json` 和 `confusion_matrix.csv`。JSON 含准确率、宏平均 F1、有螨召回率、漏判数和漏判率。矩阵的行是真实类别，列是预测类别。

`results.csv` 记录训练过程中的验证曲线，论文对照表使用加载 `best.pt` 后的复评结果。Ultralytics 保存权重时会转成半精度，两处指标可能略有差异。后续独立评估或做单图推理时，从 `run_summary.json` 读取该权重使用的预处理方案，并应用相同的几何处理。

## 多种子汇总与最终测试

以相同参数再运行种子 123 和 2026，每种预处理各跑一次。运行名在上面的名称后分别加 `-s123` 和 `-s2026`。三组完成后按验证集有螨召回率选定方法，召回率相同时比较宏平均 F1。以下命令生成对照报告，并只用选定的权重评估保留的测试集。

```powershell
docker compose exec backend python compare.py --default compare-default compare-default-s123 compare-default-s2026 --pad compare-pad compare-pad-s123 compare-pad-s2026
$winner = (Get-Content -LiteralPath 'runs/study_summary.json' -Raw | ConvertFrom-Json).selection.run_id
docker compose exec backend python evaluate.py --run "/runs/$winner" --split test --output /runs/final-test
```

输出保存在 `runs/study_summary.json` 和 `runs/final-test/`。后者包含测试指标、混淆矩阵和逐图预测，网页会读取这些文件。该命令保留已有的测试结果目录。所有 `runs/` 文件留在本机。

## 追加准确率优化

以原最终模型 `compare-default-s2026` 为参照。追加比较默认裁剪、224×224 输入、30 轮的 YOLOv8n-cls 和 YOLOv8s-cls，批大小 32，随机种子 2026。可以从网页选择相同的模型和训练参数，或在容器中用 `train.py --model` 发起。

对原模型和两个候选在验证集分别导出逐图有螨概率。决策阈值的候选范围为 0.30 到 0.70，步长 0.025。先保留验证有螨召回率达到原方案的组合，再按验证准确率、宏平均 F1 和有螨召回率排序。选定后以固定权重、输入尺寸和阈值复评同一批 3,408 张测试图。原方案仍作为对照保存在 `runs/final-test/`。

```powershell
$nRun = (Get-Content -LiteralPath 'runs/optimization-n-run.txt').Trim()
$sRun = (Get-Content -LiteralPath 'runs/optimization-s-run.txt').Trim()
foreach ($run in 'compare-default-s2026', $nRun, $sRun) {
    docker compose exec backend python evaluate.py --run "/runs/$run" --split val --output "/runs/optimization-val/$run"
    if ($LASTEXITCODE -ne 0) { throw "评估失败：$run" }
}
docker compose exec backend python optimize.py --baseline compare-default-s2026 --candidates $nRun $sRun
if ($LASTEXITCODE -ne 0) { throw '验证集筛选失败' }
$winner = (Get-Content -LiteralPath 'runs/optimization_summary.json' -Raw | ConvertFrom-Json).selection
docker compose exec backend python evaluate.py --run "/runs/$($winner.run_id)" --split test --mite-threshold ($winner.threshold.ToString([Globalization.CultureInfo]::InvariantCulture)) --output /runs/optimized-test
```

网页的“最终测试”页显示验证集筛选规则，并并排展示原方案和优化候选的测试指标。复评结果与原论文的预处理对照分开记录。

## 分类算法对照

在「训练任务 → 有无分类训练」中可选 ResNet18。使用 TorchVision 的 ImageNet 预训练权重初始化，换成有螨和无螨两个类别后，用项目的训练数据微调。与已有 YOLOv8n-cls 和 YOLOv8s-cls 的对照固定官方数据划分、默认裁剪、224×224 输入、30 轮、批大小 32 和种子 2026。ResNet18 使用 ImageNet 归一化和 AdamW，详细参数保存在 `args.yaml`。 [ResNet 原论文](https://arxiv.org/abs/1512.03385) · [TorchVision ResNet18](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet18.html)

在 `code/docker/yolo/` 下运行以下命令。`$resnet` 是网页训练完成后显示的实验编号。脚本核对测试图片、标签、训练设置和 0.50 阈值，并保存分类补充实验报告 `runs/classifier_comparison.json`。

```powershell
$resnet = 'default-8ba5a75afc'
$small = 'default-585520c466'
foreach ($run in $resnet, $small) {
    docker compose exec backend python evaluate.py --run "/runs/$run" --split test --mite-threshold 0.50 --output "/runs/$run/classifier-test"
    if ($LASTEXITCODE -ne 0) { throw "分类测试失败 $run" }
}
docker compose exec backend python compare_classifiers.py --reference /runs/optimized-test --candidates "/runs/$small/classifier-test" "/runs/$resnet/classifier-test"
```

| 官方测试集（3,408 张） | 准确率 | 宏平均 F1 | 有螨召回率 | 有螨漏判 | 无螨误报 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n-cls | 93.10% | 91.33% | 86.62% | 126 | 109 |
| YOLOv8s-cls | 92.87% | 90.79% | 82.06% | 169 | 74 |
| ResNet18 | 92.58% | 90.37% | 80.89% | 180 | 73 |

更新运行中的平台时，从 `code/docker/yolo/` 执行 `python update.py`。脚本在重建容器前检查训练任务。YOLOv8 分类任务因后端重启而中断时，可以在该实验记录点击「从检查点续训」，平台会用 `last.pt` 中保留的训练状态接着跑，并继续写入原实验的曲线与日志。

## 上传照片检测瓦螨

网页的「瓦螨检测」页接受单只蜜蜂的 PNG、JPG 或 WebP 近照。检测模型直接处理照片，显示找到的疑似瓦螨及位置。每张照片最多 10 MB 和 2500 万像素。

### 定位模型对照和图表

在「算法对照」填写每个模型的训练轮数、输入尺寸、批大小和随机种子，点击「开始定位对照」。平台依次训练 YOLOv3-tiny、YOLOv5n、YOLOv7-tiny、YOLOv8n 和 Faster R-CNN（ResNet18-FPN）。五个模型使用同一份训练、验证和测试划分及瓦螨框标注。

默认采用随机初始化。选择预训练初始化时，YOLO 使用各自的官方检测权重，Faster R-CNN 使用 ImageNet 预训练的 ResNet18 骨干并随机初始化检测头。Faster R-CNN 每次最多前向计算 4 张，再累积梯度到设置的批大小。各模型保留原生训练器的损失及优化器实现。

「对照记录」会显示各模型的状态和训练日志入口。所有模型完成后，该配置的测试指标和图表会自动展示。可以修改参数另起一组，或从「选择对照实验」回看已保存的结果。分组记录保存在挂载的 `runs/.comparisons/`，每个模型的产物保存在各自的运行目录。

也可以在「训练任务 → 瓦螨定位训练」单独启动任一模型。训练单个 YOLOv8n 对照时勾选「作为定位对照」。旧版源码和预训练权重按需下载到 `runs/legacy-sources/` 和 `runs/pretrained/`。

在「算法对照」选择一组已完成的配置。「同一张测试图的画框对照」并排显示人工标注框和各实验保存的预测框。每个模型的置信度阈值用同一验证集和选择规则确定。主表与 CSV 使用统一的框 mAP、框精确率、框召回率、框 F1、漏检框和误报框。测试集 P–R 曲线使用 IoU 0.50。每个模型另有一张从预测框推得的单图混淆矩阵。训练与验证框损失、每轮验证 mAP 用来观察各自的收敛趋势。图表支持 PNG 和 SVG。

旧版实现固定到 [YOLOv3 v9.6.0](https://github.com/ultralytics/yolov3/releases/tag/v9.6.0)、[YOLOv5 v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0) 和 [YOLOv7 仓库固定提交](https://github.com/WongKinYiu/yolov7/tree/a207844b1ce82d204ab36d87d496728d3d2348e7)。对照任务保存训练配置、权重、测试报告和逐图检测结果，便于核对论文图表。

在网页打开「训练任务 → 瓦螨定位训练」，设置初始权重、轮数、输入尺寸、批大小和随机种子，点击「开始定位训练」。默认是从官方预训练权重开始，输入 384、批大小 16、18 轮、种子 2026。选择「当前检测模型（继续微调）」时，实验记录会保存初始模型的实验编号。

网页任务先训练 YOLOv8n 目标检测模型，再用验证集选阈值，最后评估官方测试集并试跑一次上传推理。评估成功后，检测页显示新实验编号并自动使用该实验的权重。可从检测页的「查看此模型的训练记录」返回相应实验，查看每轮 mAP、训练日志和测试指标，下载配置、曲线数据、检测报告和逐图预测。前两版由后端命令创建的检测实验也可在同一页面查询。

换机器训练时，从仓库根目录用原始 `gt.csv` 转换一次检测数据集。生成的 `runs/detection-dataset/` 随运行目录挂载，已有目录可直接使用。然后启动平台，在网页开始训练。

```powershell
python code/backend/prepare_detection.py --data DataSet/VarroaDataset-cls --annotations DataSet/mate/gt.csv --output code/docker/yolo/runs/detection-dataset
cd code/docker/yolo
docker compose up -d --build
```

也可以从 `code/docker/yolo/` 用命令行运行相同的检测训练和评估。

```powershell
$run = 'mite-detection-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
docker compose exec backend yolo detect train model=yolov8n.pt data=/runs/detection-dataset/dataset.yaml imgsz=384 batch=16 epochs=18 workers=4 amp=False mosaic=0 seed=2026 device=0 plots=False project=/runs name=$run
if ($LASTEXITCODE -ne 0) { throw '瓦螨模型训练失败' }
docker compose exec backend python evaluate_detection.py --run "/runs/$run" --data /runs/detection-dataset --nms-iou 0.45 --retire-previous
```

平台从 `runs/mite-detection*` 中读取训练时间最新、同时具备 `weights/best.pt` 和完整 `detection_report.json` 的检测实验。评估并完成一次 CPU 推理检查后，报告生效，网页自动切换并显示当前实验名。`--retire-previous` 清理旧检测实验的 `best.pt` 和 `last.pt`。测试报告和逐图结果 `detection_predictions.csv` 可用于论文对照。当前检测权重和报告可随运行目录复制到另一台机器使用。

检测训练使用 8,223 张训练图和 2,903 个瓦螨标注框。`mite-detection-20261125-1646` 从首版最佳权重继续微调 6 轮，使用 `hsv_v=0.50`、AdamW、起始学习率 `0.0005` 和 1 轮预热。两版均从验证集选择判断阈值，并在同一批 3,408 张官方测试图上评估。

| 指标 | 首版 `mite-detection` | 第二版 `mite-detection-20261125-1646` |
|---|---:|---:|
| 置信度阈值 | 0.30 | 0.20 |
| 单图有无判断准确率 | 93.43% | 93.16% |
| 有螨图片召回率 | 82.06% | 87.37% |
| 位置检测 mAP@0.50 | 77.98% | 81.00% |
| 漏检的有螨图片 | 169 | 119 |
| 误报的无螨图片 | 55 | 114 |

两版在三张来自不同拍摄条件的独立公开近照上均漏检。官方测试结果的适用范围是 VarroaDataset 的拍摄条件。外部抽查照片见 [USDA 蜂巢近照](https://commons.wikimedia.org/wiki/File:Apis_mellifera76-300.jpg)、[Stefan de Konink 近照](https://commons.wikimedia.org/wiki/File:Honey_bee_with_Deformed_Wing_Virus_and_Varroa_destructor.jpg)、[USGS 近照](https://commons.wikimedia.org/wiki/File:Apis_mellifera_and_Verroa2_2019-09-06-19.00.03_ZS_PMax_UDR_(48690841378).jpg)。

### 减少重复圈选

以下对照使用 `mite-detection-20261125-1646` 的 `best.pt`，输入尺寸 384、置信度阈值 0.20。用验证集比较重叠框抑制设置后，选用 `evaluate_detection.py --nms-iou 0.45`。网页读取检测报告的 `nms_iou` 参数。早期报告默认使用 0.70。

| 同一官方测试集上的位置指标 | 重叠阈值 0.70 | 重叠阈值 0.45 |
|---|---:|---:|
| 错误框 | 487 | 214 |
| 漏掉的标注框 | 190 | 208 |
| 框精确率（IoU ≥ 0.50） | 64.32% | 80.07% |
| 框召回率（IoU ≥ 0.50） | 82.21% | 80.52% |
| 框 F1（IoU ≥ 0.50） | 72.17% | 80.30% |

单图混淆矩阵相同，有螨图片仍漏检 119 张，无螨图片仍误报 114 张。切换前的报告和逐图结果保存在该运行的 `before-overlap-filter/`。

## 查看训练集里的人工标注

在网页打开「瓦螨检测 → 查看训练集标注」，选择训练集原始 PNG，或在样本列表按文件名、`bee_id` 查找。黄色圈按原始人工标注绘制。上传控件在浏览器按照片内容匹配训练集，文件名可以更改。保持图片内容原样，裁剪或另存后的图需要重新使用原图。

换机器后，在仓库根目录使用原训练集的 `gt_one.csv` 生成位置索引。

```powershell
python code/backend/locations.py --data DataSet/VarroaDataset-cls --annotations DataSet/mate/train/train/gt_one.csv --output code/docker/yolo/runs/train-locations.json
```

`runs/train-locations.json` 通过现有的运行目录挂载提供给后端。重复的人工标注取首条记录，无有效面积的位置框在页面单独提示。此页面读取原始训练标注，论文模型的训练和评估口径沿用单图二分类。
