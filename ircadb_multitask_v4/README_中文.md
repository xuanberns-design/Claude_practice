# 3D-IRCADb-01 2.5D 重建与分割联合模型 · V4

## V4 更新（本次）

V4 针对 V3 结果中的两个问题：**32 视角条纹伪影明显、重建指标比 FBP 提高有限**，以及**肿瘤分割不准、肝外假阳性多**。完整诊断、每项改动的依据和合成消融结果见 [V4优化说明](docs/V4优化说明.md)。新增参数见 [参数说明](docs/参数说明.md) 末尾“V4 新增参数”一节。

- **双域重建**（`reconstruction_mode: dual_domain`）：流程为稀疏正弦图 → 角度插值 + 正弦图 U-Net 补全缺失角度 → 已测角度硬数据一致性 → 与 skimage 逐像素一致的可微 FBP → 图像域 RCAB U-Net 精修。改为整图训练，加入精确的正弦图同步几何增强，以及正弦图、双域中间图、频域、投影一致性损失。
- **分割**（`seg_backbone: resattn`）：残差注意力编码器 + 注意力门 + 深监督 + 以肝概率为条件的肿瘤头 + 宽窗输入；肿瘤使用逐样本 Focal-Tversky 与层级一致性损失；分割满权重后才开始选 best.pt。
- **3D 后处理**：肝取最大连通域并填洞，肿瘤限制在预测肝 ±5 mm 内并删除小碎片；阈值和最小体积**只在验证集**上自动选择。新增 HD95/ASSD 与病灶级检出率指标。
- **训练**：EMA 权重、学习率预热、分割 TTA。
- **兼容**：新参数默认值等于 V3 行为；V3 的 YAML 和权重可原样评估。后处理可以直接用于 V3 的 best.pt，无需重训。

**当前包内测试结果的定义**：目标 tumor 仅指专家肝脏 mask 内的肿瘤体素。patient 7 虽有 `tumor` DICOM 目录，但其病灶主要位于肝外，因此本任务的目标真值为 0 ml；不能把它的肝外病灶当作肝内肿瘤漏检。当前 `runs/v4_dualdomain256/evaluation_test` 的 32 views 结果中，patient 6 的肿瘤真值/预测体积为 425.35/118.69 ml、Dice 0.407；patient 8 为 10.23/1.54 ml、Dice 0.040。两者的主要问题是漏检与低估体积，且 64/128 views 也存在，不能只靠增强 32 views 的重建指标解决。详见 [32 views 改进路径](docs/32views改进路径.md)。

V4 快速开始（复用 V3 的 256 缓存，prepare 只补充正弦图）：

~~~powershell
python -m src.cli --config configs/v4_dualdomain_256.yaml prepare
python -m src.cli --config configs/v4_dualdomain_256.yaml train
python -m src.cli --config configs/v4_dualdomain_256.yaml evaluate --checkpoint runs/v4_dualdomain256/best.pt --split test
~~~

### 32 views 肝内肿瘤改进

完整的两版流程和结果判读见 [32 views 改进路径](docs/32views改进路径.md)。在有原 `best.pt` 和患者缓存的机器上，不重训版先运行 `evaluate --split val --no-export --save-probabilities`，再运行 `tune-32-postprocess`，固定参数后重评验证集和测试集；校准只改 32 views，64/128 views 延用原参数。需要重训版使用 [分割微调配置](configs/v4_32_seg_finetune.yaml) 固定双域重建器，从首轮起在模型重建图上训练分割器。可选 [输入/调度消融配置](configs/v4_32_schedule_ablation.yaml) 与 [联合组合配置](configs/v4_32_joint_combined.yaml) 用于分辨各项改动的作用。输出包含原始概率 NIfTI、32 views 肝脏门控前后掩膜和按病灶诊断 CSV。

最终报告建议使用 [configs/v4_dualdomain_native512.yaml](configs/v4_dualdomain_native512.yaml)，原因是 256 网格的指标以原 512 CT 为参照时存在上限。本包**未包含**在真实 20 例上训练的 V4 权重，真实提升须重训后以测试集或五折结果为准。

---

以下为 V3 说明，仍然适用（V4 配置沿用相同的审计、划分、缓存、评估和五折流程）。


本包依据你上传的旧代码、当前 V2 的三份测试 CSV 和新的标注范围要求修改。**交付的是可运行代码与配置，不包含在真实 20 位患者上重新训练的权重，也不预先声称 SSIM/Dice 已提高。** 修正后的实验须从原始 DICOM 重做审计、投影缓存、训练和折外测试。

配套说明：[结果诊断与改动依据](docs/结果诊断与改动依据.md)、[所有配置参数及原因](docs/参数说明.md)、[各 Python 文件功能](docs/Python文件索引.md)、[验证记录](docs/验证记录.md)。

## 改动与数据定义

- 专家标注读取会审计独立 tumor 及 livertumor 系列目录，但训练标签限定为与专家肝脏 mask 相交的部分；肝内占比低于 `min_liver_tumor_fraction` 的目录整体排除。patient 7 的 `tumor` 目录属于肝外病灶，当前任务中该患者肿瘤真值为 0 ml。官方数据说明见 [IRCAD](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/)。
- 审计输出每人的 CT 层数、全部 mask 目录、选中目录各自非零体素、肿瘤位于专家肝 mask 内/外的体素数。预期阳性患者缺少匹配目录或掩膜为空时失败，防止静默变成负例。
- 保留原 V2 plain U-Net，同时提供可选 RCAB 残差通道注意力 + PixelShuffle 重建分支、真值约束二阶边缘误差、肿瘤边缘优先裁块，以及分割边界/肝外困难阴性监督。推理不读取专家 mask；肿瘤后处理按当前肝内目标限制在预测肝脏邻域。
- 原患者 train/val/test 名单可显式迁移到**新标签审计**，五折 fold 0 完整保留它们；另外四折让余下患者各担任一次外层测试。先在验证集固定模型与参数，再汇总 20 人的折外结果。
- 正式有真值评估对每人、每个 32/64/128 视角、每层保存一张六联图和六张同名序号的独立 PNG。第三幅现为专家 mask，第五幅为纯模型重建。

原始 PATIENT_DICOM 是已重建的患者 CT，MASKS_DICOM 是专家标注；本包没有扫描仪原始投影。预处理把 CT 转成模拟平行束投影，1024 角作为模拟全视角参考，从中嵌套抽取 32/64/128 角并作 FBP。训练目标是原 CT；同时分别报告以原 CT、模拟 full_FBP 为参照的 SSIM、PSNR、MAE、RMSE。图像可因稀疏角度产生条纹，但真实改进幅度只能由新权重验证。

## 目录、安装和配置

推荐 Python 3.11/3.12。先根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装与本机 GPU/驱动匹配的 torch，然后在**本包根目录**运行：

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.cli --help
python -m pytest -q
~~~

如 PowerShell 不允许激活环境，把命令中的 python 改为 .venv\Scripts\python.exe。配置参数集中于 YAML 与 src/config.py 开头；路径按**当前命令目录**解析。推荐的配置：

| YAML | 用途 | 代价 |
|---|---|---|
| configs/enhanced_native512.yaml | 首选质量实验：512 投影缓存、256 patch、RCAB/PixelShuffle，32 视角采样权重 0.6 | 投影与全层出图较慢，GPU 显存要求较高 |
| configs/enhanced_256_low_memory.yaml | 低显存增强实验：256 整图、base16、batch1 | 小病灶及边缘受到 256 缩放限制 |
| configs/default.yaml | 与旧训练超参数尽量一致的 plain 对照，但使用**新标签定义** | 旧权重和旧缓存不能用于新标签实验 |

其他 YAML 保留消融/无噪声或低显存参考，均须先检查 data_root、cache_dir 和 run_dir。未获知你的准确显卡/显存，512 增强配置把 batch 设为 1，并在 CUDA 可用时开启 AMP；如果仍不足，优先降低 base_channels 到 16，作为**另一独立实验**记录。所有参数作用及推荐理由见 [参数说明](docs/参数说明.md)。

数据根目录应含 3Dircadb1.1 至 3Dircadb1.20，每人含解压的 PATIENT_DICOM/MASKS_DICOM，或对应内层 zip。不要将 LABELLED_DICOM 自动当作本任务的专家 mask。官方 20 人分别有 129、172、200、91、139、135、151、124、111、122、132、260、122、113、125、155、119、74、124、225 层；运行时仍以本地 DICOM 和几何配对审计为准。

## 从原划分迁移：必须用新缓存与新权重

先复制原实验的真实 splits.json 路径；**不复制原 audit.json、FBP 数组、mask 缓存或 checkpoint 到新实验**。修改首选 YAML 中的 data_root 为实际数据根目录，cache_dir/run_dir 指向尚未用于旧实验的新路径；split_path 保持 null。以下命令中的 E:/旧实验/cache/size256/splits.json 只作示例，替换成你机器上原文件的真实路径：

~~~powershell
python -m src.cli --config configs/enhanced_native512.yaml audit
python -m src.cli --config configs/enhanced_native512.yaml split --base-split "E:/旧实验/cache/size256/splits.json"
python -m src.cli --config configs/enhanced_native512.yaml prepare
python -m src.cli --config configs/enhanced_native512.yaml cache-audit --with-dicom
~~~

审计时先看新 cache_dir/audit.csv：patient 7 的 `tumor_mask_names` 应有 `tumor`，但肝内目标的 `tumor_voxels` 应为 0；病例 5 的 leftsurretumor/rightsurretumor 不进入目标标签。splits.json 会保留旧 train/val/test 患者 ID 和列表顺序，但用新 audit 指纹、新肿瘤负担重签。若旧 splits.json 遗失，CSV 只告诉我们测试有 6/7/8，**无法恢复原 train/val**；此时只能重新 split，并明确不再称“保留原划分”。已有审计/划分文件若与当前内容不一致会拒绝覆盖，请新建 cache_dir，保留旧实验。

512 分辨率的 1024 角投影要对 2823 个切片模拟，预处理时间可能很长；prepare 可按患者断点续跑。改图像尺寸、标签规则、投影参数后必须新缓存；改变网络/损失后必须新 run 和新权重。

## 单划分训练、验证和测试

~~~powershell
python -m src.cli --config configs/enhanced_native512.yaml train
python -m src.cli --config configs/enhanced_native512.yaml evaluate --checkpoint runs/joint_enhanced512/best.pt --split test
~~~

训练先用本折 train 做重建及 clean 图分割预热，再逐步让分割使用重建图并联合优化。best.pt 只由 val 指标和逐视角重建门槛选出；若门槛未过，程序只保留 best_candidate.pt 供诊断，不能把它默认为已合格。恢复同协议中断训练时，用该 run 的 resolved_config.yaml 与 last.pt；**不要**用旧 V2 的错误标签权重续训新任务。

## 患者级五折交叉验证

想直接做完整五折，可跳过上面的单划分 train，避免重复训练 fold 0：

~~~powershell
python -m src.cli --config configs/enhanced_native512.yaml cv-init --output runs/cv_enhanced512
python -m src.cli --config configs/enhanced_native512.yaml cv-train --folds runs/cv_enhanced512/folds.json
python -m src.cli --config configs/enhanced_native512.yaml cv-evaluate --folds runs/cv_enhanced512/folds.json
python -m src.cli --config configs/enhanced_native512.yaml cv-aggregate --folds runs/cv_enhanced512/folds.json
~~~

原测试集若为 3 人，保留原三组名单意味着五折测试大小为 **3/4/4/4/5**，无法同时严格做到每折 4 人。每人恰好进入一次外层 test；其他折的 train/val 不共享本折测试患者，模型每折从头初始化。val 选择权重，test 只在训练完成后使用；汇总先合并 20 名折外患者，再按患者等权求平均，不直接平均大小不等的五个折均值。按肝内目标，patient 7 也属于 tumor 阴性病例；不能保证每个外层测试折都有阴性例，程序会提示并在合并的折外结果上报告阴性假阳性。可用 cv-train/cv-evaluate 的 --only 0 1 等参数分批运行。原测试患者已参与本次问题诊断，因此 fold 0 不再是全新盲测；五折属于内部验证，医学应用仍需独立外部验证。

## 指标与图像输出

每个测试 run 的 evaluation_test 下保存 reconstruction_per_patient.csv、segmentation_per_patient.csv、summary.json、provenance.json；启用图像导出时另存 reconstruction_regions_per_patient.csv，图像位于 volumes/3Dircadb1.N/32、64、128 目录。重建 CSV 包含 original_CT/full_FBP 两种参照，FBP/Joint 两种方法；分割按完整患者 3D 体积算 liver/tumor Dice，并报告肿瘤假阳性体积。SSIM/PSNR 用同一固定窗，MAE/RMSE 用未截窗 HU；小病灶也要看逐患者与肿瘤区域指标，不能只看全图均值。

~~~text
evaluation_test/volumes/3Dircadb1.7/32/
  restored_hu.nii.gz  fbp_hu.nii.gz  liver.nii.gz  tumor.nii.gz
  comparison.png                  代表层六联图，便于快速浏览
  slices/slice_0000/
    comparison.png
    01_ground_truth.png           原 CT
    02_sparse_fbp.png             当前视角 FBP
    03_expert_segmentation.png    专家 liver/tumor 标签
    04_reconstruction_segmentation_overlay.png  模型重建 + 预测分割
    05_model_reconstruction.png   模型重建纯图
    06_predicted_segmentation.png 单独预测标签
~~~

每层全部导出可能产生数万张 PNG 和大量磁盘占用；需要先快速验指标时，对 evaluate 或 cv-evaluate 加 --no-export。旧 save_slice_every 字段仅为兼容旧 YAML 保留，正式导出不再抽样层。无真值部署接口 infer 仅接收**已由相同协议做 FBP 的 HU NIfTI**，不能自动从原始未知几何投影生成专家真值图；其运行方式：

~~~powershell
python -m src.cli --config runs/joint_enhanced512/resolved_config.yaml infer --checkpoint runs/joint_enhanced512/best.pt --input "E:/inputs/sparse32_fbp_hu.nii.gz" --views 32 --output runs/predict32
~~~

## 对照与完成标准

先固定投影、患者划分、参照和指标，依次比较 FBP、仅修正标签的 plain 模型、增强模型；旧包约 0.86 只有同协议权重/输入/指标可复现时才列入并排表。验证集选定配置后才统一做五折测试，报告各视角患者级 SSIM/PSNR/MAE/RMSE、肝/瘤 Dice、肝外假阳性与 patient 7/8 的具体结果。代码测试和合成 DICOM smoke 只证明软件流程，不证明真实患者达到临床可用质量。
