# 各 Python 文件功能

常规实验只需使用 `python -m src.cli --config ... 子命令`；不需要逐个执行模块。

| 文件 | 功能 |
|---|---|
| [src/__init__.py](../src/__init__.py) | Python包标识。 |
| [src/cache_quality.py](../src/cache_quality.py) | 核对缓存HU、尺寸与签名；从原DICOM的存储值域识别padding并写独立有效区sidecar。 |
| [src/cli.py](../src/cli.py) | 统一命令入口，含audit、原划分迁移、训练/评估、cache-audit、四个CV命令，以及 V4 的 tune-postprocess 与 infer --sinogram/--reproject-fbp。 |
| [src/common.py](../src/common.py) | 随机种子、设备、JSON/CSV、摘要指纹公共工具。 |
| [src/config.py](../src/config.py) | Config默认参数、YAML加载/校验、缓存签名和split路径；V4 双域/分割/后处理/EMA 参数及校验（默认值保持V3行为）。 |
| [src/ct_ops.py](../src/ct_ops.py) | **V4 新增**：与 skimage radon/iradon 逐像素一致的可微平行束投影与FBP、角度插值与周期填充、正弦图/图像的精确镜像与90°旋转。 |
| [src/crossval.py](../src/crossval.py) | 冻结5折、保留fold0、独立训练/测试、检查来源并汇总折外患者。 |
| [src/dataset.py](../src/dataset.py) | 按患者采样、物理邻层、肿瘤层和可选肿瘤边界优先采样、同步patch/几何增强；V4 双域样本（未测角度置零的稀疏正弦图 + 稠密正弦图真值 + 精确几何增强）。 |
| [src/dicom_io.py](../src/dicom_io.py) | 读取文件夹或内层zip；HU标定、物理排序、CT/mask配对、tumor目录及各类mask体素审计。 |
| [src/evaluate.py](../src/evaluate.py) | 整患者推理回原生网格、重建/分割/区域评估、验证评分/门槛、NIfTI及全层六联图/独立PNG导出；V4 验证集后处理调参（postprocess.json）与推理专用参数豁免。 |
| [src/infer.py](../src/infer.py) | 无需GT的稀疏FBP HU NIfTI推理；检查三维、方形、等距层内像素与LPS轴方向；V4 双域模型接收稀疏正弦图，并套用验证集后处理参数。 |
| [src/losses.py](../src/losses.py) | 器官加权HU L1/MSE/梯度/可选二阶边缘、SSIM、FBP退化惩罚、双通道BCE/Dice及可选肝外困难阴性/边缘监督；V4 正弦图/双域中间图/频域/投影一致性损失，肿瘤逐样本Focal-Tversky、层级一致性、深监督。 |
| [src/metrics.py](../src/metrics.py) | 固定窗SSIM/PSNR、未截窗HU误差、体积Dice、区域指标、患者配对bootstrap；V4 后处理前后Dice、HD95/ASSD、病灶级检出率/精确率。 |
| [src/model.py](../src/model.py) | 级联2.5D重建器与双通道分割器；可选RCAB/PixelShuffle，视角条件及可控分割反馈；V4 双域重建器（正弦图补全+硬数据一致性+可微FBP+图像精修）、膨胀上下文瓶颈、残差注意力分割网（注意力门、深监督、肝条件肿瘤头）。 |
| [src/postprocess.py](../src/postprocess.py) | **V4 新增**：3D 后处理（肝最大连通域+填洞、肿瘤限制在预测肝±margin、去小碎片）与验证集阈值/最小体积网格搜索。 |
| [src/official.py](../src/official.py) | 官方20例切片数、逐病例目标mask目录清单、肝内与本任务广义tumor阴性区别及参考属性。 |
| [src/predict.py](../src/predict.py) | 共享整图或重叠patch推理；TTA逆变换、Hann融合，中心恢复层与概率只计算一次；V4 双域整图重建 + 分割翻转TTA。 |
| [src/prepare.py](../src/prepare.py) | 逐患者生成原生HU/mask及训练尺寸的target、密集FBP、三档稀疏FBP缓存；核对缓存签名；V4 保存/就地补充稠密角度正弦图，并以 iradon 复核与缓存FBP一致。 |
| [src/projection.py](../src/projection.py) | HU→衰减系数、平行束Radon、嵌套稀疏角度、可选计数噪声和FBP；V4 稠密角度正弦图（水等效像素单位）。 |
| [src/split.py](../src/split.py) | 按切片量、肿瘤负担、性别冻结患者划分；允许保留旧患者名单并在新标签审计后重签。 |
| [src/train.py](../src/train.py) | clean分阶段预训练/联合训练、AMP、裁剪、验证选模、独立checkpoint、严格恢复；V4 EMA、学习率预热+余弦、选模起始轮、训练后自动调后处理。 |
| [src/visualize.py](../src/visualize.py) | 生成2×3六联图及六张独立PNG；第三幅专家mask、第五幅纯重建，所有CT统一窗。 |
| [tests/__init__.py](../tests/__init__.py) | 测试包标识。 |
| [tests/phantom.py](../tests/phantom.py) | 生成无真实患者信息的合成CT/mask DICOM。 |
| [tests/smoke_crossval.py](../tests/smoke_crossval.py) | 20位合成患者真实训练5次、测试5折和OOF汇总；2轮仅验证流程。 |
| [tests/benchmark_dual_vs_image.py](../tests/benchmark_dual_vs_image.py) | **V4 新增**：合成体模消融，同等步数比较 V3 图像域与 V4 双域去条纹效果（仅验证方法方向）。 |
| [tests/smoke_pipeline.py](../tests/smoke_pipeline.py) | 单划分完整审计→投影→训练→恢复→测试→NIfTI推理自检；`--mode dual` 覆盖 V4 双域、旧缓存就地补正弦图、EMA恢复、后处理调参与正弦图推理。 |
| [tests/test_cache_quality.py](../tests/test_cache_quality.py) | padding存储域/闭区间、HU/几何/hash核验、sidecar来源与原缓存不变。 |
| [tests/test_core.py](../tests/test_core.py) | DICOM几何、掩码、患者互斥、投影、指标、物理层索引、NIfTI核心测试。 |
| [tests/test_crossval.py](../tests/test_crossval.py) | fold0保留、OOF覆盖、分层、独立目录/权重、配置/结果来源和非法改写拒绝。 |
| [tests/test_patch_integration.py](../tests/test_patch_integration.py) | 同步裁剪增强、采样复现、512/256patch以及evaluate与无GT infer一致性。 |
| [tests/test_revision.py](../tests/test_revision.py) | 损失尺度/梯度、负监督、滑窗/TTA和验证选模门槛回归测试。 |
| [tests/test_model_optimization.py](../tests/test_model_optimization.py) | 新重建器前/反向、边界/困难阴性损失、肿瘤边界采样回归测试。 |
| [tests/test_visualization.py](../tests/test_visualization.py) | 六图顺序、专家与预测mask来源、全层独立PNG、输入合法性。 |
| [tests/test_v4_dual_domain.py](../tests/test_v4_dual_domain.py) | **V4 新增**：投影/FBP与skimage一致、正弦图精确增强、双域模型初值与数据一致性、数据集等变性、Tversky/层级损失、后处理与调参、V3权重键兼容。 |
