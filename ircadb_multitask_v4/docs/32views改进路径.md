# 32 views 肝内肿瘤分割：两版可运行方案

目标只包括专家肝脏 mask 内的肿瘤。patient 7 的肝外病灶不是肿瘤真值；评估它时关注假阳性。现有包内 `evaluation_test` 显示，32 views 下 patient 6 的肿瘤 Dice 为 0.407、病灶召回为 4/15，patient 8 的 Dice 为 0.040、召回为 1/3；patient 7 为阴性，肿瘤假阳性 4.42 mL。这些数字来自旧预测，不能当作以下方案的效果。

本复制包保存了 CSV、PNG 和验证集 `postprocess.json`（阈值 0.30、最小连通域 0.2 mL），没有 `best.pt`、缓存数组或 DICOM。下列命令须在有原权重和缓存的训练机器上运行。先核对 YAML 的 `cache_dir`、`data_root`、`run_dir` 与 checkpoint 路径。

## A. 不重训：32 views 概率校准

在仓库根目录运行。第一步保存原始肝/瘤概率和 32 views 门控前后掩膜；`--no-export` 避免再次生成所有逐层 PNG。

```powershell
python -m src.cli --config configs/v4_dualdomain_256.yaml evaluate --checkpoint runs/v4_dualdomain256/best.pt --split val --no-export --save-probabilities
python -m src.cli --config configs/v4_dualdomain_256.yaml tune-32-postprocess --checkpoint runs/v4_dualdomain256/best.pt
python -m src.cli --config configs/v4_dualdomain_256.yaml evaluate --checkpoint runs/v4_dualdomain256/best.pt --split val --no-export --save-probabilities
python -m src.cli --config configs/v4_dualdomain_256.yaml evaluate --checkpoint runs/v4_dualdomain256/best.pt --split test --save-probabilities
```

验证集搜索肿瘤阈值 `0.10/0.15/0.20/0.25/0.30`、最小连通域 `0/0.05/0.2 mL`，以及同层预测肝脏外扩 `5/10 mm` 或取消门控。以一对一 26 邻域病灶召回优先；候选还须不降低阳性患者平均 Dice，且阴性患者平均假阳性体积不超过配置的 `max_negative_fp_ml: 5`。没有合格且优于原方案的候选，或该折验证集没有阴性患者而无法测量假阳性约束时，保留原参数。参数写入同一 run 的 `postprocess.json` 中 `per_view.32`；原顶层参数继续用于 64/128 views。测试集只读取冻结参数，不参与选择。

输出重点：`evaluation_val/volumes/<患者>/32/{liver,tumor}_probability.nii.gz` 是映射回原生网格后、**未阈值化**的概率；`tumor_before_liver_gate.nii.gz` 和 `tumor_after_liver_gate.nii.gz` 用于看门控删除了什么；`postprocess_lesion_diagnostics_32.csv` 按真值连通域列出最大概率、95 分位概率及门控前后命中体素。冻结后在 `evaluation_test` 的同名文件检查 patient 6 是否主要被门控删除，patient 8 的漏检病灶是否在 0.10–0.30 有响应，并核对 patient 7 的假阳性。最终测试命令默认重新导出 PNG，使图像与新参数一致。真值只用于**诊断与评估**，推理后处理不读取真值。若病灶内原始概率也没有响应，降低阈值无法可靠补回。

`evaluate` 有逐患者 CSV 与 `summary.json`，同时保存 liver Dice、阳性 tumor Dice、病灶召回、阴性假阳性体积及重建指标。新病灶召回用一对一匹配；一个连在一起的预测区域不能同时算检出两个真值病灶，因此旧 CSV 病灶指标应使用当前代码重新评估后再比较。

## B. 重训：固定重建器，微调分割器

主实验配置是 [`configs/v4_32_seg_finetune.yaml`](../configs/v4_32_seg_finetune.yaml)。它加载已有 V4 `best.pt`，检查模型结构、缓存准备签名、患者划分及审计指纹；新 run 仅训练分割分支，双域重建器保持固定。训练输入从首轮起即为模型重建图，分割损失始终保持 1.0；32/64/128 views 抽样概率为 `0.7/0.15/0.15`。先按患者等概率，再在阳性患者内按病灶等概率选择病灶及其所在层，减轻大病灶占据多数阳性切片的问题。保留 Focal-Tversky、边界损失、深监督。验证集以 32 views 病灶召回为主，阳性 Dice、肝脏 Dice 和重建得分参与评分，阴性患者假阳性体积及逐视角重建退化作为选模约束。

```powershell
python -m src.cli --config configs/v4_32_seg_finetune.yaml train
python -m src.cli --config configs/v4_32_seg_finetune.yaml evaluate --checkpoint runs/v4_32_seg_finetune/best.pt --split test --save-probabilities
```

训练结束会在**验证集**自动完成原全视角参数搜索，再搜索 32 views 参数。若阴性假阳性或重建约束导致没有合格 `best.pt`，查看 `selection_report.json` 和 `best_candidate.pt`；候选权重只供诊断，不应当作满足约束的最终模型。

隔离输入/调度改动的可选对照使用 [`configs/v4_32_schedule_ablation.yaml`](../configs/v4_32_schedule_ablation.yaml)：原抽样比例、原联合训练结构，clean 预训练最后 10 个 epoch 逐渐换成重建图，阶段切换时分割损失权重保持 1.0。再用 [`configs/v4_32_joint_combined.yaml`](../configs/v4_32_joint_combined.yaml) 比较输入/调度修复与病灶均衡、32 views 加权的组合效果。两者都是独立 run，从头训练。若 256 缓存下小病灶持续漏检，应从原生 512 缓存训练对应对照；把 256 图像放大到 512 不会恢复病灶信息。

## 五折折外比较

每个折的微调来源必须是**同折**原 V4 权重，绝不能用包含该折测试患者的单个 `best.pt` 初始化。已有基线五折权重时，可先用同折验证集校准 32 views，再做折外评估：

```powershell
python -m src.cli --config configs/v4_dualdomain_256.yaml cv-tune-32-postprocess --folds runs/cv_v4/folds.json
python -m src.cli --config configs/v4_dualdomain_256.yaml cv-evaluate --folds runs/cv_v4/folds.json --no-export
python -m src.cli --config configs/v4_dualdomain_256.yaml cv-aggregate --folds runs/cv_v4/folds.json --output runs/cv_v4/post32_aggregate
```

五折微调另建协议目录。将微调 YAML 的 `seg_finetune_checkpoint` 改为类似 `runs/cv_v4/fold_{fold}/run/best.pt` 的模式、`run_dir` 改为新目录，再运行 `cv-init`、`cv-train`、`cv-evaluate`、`cv-aggregate`。`cv-init` 会为每折写具体来源路径，加载时仍逐折核查 split/audit 指纹；不一致就拒绝训练。每折的 32 views 参数在该折验证集上自动选择。比较两版的每名病人漏检与误报、肝脏/阳性肿瘤 Dice、阴性假阳性及重建指标。已用旧测试结果作设计依据，五折仍属内部验证。

## 运行后统一打包

完成两版验证集校准与真实测试评估后，可在仓库根目录运行：

```powershell
python -m scripts.package_delivery --output ../ircadb_32views_complete_results.zip --last-checkpoint "C:/Users/gusta/OneDrive/Desktop/xinsong/last.pt"
```

脚本会先核对两版 `best.pt`、校准文件的权重哈希、验证/测试 provenance、逐患者逐视角 CSV、概率 NIfTI、门控诊断和测试 PNG；缺失就拒绝生成压缩包。压缩包包含代码、配置、文档、已运行结果和权重，不包含原始 DICOM 或投影缓存；生成后执行 ZIP CRC 校验并报告 SHA-256。

## 审阅后的修正（2026-09-27）

1. **训练期选模阈值与部署阈值不一致。** 微调、联合配置的 `tumor_threshold: null`，所以每轮验证在 0.5 阈值下计算 32 views 病灶召回；部署时 32 views 实际使用 `tune_32_thresholds`（0.10–0.30）中选出的阈值。两者不一致时，选出的权重未必在低阈值下召回最好。现在开启 `selection_32_lesion_priority` 后，32 views 另外计算 `tumor_lesion_recall_tuned_range`，即候选阈值区间上的平均一对一召回，选模优先使用它。旧阈值召回仍在 CSV 中保留。
2. **联合训练时选模几乎不看重建。** 原先的病灶优先评分中重建只占 0.025，而 `v4_32_schedule_ablation`、`v4_32_joint_combined` 会同时训练重建器，best.pt 可能在重建明显变差时仍被选中，只要不低于 FBP 即可。现在评分改为 `w·重建 + (1-w)·(0.75 召回 + 0.20 Dice + 0.05 肝 Dice)`，`w = selection_reconstruction_weight`（配置中为 0.5）。固定重建器微调时重建分数是常数，排序与原来相同。
3. **验证集没有阴性患者时，32 views 校准不起作用。** 20 例中只有 5/11/14/20 为阴性，3 人验证集经常一个阴性都没有，此时 `tune-32-postprocess` 一定保留原参数。新增 `fp_guard_fallback: all_patients`（三个 32 views 配置已开启）：没有阴性患者时，改用全部验证患者“真值以外的预测体积”作为假阳性约束，要求不比原参数多出 `max_fp_increase_ml`（5 mL）；同时保留阳性 Dice 不下降的要求。有阴性患者时仍按原规则执行。`postprocess.json` 的 `selection_32.fp_guard` 会记录实际使用的约束。代码默认值仍为 `keep_baseline`，旧配置行为不变。
4. 如果一个 epoch 的所有 batch 都因梯度非有限被跳过，现在会明确报错为“训练发散”，不再出现除零错误。

## 仍值得做的方向（需要真实结果确认）

- **分辨率是小病灶召回的主要瓶颈。** patient 8 的 3 个病灶合计约 10 mL，在 256 网格上每个病灶只剩很少的像素。阈值和门控只能找回“有响应但被阈值删掉”的病灶。运行 `--save-probabilities` 后，如果 `postprocess_lesion_diagnostics_32.csv` 中漏检病灶的 `tumor_probability_max` 本身就低于 0.1，应改用原生 512 双域配置；更进一步可以增加一个在预测肝脏包围框内、原生分辨率上运行的第二阶段肿瘤分割器。
- **固定重建器微调可以提速。** 重建器不更新时，每步仍要完整计算一次正弦图网络、FBP 和图像网络。可以对训练集各视角的重建结果预计算一次并缓存。代价是几何增强要作用在缓存的重建图上，与在线重建不完全等价，建议作为独立实验。
- **验证集只有 3 人，基于召回的选模噪声很大。** 最终结论应以五折折外结果为准，每折的 32 views 参数只在本折验证集上选择。
