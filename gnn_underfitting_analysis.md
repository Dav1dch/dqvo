# GNN BA 欠拟合分析记录（2026-04-11）

## 现象
- 训练阶段出现“连训练集都压不下去”，表现为明显欠拟合。
- 已修复一次训练错误（`huber_loss` 返回向量导致 `torch.isfinite(loss)` 报错），但整体拟合能力仍不足。

## 高优先级可疑原因（按优先级）
1. 模型输出是“VO 基线 + 小残差”，偏离能力受限
- c2c edge_attr 直接使用 VO 相对位姿作为基线。
- 最终输出是 `camera_relative = c2c_edge_attr + c2c_delta`。
- 且 `c2c_pose_out_proj` 初始化非常小（gain=0.01），前期更新幅度小。

2. 点分支几乎不参与迭代更新，重投影监督信号不够强
- 当前结构主要更新 camera hidden 与 c2c edge hidden。
- point hidden 未在迭代中与 camera 双向共同更新，point 输出学习能力偏弱。

3. 输入尺度不一致（camera node 与 c2c edge 的统计分布不同）
- camera node 输入使用 denormalized 绝对位姿（量纲较大）。
- c2c edge_attr 使用 normalized 相对位姿。
- 节点与边尺度混杂，优化器收敛效率变差。

4. 有效训练样本可能偏少
- 训练中存在多处 `continue` 跳过样本：track 数量不足、三角化失败、深度无效、loss_clip 超阈值等。
- 实际参与反传的样本比例可能远低于表面 batch 数。

5. 损失项之间可能冲突
- reproj 与 pose 当前同权重配置，不一定适合初期学习。
- 在几何噪声较大时，联合优化容易互相牵制。

## 快速验证顺序（建议）
1. 单窗口/小样本过拟合测试
- 固定 1-4 个 window，观察能否快速把训练 loss 降下来。
- 若小样本都不能过拟合，优先看模型表达/尺度问题；
  若能过拟合，再看数据筛样与泛化。

2. 统计“有效样本率”与跳过原因
- 每个 epoch 输出：总样本、有效样本、各类跳过计数（tracks 不足、triangulation 失败、invalid depth、loss clip）。

3. 暂时关闭残差基线（对照实验）
- 将 c2c 输出从 `base + delta` 改为 `delta`，测试是否更容易过拟合。

4. 统一输入尺度（对照实验）
- 将 camera node 与 c2c edge_attr 统一到同一规范（建议先都 normalized）。

5. 分阶段训练损失
- 先 `reproj_weight=0` 仅训练 pose，验证可过拟合；
- 再逐步提高 reproj 权重。

## 当前状态
- 本次仅记录分析，不改动模型逻辑。
- 后续按上述顺序做最小对照实验，可最快定位瓶颈来源。
