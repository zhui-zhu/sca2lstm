# SCA2LSTM 流域微调使用指南

## 概述

`fine_train.py` 是一个专门用于在预训练SCA2LSTM模型基础上对特定流域进行微调的脚本。通过微调，可以使模型更好地适应特定流域的水文特征，提高预测精度。

## 主要特性

- **智能层冻结**: 默认冻结嵌入层和LSTM1层，只微调LSTM2层和预测头
- **渐进式学习**: 使用较小的学习率，避免破坏预训练权重
- **早停机制**: 防止过拟合，确保泛化能力
- **详细日志**: 提供完整的训练过程和评估指标
- **可视化支持**: 自动生成训练曲线和性能图表

## 使用方法

### 基本用法

```bash
# 对流域ID为32006的流域进行微调
python fine_train.py --target_basin 32006 --pretrained_model ./model_output/sca2lstm.pth
```

### 高级用法

```bash
# 自定义微调参数
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --patience 10 \
    --fine_tune_ratio 0.85
```

### 解冻特定层

```bash
# 解冻嵌入层和LSTM1层进行更全面的微调
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --unfreeze_embedding \
    --unfreeze_lstm1 \
    --learning_rate 1e-5
```

## 参数说明

### 必需参数

- `--target_basin`: 目标流域ID (整数)
- `--pretrained_model`: 预训练模型文件路径

### 微调参数

- `--epochs`: 微调轮次，默认20
- `--batch_size`: 批次大小，默认16
- `--learning_rate`: 学习率，默认1e-4 (0.0001)
- `--weight_decay`: 权重衰减，默认1e-5
- `--patience`: 早停耐心值，默认8
- `--fine_tune_ratio`: 微调数据比例，默认0.8 (80%用于训练，20%用于验证)

### 层解冻参数

- `--unfreeze_embedding`: 解冻流域嵌入层
- `--unfreeze_lstm1`: 解冻LSTM1层
- `--unfreeze_weight_head`: 解冻权重头

### 其他参数

- `--no_cuda`: 禁用CUDA，强制使用CPU
- `--seed`: 随机种子，默认42

## 微调策略建议

### 1. 保守微调（推荐）

适用于数据量较小或希望保持预训练特征的场合：

```bash
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --epochs 15 \
    --learning_rate 1e-4 \
    --fine_tune_ratio 0.8
```

特点：
- 只微调LSTM2层和预测头
- 使用较小学习率
- 训练时间短
- 泛化能力强

### 2. 深度微调

适用于数据量充足且需要充分适应新流域的场合：

```bash
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --epochs 30 \
    --learning_rate 5e-5 \
    --unfreeze_embedding \
    --unfreeze_lstm1 \
    --fine_tune_ratio 0.9
```

特点：
- 解冻更多层
- 使用更小学习率
- 训练时间长
- 适应性更强

### 3. 快速微调

适用于快速验证或时间紧迫的场合：

```bash
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --patience 5
```

特点：
- 训练轮次少
- 批次大
- 学习率较大
- 快速见效

## 输出说明

微调完成后，脚本会生成以下输出：

1. **微调模型文件**: `best_model_basin_{basin_id}.pth`
2. **训练曲线图**: 显示训练和验证损失的变化趋势
3. **详细日志**: 包含每轮的训练和验证指标
4. **配置信息**: 保存微调时的参数设置

## 模型使用

微调完成后，可以使用生成的模型进行预测：

```python
import torch
from sca2lstm import SCA2LSTM, load_config

# 加载配置
config = load_config()

# 初始化模型
model = SCA2LSTM(config)

# 加载微调后的模型
checkpoint = torch.load('./model_output/fine_tune/basin_32006/20241123_143022/best_model_basin_32006.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 现在可以使用这个模型对流域32006进行预测
```

## 注意事项

1. **数据要求**: 确保目标流域有足够的数据量（建议至少5000条有效记录）
2. **学习率选择**: 微调时使用较小的学习率，避免破坏预训练权重
3. **早停设置**: 合理设置耐心值，防止过拟合
4. **层解冻**: 谨慎解冻深层，可能会导致灾难性遗忘
5. **硬件要求**: 建议使用GPU加速训练过程

## 故障排除

### 常见问题

1. **内存不足**: 减小批次大小或使用更小的fine_tune_ratio
2. **训练不稳定**: 降低学习率或增加层冻结
3. **过拟合**: 增加早停耐心值或减少训练轮次
4. **收敛慢**: 适当增加学习率或减少冻结层

### 性能优化

1. **数据预处理**: 确保数据质量，处理缺失值和异常值
2. **批次大小**: 根据GPU内存选择合适的批次大小
3. **学习率调度**: 使用学习率衰减策略
4. **正则化**: 合理设置权重衰减参数

## 示例流程

```bash
# 1. 首先训练基础模型
python sca2lstm.py --epochs 60 --parallel

# 2. 对特定流域进行微调
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --epochs 25 \
    --learning_rate 8e-5 \
    --fine_tune_ratio 0.85
    
# 3.深度微调策略
python fine_train.py \
    --target_basin 32006 \
    --pretrained_model ./model_output/sca2lstm.pth \
    --unfreeze_embedding \
    --unfreeze_lstm1 \
    --learning_rate 5e-5 \
    --epochs 30

# 4. 使用微调模型进行预测
python predict.py --model ./model_output/fine_tune/basin_32006/*/best_model_basin_32006.pth --basin 32006
```


通过合理的微调策略，可以显著提升模型在特定流域的预测性能。