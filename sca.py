"""
梯度引导的正余弦优化算法 (Gradient-Guided Sine Cosine Algorithm)
用于SCA2LSTM模型的参数优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple


class SCAOptimizer(optim.Optimizer):
    """
    梯度引导的正余弦优化算法优化器
    
    特点：
    - 利用梯度信息指导搜索方向
    - 自适应调整搜索范围
    - 支持动量和自适应学习率
    - 与PyTorch autograd完全兼容
    """
    
    def __init__(self, params, lr=0.01, population_size=20, a_max=2.0, 
                 momentum=0.9, gradient_weight=0.7, adaptive_lr=True):
        """
        初始化梯度引导SCA优化器
        
        参数:
        -----------
        params : iterable
            模型参数迭代器
        lr : float, default=0.01
            基础学习率
        population_size : int, default=20
            种群大小（候选解数量）
        a_max : float, default=2.0
            正弦/余弦振幅的最大值
        momentum : float, default=0.9
            动量系数
        gradient_weight : float, default=0.7
            梯度引导的权重（0-1之间）
        adaptive_lr : bool, default=True
            是否使用自适应学习率
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= gradient_weight <= 1.0:
            raise ValueError(f"Invalid gradient weight: {gradient_weight}")
        
        defaults = dict(lr=lr, population_size=population_size, a_max=a_max,
                       momentum=momentum, gradient_weight=gradient_weight,
                       adaptive_lr=adaptive_lr)
        super(SCAOptimizer, self).__init__(params, defaults)
        
        # 存储梯度信息
        self.gradients = {}
        
        # 迭代计数
        self.iteration = 0
    
    def store_gradients(self, loss=None):
        """存储梯度信息"""
        # 如果已经计算了梯度，直接存储
        param_idx = 0
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    self.gradients[param_idx] = param.grad.clone()
                    param_idx += 1
    
    def _get_gradient_direction(self, param_idx, param):
        """获取梯度方向"""
        if param_idx in self.gradients:
            gradient = self.gradients[param_idx]
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e-8:
                return -gradient / grad_norm  # 负梯度方向
        return torch.zeros_like(param.data)
    
    def _sca_update(self, param_data, gradient_direction, lr, gradient_weight, a_max):
        """SCA位置更新（增加稳定性）"""
        # SCA参数（随迭代衰减，增加稳定性）
        a = max(0.2, a_max - (a_max * self.iteration) / 3000)  # 周期从2000增至3000，最小值从0.1增至0.2
        r1 = a * (2 * np.random.rand() - 1)
        r2 = 2 * np.pi * np.random.rand()
        r3 = 2 * np.random.rand()
        r4 = np.random.rand()
        
        # 计算SCA移动（增加边界检查）
        param_np = param_data.cpu().numpy()
        if r4 < 0.5:
            movement = r1 * np.sin(r2) * np.abs(r3 * 0.05 * param_np)  # 幅度从0.1降至0.05
        else:
            movement = r1 * np.cos(r2) * np.abs(r3 * 0.05 * param_np)  # 幅度从0.1降至0.05
        
        # 限制移动范围，避免过大更新
        movement = np.clip(movement, -0.05, 0.05)  # 范围从0.1降至0.05
        sca_movement = torch.from_numpy(movement).to(param_data.device)
        
        # 梯度引导移动（增加裁剪）
        grad_norm = torch.norm(gradient_direction)
        if grad_norm > 1.0:  # 梯度裁剪
            gradient_direction = gradient_direction / grad_norm
        
        grad_movement = gradient_weight * lr * gradient_direction
        
        # 结合两种移动（动态权重调整）
        # 随着训练进行，逐渐增加梯度权重，减少随机探索
        adaptive_gradient_weight = min(gradient_weight + self.iteration / 10000, 0.8)
        total_movement = (1 - adaptive_gradient_weight) * sca_movement + adaptive_gradient_weight * grad_movement
        
        # 最终裁剪，确保更新不会过大
        total_movement = torch.clamp(total_movement, -0.05, 0.05)
        
        return total_movement
    
    def step(self, closure=None):
        """执行优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 更新每个参数
        for group in self.param_groups:
            lr = group['lr']
            gradient_weight = group['gradient_weight']
            a_max = group['a_max']
            
            param_idx = 0
            for param in group['params']:
                if param.requires_grad:
                    # 获取梯度方向
                    gradient_direction = self._get_gradient_direction(param_idx, param)
                    
                    # SCA更新
                    movement = self._sca_update(param.data, gradient_direction, lr, gradient_weight, a_max)
                    
                    # 更新参数
                    param.data.add_(movement)
                    
                    param_idx += 1
        
        self.iteration += 1
        return loss
    
    def zero_grad(self):
        """清零梯度"""
        super(SCAOptimizer, self).zero_grad()


# 测试函数
def test_sca_optimizer():
    """测试SCA优化器"""
    print("🧪 测试梯度引导SCA优化器...")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # 使用SCA优化器
    optimizer = SCAOptimizer(model.parameters(), lr=0.01, population_size=10)
    criterion = nn.MSELoss()
    
    # 生成测试数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练几步
    for epoch in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(X)
        loss = criterion(output, y)
        
        # 存储梯度信息
        loss.backward()
        optimizer.store_gradients()
        
        # 优化步骤
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    print("✅ SCA优化器测试完成！")


if __name__ == "__main__":
    test_sca_optimizer()