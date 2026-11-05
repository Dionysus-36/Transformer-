# Adam 优化器与 warmup / 逆平方根学习率调度

from torch import nn
from torch.optim import Adam


"""
create_optimizer：创建论文使用的 Adam 优化器

输入：
    model：参与 DP 训练的模型
返回：
    optimizer：管理模型参数的 Adam 优化器

初始学习率设为 0，每次参数更新前由 update_learning_rate 设置。
"""
def create_optimizer(model: nn.Module) -> Adam:
    return Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)


"""
update_learning_rate：按当前参数更新次数设置学习率

输入：
    optimizer：上面创建的优化器
    step：即将进行的第几次参数更新，从 1 开始
    d_model：模型特征维数，由模型配置提供
    warmup_steps：学习率线性增大的更新次数，论文为 4000
返回：
    lr：本次参数更新使用的学习率，浮点数

梯度累积的多个 batch 合起来算一次参数更新，只调用一次本函数。
"""
def update_learning_rate(optimizer: Adam, step, d_model, warmup_steps=4000) -> float:
    lr = d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
    for param_group in optimizer.param_groups:
        # 把更新后的学习率写入优化器，供下一次参数更新使用
        param_group["lr"] = lr
    return lr