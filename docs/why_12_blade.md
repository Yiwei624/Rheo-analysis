# 为什么 12-blade 最好

这个页面对应你文稿里想加入的 “why 12 is the best”。

## 一句话版本

12-blade 不是因为“绝对不会 partial yield”，而是因为它在当前实验窗口里给出了 **最好的平衡**：

1. **牛顿流体 benchmark 最准**：低剪切扭矩传递更可靠。
2. **比 4-blade 更稳定**：散点更小，耦合更稳。
3. **比 24-blade / C25-like 更早进入 fully mobilized regime**：更容易在可测应力窗口里拿到更接近本征的流动响应。

## 机制解释

在你的框架里，“最好”不是简单等于“最粗糙”或者“blade 越多越好”，而是看它能不能同时满足三件事：

- 低扭矩下信号可靠
- 不要因为耦合太差而产生额外 scatter
- 尽快进入 full-gap mobilization，而不是长期停留在 partially yielded regime

12-blade 正好落在这个平衡点：

- **4-blade**：更容易出现耦合不足、波动更大。
- **24-blade / C25-like**：更接近 quasi-cylindrical coupling，低剪切 plateau 更明显，部分屈服窗口更宽。
- **12-blade**：既保持了较好的 torque transmission，又没有把系统长期锁在部分屈服主导区。

## 在软件里怎么体现

这个重建版软件里专门保留了三类证据：

- Newtonian benchmark 的拟合和误差
- yogurt 的 up/down sweep + hysteresis
- partial-yield 指标与 r_y / r2

这样你后面可以把“12-blade 最好”写成 **数据支持的 regime-accessibility conclusion**，而不是只写成经验判断。
