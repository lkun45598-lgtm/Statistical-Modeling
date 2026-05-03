# Method Section Stub

## ReefCastNet-SimVP

本文以 SimVPv2 思路的时空预测骨干为基础，构建面向南海珊瑚礁热胁迫风险预警的 ReefCastNet-SimVP。模型不直接预测绝对海温，而是预测未来多周的海表温度异常（SSTA），再结合训练集气候态与最大月平均温度（MMM）计算 HotSpot、DHW 与白化风险等级。

模型输入包括过去若干周 SSTA、MMM、礁区掩膜、礁区缓冲区掩膜、经纬度位置编码与周序号正余弦编码。相比普通时空预测模型，该设计将预测目标从全海域平均误差最小化转为珊瑚礁热胁迫风险感知预测。

总损失函数为：

```text
L = L_ssta + λ1 L_reef + λ2 L_hotspot + λ3 L_dhw + λ4 L_alert + λ5 L_gradient
```

其中 `L_reef` 对珊瑚礁及其缓冲区赋予更高权重，`L_hotspot` 和 `L_dhw` 分别约束热异常与12周累积热胁迫，`L_alert` 使用 focal loss 处理高风险白化等级样本稀疏问题。
