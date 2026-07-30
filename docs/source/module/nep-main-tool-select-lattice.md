# 按晶格参数选择（Select by Lattice）

## 作用与适用场景

按照晶格长度 `a/b/c` 和晶格角 `α/β/γ` 选择结构。适合从混合数据中找特定尺寸、
体形或畸变范围的晶胞，不依赖当前散点图坐标。

## 判定原理

对每个结构，程序从晶胞矩阵计算

$$
(a,b,c,\alpha,\beta,\gamma),
$$

其中 $a,b,c$ 是三条晶格矢量长度，单位为 Å；$\alpha,\beta,\gamma$ 分别是
$\mathbf b$ 与 $\mathbf c$、$\mathbf a$ 与 $\mathbf c$、$\mathbf a$ 与
$\mathbf b$ 的夹角，单位为度。结构只有在六个量都落入各自闭区间时才会被选中：

$$
M(i)=\bigwedge_{q\in\{a,b,c,\alpha,\beta,\gamma\}}
\left(q_{\min}-10^{-4}\le q_i\le q_{\max}+10^{-4}\right).
$$

固定的 $10^{-4}$ 容差用于吸收浮点换算误差，并不是可调的物理容差。若某一维不想参与筛选，
应把该维范围放宽到覆盖全部数据。

## 参数与执行结果

```{include} nep-dataset-display-content.md
:start-after: <!-- display-select-lattice-start -->
:end-before: <!-- display-select-lattice-end -->
```

## 操作后检查

六组范围同时生效。若没有选中结果，先放宽不关心的晶格角或长度范围，再逐项收紧，
避免把无关维度设成过窄的过滤条件。
