# 最大误差结构：先复核，再决定删不删

`Find Max Error Point` 用来快速定位当前图上误差最大的结构。高误差不等于坏数据：它也可能表示训练集缺少这一类结构。正确顺序是先选中、看结构和 `Config_type`，再决定重新计算、保留补强或删除。

## 1. 先激活要检查的图

打开训练结果后，双击要检查的小图，使它进入主图区。本例检查 `energy`；如果要找力误差最大的结构，应先激活 `force` 图。

```{image} ../_static/image/generated/tutorials/max_error_review_entry.png
:alt: 激活能量图并打开最大误差筛选
:class: docs-screenshot
```

点击 `Find Max Error Point`。这个按钮只针对当前主图，不会同时混合 energy、force、stress 和 virial 的误差。

## 2. 输入要复核的结构数

```{image} ../_static/image/generated/show_nep_reference/g_maxerr_dialog.png
:alt: 最大误差结构数量设置
:class: docs-screenshot
```

第一次可以先填 `5`。数量太大时容易失去重点；先看最极端的几个，确认问题类型后再扩大范围。

## 3. 逐个判断误差来源

```{image} ../_static/image/generated/tutorials/max_error_review_result.png
:alt: 选中的五个最大能量误差结构
:class: docs-screenshot
```

本例底部状态栏显示 `Sel: 5`。依次点击选中点，重点看右侧结构、`Config_type` 和原始文件序号：

- 结构合理、标签可信：通常应保留，并检查这一类构型是否在训练集中覆盖不足；
- 结构合理但 DFT 未收敛或标签异常：回到原计算重新检查，不要直接拿去重训；
- 结构明显破坏、原子重叠或来源错误：确认后再删除，或者先 `Export Selected` 单独留档。

不要把“删除最大误差点”当成降低 RMSE 的固定步骤。未经复核地删除，可能正好删掉模型最需要补学的区域。
