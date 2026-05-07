# 数据管理（Data Management）

`Data Management` 用来记录项目、模型版本、数据路径和备注。它不负责生成结构，也不负责训练；
它解决的是另一个问题：当你有很多轮 DFT、很多个 `nep.txt`、很多个候选池和训练结果时，
还能知道每个文件属于哪一次实验。

## 什么时候值得用

下面这些情况建议开始记录：

- 同一个材料体系已经做了多轮主动学习。
- 每一轮都有候选结构、清洗结构、DFT 结果、训练模型和测试结果。
- 你需要对比不同 `nep.txt` 的误差、备注和适用范围。
- 你经常回头找“上次那个效果好的模型在哪个目录”。

如果只是临时试一张卡片，不一定需要马上建项目。

## 推荐记录方式

可以按材料体系或研究主题建立 `Project`，再把每一轮模型作为 `Model(version)` 记录进去。

一个常见结构是：

```text
Project: TiH2 surface defects
  v001: 初始 bulk + surface 数据
  v002: 加入 vacancy / adsorbate 候选结构
  v003: 清洗后 FPS 采样并完成 DFT
  v004: 回看训练误差后补高力结构
```

每个版本建议至少记录：

- 模型或结果目录路径。
- 对应的训练集来源。
- 简短备注：这一轮补了什么结构、删了什么异常、模型适合什么范围。
- 标签：例如 `surface`、`defect`、`magnetic`、`failed`、`baseline`。

## 和主流程的关系

`Data Management` 更适合放在每轮流程结束后使用：

```text
Make Dataset 生成候选结构
-> NEP Dataset Display 清洗
-> DFT
-> GPUMD 训练
-> NEP Dataset Display 回看
-> Data Management 记录这一轮
```

这样记录的是已经有明确意义的版本，而不是每个临时文件。

## 常用操作

- 新建、修改、删除 `Project`。
- 新建、修改、删除 `Model(version)`。
- 打开本地目录或 URL。
- 用 `Ctrl+F` 搜索项目、标签、备注或路径。

## 存储位置

本地数据库默认放在用户配置目录：

- Windows：`C:\Users\<You>\AppData\Local\NepTrainKit\mlpman.db`
- Linux：`~/.config/NepTrainKit/mlpman.db`

这个数据库只记录管理信息，不会自动复制你的大体积训练数据。移动训练目录后，记得更新记录里的路径。
