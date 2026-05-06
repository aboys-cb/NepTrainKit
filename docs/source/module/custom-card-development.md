# 自定义卡片开发

本页教你为 Make Dataset 模块开发一张新卡片，按当前的 Operation/Params 架构拆分 UI 和业务逻辑。

如果你要新增内置卡片，仓库内的 `make-dataset-card-dev` skill 会引导你走完整流程（规格→实现→文档→验证）。本页更适合作快速参考。

## 架构概览

每张卡片由三层组成：

```
CardWidget (UI)
  ├── get_params() → Params dataclass
  ├── set_params(params) → 恢复控件
  └── create_operation() → Operation

CardOperation (纯逻辑，在 core/cards/)
  └── run_structure(structure, params) / run_dataset(dataset, params) / generate(params)

CardParams (frozen dataclass)
  └── 所有参数的类型、默认值、范围
```

- **UI 层**不写算法，只读控件、构造 params、调 operation。
- **Operation 层**不 import `PySide6`、`qfluentwidgets`、`MessageManager`。错误直接抛异常。

## 第一步：选 Operation 类型

| 类型 | 基类 | 签名 | 什么时候用 |
|------|------|------|-----------|
| `StructureOperation` | `core.cards.operation.StructureOperation` | `run_structure(structure, params) -> list[Atoms]` | 单结构变换：应变、扰动、扩胞、掺杂 |
| `DatasetOperation` | `core.cards.operation.DatasetOperation` | `run_dataset(dataset, params) -> list[Atoms]` | 全数据集过滤/排序：FPS |
| `GeneratorOperation` | `core.cards.operation.GeneratorOperation` | `generate(params) -> list[Atoms]` | 无输入生成：晶体原型构建 |

## 第二步：写 Params dataclass

在 `src/NepTrainKit/core/cards/` 对应模块（`lattice.py` / `alloy.py` / `defect.py` / `magnetism.py` / `structure.py` / `filter.py`）中定义：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MyCardParams:
    """Parameters for my card."""
    param_a: str = "default"
    param_b: float = 1.0
    use_seed: bool = False
    seed: int = 0
```

- 必须是 frozen dataclass
- 每个字段给默认值
- 命名用下划线，和 UI 控件的 `get_params()` key 一致

## 第三步：写 Operation

同文件内：

```python
from .operation import StructureOperation  # 或 DatasetOperation / GeneratorOperation

class MyCardOperation(StructureOperation):
    """Pure logic for my card."""

    def run_structure(self, structure, params: MyCardParams) -> list:
        # 校验
        if params.param_b <= 0:
            raise ValueError("param_b must be > 0")

        # 纯算法逻辑
        new_structure = structure.copy()
        # ... do something with params ...
        append_config_tag(new_structure, f"MyTag(...)")
        return [new_structure]
```

规则：
- 不 import UI 库
- 参数校验失败抛异常
- 用 `append_config_tag` 写可追溯标签
- 不做静默回退

## 第四步：写 UI 卡片

在 `src/NepTrainKit/ui/views/_card/` 新建文件，继承 `MakeDataCard`：

```python
from PySide6.QtWidgets import QFrame
from qfluentwidgets import BodyLabel, ComboBox, CheckBox
from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.xxx import MyCardOperation, MyCardParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, MakeDataCard

@CardManager.register_card
class MyCard(MakeDataCard):
    group = "MyGroup"
    card_name = "My Card"
    menu_icon = r":/images/src/images/my_icon.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        # 构建控件
        ...

    # ---- 必须实现的三个方法 ----

    def create_operation(self):
        return MyCardOperation()

    def get_params(self) -> MyCardParams:
        return MyCardParams(
            param_a=self.combo.currentText(),
            param_b=float(self.spinbox.get_input_value()[0]),
            ...
        )

    def set_params(self, params: MyCardParams) -> None:
        self.combo.setText(params.param_a)
        self.spinbox.set_input_value([params.param_b])
        ...

    # ---- 序列化（必须实现） ----

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw = data.get("params")
        if raw is not None:
            params = MyCardParams(**raw)
        else:
            params = MyCardParams()  # 全部默认值
        self.set_params(params)
```

> **只在迁移旧卡片时才从旧 key 恢复**。上面的模板是新卡写法，`from_dict` 只读 `params`。如果你在迁移一张已有用户的旧卡片，可以在 `else` 分支里从 `data["old_key"]` 等旧字段构造 Params 以保证向后兼容。

### 关键规则

- **禁止覆盖 `run()`**：基类已根据 `create_operation()` 返回的类型自动分发到正确线程。
- **禁止在 UI 里写算法**：`init_ui()` 只建控件，`get_params()` 只读值，算法全在 Operation 里。
- **`process_structure()` 不新增**：如果要保留旧兼容层，只做一行委托 `return self.create_operation().run_structure(structure, self.get_params())`。
- 数值参数用 `SpinBoxUnitInputFrame`，枚举用 `ComboBox`，开关用 `CheckBox`，字符串用 `LineEdit`。

## 第五步：注册、文档、测试

- `@CardManager.register_card` 装饰 UI 类。
- 在 `src/NepTrainKit/ui/views/_card/__init__.py` 导入并加入 `__all__`。
- 写文档页 `docs/source/module/make-dataset-cards/cards/my-card.md`，必须从训练集诊断场景出发。参见 [卡片文档编写规范](make-dataset-cards/writing-guide.md)。
- 写 operation 测试（不需要 Qt）：
  ```python
  def test_my_card_operation():
      op = MyCardOperation()
      result = op.run_structure(test_atoms, MyCardParams(param_b=2.0))
      assert len(result) > 0
  ```

## 验证

```bash
python skills/make-dataset-card-dev/scripts/run_card_checks.py --quick
python tools/docs/audit_card_docs.py
```

## 放置自定义卡片

内置卡片放在 `src/NepTrainKit/ui/views/_card/`。外部自定义卡片放在用户配置目录的 `cards/` 子目录下，程序启动时会自动加载。
