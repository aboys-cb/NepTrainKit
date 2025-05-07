# Make dataset


# NepTrainKit 结构数据处理工具使用手册

## 1. 核心概念与工作流程

### 1.1 数据流模型
- **线性处理链**：卡片按添加顺序执行，前一个卡片的输出自动成为下一个卡片的输入
- **组内并行流**：卡片组内所有卡片共享同一输入，输出结果自动合并
- **过滤机制**：可在卡片组末尾添加过滤器对组内所有卡片的输出进行筛选

### 1.2 基本操作
1. **导入结构**：
   - 支持格式：VASP/POSCAR、CIF、XYZ
   - 方式：点击"打开"按钮或直接拖拽文件到窗口

2. **构建处理流程**：
   - 通过"Add new card"添加处理卡片
   - 拖拽卡片调整执行顺序
   - 使用Card Group组织复杂流程

3. **保存/加载配置**：
   - 导出：保存当前卡片配置为JSON
   - 导入：加载已有配置文件

## 2. 生产类卡片详解

### 2.1 Super Cell（超胞生成）
![Super Cell卡片示意图]

**功能**：通过扩胞操作生成超胞结构

**参数配置**：
| 参数组 | 选项 | 说明 | 典型值 |
|--------|------|------|--------|
| 行为模式 | Maximum | 生成最大可能的超胞 | - |
|        | Iteration | 生成所有可能的组合 | - |
| 扩胞方式 | Super Scale | 固定扩胞倍数 | (2,2,2) |
|        | Super Cell | 按最大晶格常数计算 | (10Å,10Å,10Å) |
|        | Max Atoms | 按最大原子数限制 | 200 |

**结构标记规则**：
  ```python
  structure.info["Config_type"] +="supercell(nx,ny,nz)"  # 例如：supercell(2,2,1)
  ```

### 2.2 Vacancy Defect Generation（空位缺陷生成）
![Vacancy Defect卡片示意图]

**功能**：创建含空位缺陷的结构集合

**结构标记规则**：
```python
structure.info["Config_type"] += f" Vacancy(num={缺陷数量})"
```
**技术细节**：
- Sobol引擎：使用准随机序列保证缺陷分布均匀性
- 有机分子处理：自动识别并保持分子完整性

### 2.3 Atomic Perturb（原子微扰）
![Perturb卡片示意图]

**扰动参数**：
- 最大位移：0.1-1.0Å（建议值）
- 采样数：50-1000
- 高级选项：
  - 有机分子识别（默认开启）
  - 随机引擎选择

**结构标记规则**：
```python
structure.info["Config_type"] += f" Perturb(dist={最大位移}Å, {engine_type})"
```

### 2.4 Lattice Scaling（晶格缩放）
![Scaling卡片示意图]

**参数矩阵**：

| 参数 | 范围 | 步长 | 单位 |
|------|------|------|------|
| 缩放系数 | 0.9-1.1 | 0.01 | - |
| 角度扰动 | On/Off | - | - |
| 结构数 | 1-1000 | - | - |

**结构标记规则**：
```python
structure.info["Config_type"] += f" Scaling({缩放系数})"
```

### 2.5 Lattice Strain（晶格应变）
![Strain卡片示意图]

**应变模式**：
- 单轴（Uniaxial）
- 双轴（Biaxial）
- 三轴（Triaxial）
- **自定义轴组合**：支持任意XYZ字母组合（如"XY"、"XZ"、"YZX"等）
  ```python
  # 示例：仅对X和Z轴施加应变
  strain_axes = "XZ"  # 等效于"ZX"
  ```
- **结构标记规则**：
  ```python
  structure.info["Config_type"] += f" Strain({axis1}:{value1}%, {axis2}:{value2}%)"
  ```

## 3. 过滤类卡片

### 3.1 FPS Filter（最远点采样过滤）
![FPS卡片示意图]

**算法流程**：
1. 计算所有结构的NEP描述符
2. 在高维空间执行FPS算法：
 

**关键参数**：
- NEP文件路径（必需）
- 最大选择数
- 最小间距阈值
3. **过滤机制**：
   - 过滤器仅影响导出结果，不影响数据传递
   - 导出时逻辑：
     ```python
     if 过滤器激活:
         导出过滤后结果
     else:
         导出原始合并结果
     ```

## 4. 容器卡片

### 4.1 Card Group（卡片组）
![Card Group示意图]

**操作指南**：
1. **创建组**：添加Card Group卡片
2. **添加成员**：拖拽其他卡片到组内
3. **设置过滤**：将过滤卡片拖到组底部区域

4. **执行示例**：
   - **场景**：组内3个卡片分别生成10、15、20个结构
   - **无过滤**：传递45个结构到下一环节
   - **有过滤**：传递45个结构，但导出时可能只保留30个
# NepTrainKit 自定义卡片开发指南

## 1. 开发环境准备

### 1.1 卡片目录结构
```
用户配置目录/
├── cards/
│   ├── custom_card1.py  # 自定义卡片文件
│   └── custom_card2.py
```

### 1.2 获取配置目录路径
```python
import os
import platform

def get_user_config_path():
    if platform.system() == 'Windows':
        local_path = os.getenv('LOCALAPPDATA', None)
        if local_path is None:
            local_path = os.getenv('USERPROFILE', '') + '\\AppData\\Local '
        user_config_path = os.path.join(local_path, 'NepTrainKit')
    else:
        user_config_path = os.path.expanduser("~/.config/NepTrainKit")
    return user_config_path
```
一般情况下目录为：
windows：C:\Users\用户名\AppData\Local\NepTrainKit\
linux：~/.config/NepTrainKit
## 2. 卡片开发模板

### 2.1 基础模板结构
```python
from NepTrainKit.core.views.cards import MakeDataCard, register_card_info

@register_card_info
class CustomCard(MakeDataCard):
    # 必须定义的类属性
    card_name = "自定义卡片名称"
    menu_icon = ":/images/src/images/default_icon.svg"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("卡片标题")
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setObjectName("custom_card_widget")
        # 在此添加控件和布局代码
    
    def process_structure(self, structure):
        """核心处理逻辑"""
        processed_structures = []
        # 处理代码...
        return processed_structures
    
    def to_dict(self):
        """序列化卡片配置"""
        return {
            'class': self.__class__.__name__,
            'name': self.card_name,
            'check_state': self.check_state,
            # 自定义参数...
        }
    
    def from_dict(self, data_dict):
        """反序列化配置"""
        self.state_checkbox.setChecked(data_dict['check_state'])
        # 自定义参数恢复...
```

## 3. 核心功能实现

### 3.1 处理函数规范
```python
def process_structure(self, structure):
    """
    参数:
        structure (ase.Atoms): 输入的结构对象
    
    返回:
        List[ase.Atoms]: 处理后的结构列表
    
    注意:
        - 必须返回列表，即使只有一个结构
        - 每个结构应使用copy()避免修改原始数据
    """
    new_structure = structure.copy()
    # 处理逻辑...
    return [new_structure]
```

### 3.2 界面开发建议
```python
def init_ui(self):
    # 示例：添加一个参数输入框
    from qfluentwidgets import SpinBox, BodyLabel
    
    self.param_label = BodyLabel("参数值:", self)
    self.param_input = SpinBox(self)
    self.param_input.setRange(1, 100)
    self.param_input.setValue(10)
    
    self.settingLayout.addWidget(self.param_label, 0, 0)
    self.settingLayout.addWidget(self.param_input, 0, 1)
```

## 4. 高级功能实现

### 4.1 状态持久化
```python
def to_dict(self):
    data = super().to_dict()
    data.update({
        'custom_param': self.param_input.value(),
        'other_setting': True
    })
    return data

def from_dict(self, data):
    super().from_dict(data)
    self.param_input.setValue(data.get('custom_param', 10))
```

### 4.2 进度反馈
```python
def process_structure(self, structure):
    total = 100  # 总步骤数
    for i in range(total):
        # 处理逻辑...
        progress = int((i+1)/total*100)
        self.progressSignal.emit(progress)  # 发送进度信号
```

## 5. 调试与测试

### 5.1 调试建议
```python
# 在process_structure中添加日志
from loguru import logger

def process_structure(self, structure):
    logger.debug(f"Processing structure with {len(structure)} atoms")
    try:
        # 处理代码...
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
```

### 5.2 单元测试示例
```python
def test_custom_card():
    from ase.build import molecule
    card = CustomCard()
    test_structure = molecule('H2O')
    
    # 测试处理功能
    results = card.process_structure(test_structure)
    assert isinstance(results, list)
    assert len(results) > 0
    
    # 测试序列化
    config = card.to_dict()
    new_card = CustomCard()
    new_card.from_dict(config)
```

## 6. 发布与共享

### 6.1 卡片打包建议
```
my_card_package/
├── __init__.py
├── card_definition.py
└── resources/
    └── icon.svg
```

### 6.2 安装方式
用户只需将卡片文件(.py)放入cards目录即可自动加载

## 附录：完整示例卡片

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from NepTrainKit.core.views.cards import MakeDataCard, register_card_info
from qfluentwidgets import SpinBox, BodyLabel
from ase import Atoms
import numpy as np

@register_card_info
class RandomDisplacementCard(MakeDataCard):
    card_name = "随机位移"
    menu_icon = ":/images/src/images/perturb.svg"
    separator = True
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("原子随机位移")
        self.init_ui()
    
    def init_ui(self):
        self.setObjectName("random_disp_card")
        
        # 位移幅度设置
        self.disp_label = BodyLabel("最大位移 (Å):", self)
        self.disp_input = SpinBox(self)
        self.disp_input.setRange(0.01, 5.0)
        self.disp_input.setValue(0.3)
        self.disp_input.setSingleStep(0.05)
        
        # 生成数量设置
        self.num_label = BodyLabel("生成数量:", self)
        self.num_input = SpinBox(self)
        self.num_input.setRange(1, 1000)
        self.num_input.setValue(50)
        
        self.settingLayout.addWidget(self.disp_label, 0, 0)
        self.settingLayout.addWidget(self.disp_input, 0, 1)
        self.settingLayout.addWidget(self.num_label, 1, 0)
        self.settingLayout.addWidget(self.num_input, 1, 1)
    
    def process_structure(self, structure):
        max_disp = self.disp_input.value()
        num_structures = self.num_input.value()
        
        structures = []
        for _ in range(num_structures):
            new_struct = structure.copy()
            displacements = np.random.uniform(
                -max_disp, max_disp, 
                size=(len(structure), 3)
            
            new_struct.positions += displacements
            new_struct.info["random_disp"] = max_disp
            structures.append(new_struct)
        
        return structures
    
    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'name': self.card_name,
            'check_state': self.check_state,
            'max_disp': self.disp_input.value(),
            'num_structures': self.num_input.value()
        }
    
    def from_dict(self, data):
        self.state_checkbox.setChecked(data['check_state'])
        self.disp_input.setValue(data.get('max_disp', 0.3))
        self.num_input.setValue(data.get('num_structures', 50))
```