# Make Dataset

## 1. Core Concepts & Workflow

### 1.1 Data Flow Model
- **Linear Processing Chain**: Cards execute sequentially, with each card's output automatically becoming the next card's input
- **In-Group Parallel Flow**: All cards within a group share the same input, with outputs automatically merged
- **Filter Mechanism**: Filters can be added at group ends to screen outputs from all group cards

### 1.2 Basic Operations
1. **Import Structures**:
   - Supported formats: VASP/POSCAR, CIF, XYZ
   - Methods: Click "Open" button or drag files directly into window

2. **Build Processing Pipeline**:
   - Add processing cards via "Add new card"
   - Reorder cards via drag-and-drop
   - Use Card Groups for complex workflows

3. **Save/Load Configuration**:
   - Export: Save current card setup as JSON
   - Import: Load existing configuration files

Sample configuration file:
```json
{
    "software_version": "2.0.6.dev35",
    "cards": [
        {
            "class": "SuperCellCard",
            "check_state": true,
            "super_cell_type": 0,
            "super_scale_radio_button": false,
            "super_scale_condition": [1,1,1],
            "super_cell_radio_button": true,
            "super_cell_condition": [20,20,20],
            "max_atoms_radio_button": false,
            "max_atoms_condition": [1]
        },
        {
            "class": "CardGroup",
            "check_state": true,
            "card_list": [
                {
                    "class": "CellStrainCard",
                    "check_state": true,
                    "engine_type": "triaxial",
                    "x_range": [-5,5,1],
                    "y_range": [-5,5,1],
                    "z_range": [-5,5,1]
                },
                {
                    "class": "PerturbCard",
                    "check_state": true,
                    "engine_type": 0,
                    "organic": true,
                    "scaling_condition": [0.3],
                    "num_condition": [50]
                },
                {
                    "class": "CellScalingCard",
                    "check_state": true,
                    "engine_type": 0,
                    "perturb_angle": true,
                    "scaling_condition": [0.04],
                    "num_condition": [50]
                }
            ],
            "filter_card": {
                "class": "FPSFilterDataCard",
                "check_state": true,
                "nep_path": "D:\\PycharmProjects\\NepTrainKit\\src\\NepTrainKit\\Config\\nep89.txt",
                "num_condition": [100],
                "min_distance_condition": [0.001]
            }
        }
    ]
}
```

## 2. Production Cards Explained

### 2.1 Super Cell Generation
**Function**: Creates supercells through expansion

**Parameters**:
| Parameter Group | Option | Description | Typical Values |
|-----------------|--------|-------------|----------------|
| Mode | Maximum | Generates largest possible supercell | - |
|      | Iteration | Generates all possible combinations | - |
| Expansion Method | Super Scale | Fixed expansion multiplier | (2,2,2) |
|                 | Super Cell | Calculates by max lattice constant | (10Å,10Å,10Å) |
|                 | Max Atoms | Limits by maximum atom count | 200 |

**Structure Tagging**:
```python
structure.info["Config_type"] += "supercell(nx,ny,nz)"  # e.g., supercell(2,2,1)
```

### 2.2 Vacancy Defect Generation
**Function**: Creates vacancy-defect structures by deleting random atoms

**Key Parameters**:
- **Random engine**: `Sobol` sequence or `Uniform` distribution
- **Vacancy specification**:
  - *Vacancy num*: fixed number of vacancies
  - *Vacancy concentration*: fraction of atoms to remove
- **Max structures**: number of structures to generate

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Vacancy(num={defect_count})"
```

### 2.3 Atomic Perturbation
**Function**: Adds random displacements to atomic positions

**Key Parameters**:
- **Random engine**: `Sobol` or `Uniform`
- **Max distance**: maximum displacement amplitude in Å
- **Identify organic**: treat organic molecules as rigid units
- **Max structures**: number of structures to generate

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Perturb(distance={max_displacement}, {engine_type})"
```

### 2.4 Lattice Scaling
**Function**: Randomly scales lattice vectors

**Key Parameters**:
- **Random engine**: `Sobol` or `Uniform`
- **Max scaling**: 0–1, applied symmetrically as `1±value`
- **Perturb angle**: whether lattice angles are also perturbed
- **Identify organic**: treat organic molecules as rigid units
- **Max structures**: number of structures to generate

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Scaling({scaling_factor})"
```

### 2.5 Lattice Strain
**Strain Modes**:
- Uniaxial
- Biaxial
- Triaxial
- Isotropic (uniform scaling; only X range is used)
- **Custom Axis Combinations**: Supports any XYZ combinations (e.g., "XY", "XZ", "YZX")
  ```python
  # Example: Apply strain only to X and Z axes
  strain_axes = "XZ"  # Equivalent to "ZX"
  ```

**Key Parameters**:
- **Axes**: built‑in modes or custom strings like `X`, `XY`
- **X/Y/Z range**: strain percentage ranges. In isotropic mode only `X` values are used
- **Identify organic**: treat organic molecules as rigid units

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Strain({axis1}:{value1}%, {axis2}:{value2}%)"
```

### 2.6 Random Doping Substitution
**Function**: Randomly substitute atoms according to user-defined rules

**Key Parameters**:
- **Doping rules**: each rule contains a target element, dopant elements and their ratios,
  a concentration or count range, and optional groups
- **Doping algorithm**: `Random` (sample dopants by probability) or `Exact` (follow ratios exactly)
- **Max structures**: number of structures to generate

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Doping(num={dopant_count})"
```

### 2.7 Random Vacancy Deletion
**Function**: Removes atoms according to vacancy rules

**Key Parameters**:
- **Vacancy rules**: each rule specifies an element, a deletion count range,
  and optional groups to restrict affected sites
- **Max structures**: number of structures to generate

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Vacancy(num={removed_count})"
```

### 2.8 Random Slab Generation
**Function**: Builds slabs with random Miller indices and vacuum thickness

**Key Parameters**:
- **h/k/l range**: Miller index ranges
- **Layer range**: minimum, maximum and step
- **Vacuum range**: vacuum thickness in Å

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Slab(hkl={h}{k}{l},layers={layers},vacuum={vac})"
```

## 3. Filter Cards

### 3.1 FPS Filter (Farthest Point Sampling)
**Algorithm**:
1. Calculates NEP descriptors for all structures
2. Executes FPS algorithm in high-dimensional space

**Key Parameters**:
- NEP file path (required)
- Maximum selection count
- Minimum distance threshold

**Filter Mechanism**:
- Filters only affect exported results, not data flow
- Export logic:
  ```python
  if filter_active:
      export_filtered_results
  else:
      export_raw_merged_results
  ```

## 4. Container Cards

### 4.1 Card Group
**Usage Guide**:
1. **Create Group**: Add Card Group card
2. **Add Members**: Drag cards into group
3. **Set Filter**: Drag filter card to group bottom area

**Execution Example**:
- **Scenario**: 3 group cards generating 10, 15, and 20 structures respectively
- **Without Filter**: Passes 45 structures to next stage
- **With Filter**: Passes 45 structures but may only export 30

# NepTrainKit Custom Card Development Guide

## 1. Development Environment Setup

### 1.1 Card Directory Structure
```
User_Config_Directory/
├── cards/
│   ├── custom_card1.py  # Custom card files
│   └── custom_card2.py
```

### 1.2 Get Config Directory Path
```python
import os
import platform

def get_user_config_path():
    if platform.system() == 'Windows':
        local_path = os.getenv('LOCALAPPDATA', None)
        if local_path is None:
            local_path = os.getenv('USERPROFILE', '') + '\\AppData\\Local'
        user_config_path = os.path.join(local_path, 'NepTrainKit')
    else:
        user_config_path = os.path.expanduser("~/.config/NepTrainKit")
    return user_config_path
```
Default paths:
Windows: C:\Users\Username\AppData\Local\NepTrainKit\
Linux: ~/.config/NepTrainKit

## 2. Card Development Template

### 2.1 Basic Template Structure
```python
from NepTrainKit.core.views.cards import MakeDataCard, register_card_info

@register_card_info
class CustomCard(MakeDataCard):
    # Required class attributes
    card_name = "Custom Card Name"
    menu_icon = ":/images/src/images/logo.svg"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Card Title")
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setObjectName("custom_card_widget")
        # Add controls and layout code here
    
    def process_structure(self, structure):
        """Core processing logic"""
        processed_structures = []
        # Processing code...
        return processed_structures
    
    def to_dict(self):
        """Serialize card configuration"""
        return super().to_dict()
        
    def from_dict(self, data_dict):
        """Deserialize configuration"""
        super().from_dict(data_dict)
        # Custom parameter restoration...
```

## 3. Core Function Implementation

### 3.1 Processing Function Specification
```python
def process_structure(self, structure):
    """
    Parameters:
        structure (ase.Atoms): Input structure object
    
    Returns:
        List[ase.Atoms]: Processed structure list
    
    Notes:
        - Must return a list, even with single structure
        - Each structure should use copy() to avoid modifying original
    """
    new_structure = structure.copy()
    # Processing logic...
    return [new_structure]
```

### 3.2 UI Development Recommendations
```python
def init_ui(self):
    # Example: Add parameter input
    from qfluentwidgets import SpinBox, BodyLabel
    
    self.param_label = BodyLabel("Parameter:", self)
    self.param_input = SpinBox(self)
    self.param_input.setRange(1, 100)
    self.param_input.setValue(10)
    
    self.settingLayout.addWidget(self.param_label, 0, 0)
    self.settingLayout.addWidget(self.param_input, 0, 1)
```

## 4. Advanced Features

### 4.1 State Persistence
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

## Appendix: Complete Example Card
https://github.com/aboys-cb/NepTrainKit/blob/master/src/NepTrainKit/core/views/cards.py