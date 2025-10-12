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

#### Algorithm
- Super Scale: directly uses `[na, nb, nc]` as diagonal expansion factors and builds `make_supercell(structure, diag([na,nb,nc]))`.
- Super Cell (max lattice): computes `[na, nb, nc] = floor([amax/a, bmax/b, cmax/c])` with safety checks, then clamps to at least 1 in each direction.
- Max Atoms: enumerates integer triplets `[na, nb, nc]` whose product times `N_atoms` does not exceed the limit; sorted by total atoms, return either all (Iteration) or the largest (Maximum).

#### Caveats
- If any lattice vector length is near zero, max‑cell estimation falls back to 1 to avoid divide‑by‑zero.
- Very large supercells can create memory pressure; prefer Max Atoms to cap size.
- Iteration may generate many structures—use a follow‑up filter (e.g., FPS) to down‑select.

#### Best Practices
- Use Max Atoms to scale safely across diverse inputs.
- Prefer Iteration when exploring and Maximum when preparing a single supercell.
- Combine with Lattice Strain or Perturb after expansion to enrich diversity.

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

#### Algorithm
- If using concentration `c`, compute `max_defects = floor(c * N_atoms)`; else use fixed number.
- Engine `Sobol`: sample both defect count and positions deterministically in [0,1), then map to indices.
- Engine `Uniform`: draw a random integer count and random, unique atom indices to remove.

#### Caveats
- A concentration of 1.0 is clamped to avoid removing all atoms.
- Removing many atoms can break intended stoichiometry or PBC artifacts—validate downstream.

#### Best Practices
- Keep `c` small (e.g., ≤ 0.2) for incremental datasets.
- Prefer Sobol for repeatable sweeps; use Uniform for stochastic augmentation.

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

#### Algorithm
- Builds a 3N‑dimensional random vector from Sobol or Uniform in [‑1,1], reshaped to N×3, then scaled by `max_distance` and added to atomic positions.
- If “Identify organic” is enabled: organic clusters are translated as rigid units; inorganic clusters perturb per‑atom.
- Wraps atoms back into the cell.

#### Caveats
- Large `max_distance` may cause unrealistic overlaps even with wrapping.
- Cluster detection is heuristic; verify for complex organics.

#### Best Practices
- Start small (0.1–0.3 Å) and inspect min interatomic distances.
- Combine with FPS Filter to keep diverse displacements while avoiding redundancy.



#### Vibrational Mode Perturb Card
- **Purpose**: reuse precomputed vibrational eigenmodes (or normal modes) to introduce physically guided displacements.
- **Data expectation**: modes and frequencies are stored as per-atom arrays (EXTXYZ columns). Columns should follow the pattern `vibration_mode_<index>_x/y/z`, with optional `vibration_frequency_<index>` (constant per atom). Packed arrays named `vibration_modes` or `normal_modes` with shape `(natoms, 3 * nmodes)` are also recognised.
- **Processing**:
  1. Assemble each mode matrix from the available columns.
  2. Sample a configurable number of modes and random coefficients (Normal or Uniform).
  3. Optionally divide coefficients by `sqrt(|frequency|)` and drop near-zero-frequency modes.
  4. Sum the weighted displacements, scale by the requested amplitude, update positions, and wrap.
- **Usage tips**:
  - Ensure the upstream exporter writes the mode columns; missing data results in no generated structures.
  - Keep amplitudes modest (≤0.05 Å) to remain in the harmonic regime.
  - Frequencies are optional but recommended when enabling frequency-based scaling.


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

#### Algorithm
- Random engine produces a sequence of factors; lattice lengths are scaled by `1 ± s` (uniform or Sobol).
- If “Perturb angle” is enabled, angles are also perturbed and a new lattice is reconstructed from (lengths, angles).
- Optionally treats organics as rigid clusters during scaling.

#### Caveats
- Angle perturbation rebuilds the lattice; extreme angles can generate ill‑conditioned cells.
- Ensure `Max scaling` is small (≤ 0.05) when also perturbing angles.

#### Best Practices
- Use Uniform for broad coverage; Sobol for low‑discrepancy sweeps.
- Keep scaling conservative for stability, especially before DFT relaxations.

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

#### Algorithm
- Builds mesh over selected axes: `np.arange(start, end+step, step)` per axis; isotropic uses a single scalar applied to all.
- Scales selected lattice vectors by `(1 + strain/100)`; optionally applies rigid handling of organic clusters.

#### Caveats
- Large strains (>10–15%) can produce severe distortions; prefer smaller increments.
- Custom axis strings must only include X/Y/Z.

#### Best Practices
- Use biaxial/triaxial for comprehensive scans; isotropic for quick sweeps.
- Pair with FPS or error‑based selection to trim the grid.

### 2.6 Random Doping Substitution
**Function**: Randomly substitute atoms according to user-defined rules

**Key Parameters**:
- **Doping rules**: each rule contains a target element, dopant elements and their ratios,
  a concentration or count range, and optional groups
- **Doping algorithm**: `Random` (sample dopants by probability) or `Exact` (follow ratios exactly)
- **Max structures**: number of structures to generate

### 2.7 Organic Molecular Rotation (TorsionGuard, PBC-aware)
**Function**: Generate organic molecular conformations by rotating torsional subtrees around rotatable bonds, with physical guards to keep bonded distances reasonable and avoid non‑bonded clashes. Supports PBC or non‑PBC frames and adds optional Gaussian noise.

**Key Parameters**:
- Confs per structure (`perturb_per_frame`): number of conformations generated per input structure
- Torsion range (`torsion_range_deg`): angle range in degrees, e.g., [-180, 180]
- Max torsions/conf (`max_torsions_per_conf`): maximum different torsion bonds to rotate per conformation
- Gaussian sigma (`gaussian_sigma`): standard deviation (Å) of added positional noise
- PBC mode (`pbc_mode`): `auto` (use cell if present), `yes` (force PBC if lattice exists), `no` (non‑PBC)
- Local‑mode cutoff atoms (`local_mode_cutoff_atoms`): if atom count exceeds this, rotate only a local subtree for efficiency
- Local torsion max subtree (`local_torsion_max_subtree`): max atoms in the rotated local subtree
- Bond detect factor (`bond_detect_factor`): scale on (r_i + r_j) to detect bonded pairs when building the graph
- Bond keep min/max factor (`bond_keep_min_factor`, `bond_keep_max_factor`): allowed bonded distance window as factors of (r_i + r_j); if max is None, uses detect factor
- Non‑bond min factor (`nonbond_min_factor`): minimum separation factor for non‑bonded pairs to avoid clashes
- Max retries/conf (`max_retries_per_frame`): retries with smaller angle/noise scales if guards fail
- Multi‑bond factor (`mult_bond_factor`): skip “short” edges (likely multiple bonds) from torsion set
- Non‑PBC box size (`nonpbc_box_size`): cubic box length used to center non‑PBC outputs for visualization

**Algorithm**:
- Build adjacency from covalent radii using `bond_detect_factor`; find rotatable torsions via graph analysis (prefer bridges; fall back to internal bonds)
- For each conformation:
  - Randomly pick up to `max_torsions_per_conf` torsion axes
  - For each, collect a local subtree to rotate; in PBC use MIC to define the bond axis
  - Apply a random rotation within `torsion_range_deg`, then add Gaussian noise (`gaussian_sigma`)
  - Enforce guards: keep bonded pairs within [min,max] factors; ensure non‑bonded distances above `nonbond_min_factor`
  - If a guard fails, retry up to `max_retries_per_frame` times with halved ranges/noise each attempt
  - Wrap to cell for PBC; otherwise center to a cubic box (`nonpbc_box_size`)

**Tagging**:
```python
structure.info["Config_type"] += f" TorsionGuard(n={perturb_per_frame}, sigma={gaussian_sigma}, pbc={pbc_mode})"
```

**Caveats**:
- Ensure a valid lattice for PBC frames; otherwise prefer `auto` or `no`
- Large angle ranges with high `max_torsions_per_conf` can raise rejection rate; consider reducing ranges or enabling local mode
- `bond_keep_max_factor=None` uses `bond_detect_factor` as the upper bound
- Very large molecules: increase `local_mode_cutoff_atoms` and adjust `local_torsion_max_subtree`

**Best Practices**:
- Start with `torsion_range_deg = [-60, 60]` and `gaussian_sigma = 0.02–0.05 Å`
- Enable local mode on big organics for faster, safer rotations
- Combine with FPS/filters downstream to curate diverse yet non‑redundant conformers

**Structure Tagging**:
```python
structure.info["Config_type"] += f" Doping(num={dopant_count})"
```
When using grouping (`group`), you must use files in XYZ format and specify the `group` column. For example:

```text
5
Lattice="6.383697472927415 0.0 0.0 0.0 6.383697472927415 1.4e-15 0.0 8e-16 6.383697472927415" Properties=species:S:1:pos:R:3:group:S:1 pbc="T T T"
Cs       0.00000000       0.00000000       0.00000000 a
I        3.19184873       3.19184873      -0.00000000 b
I        3.19184873      -0.00000000       3.19184873 c
I       -0.00000000       3.19184873       3.19184873 c
Pb       3.19184873       3.19184873       3.19184873 d
```

#### Rule Schema
- Rules is a list; each item is a JSON object with keys:
  - `target` (string): Element symbol to be replaced (e.g., "Si").
  - `dopants` (object): Mapping of element symbol → ratio. Ratios are normalized internally (e.g., "Ge:0.7,C:0.3").
  - `use` (string): One of `"concentration"` or `"count"`.
  - `concentration` ([min, max], floats in 0–1): Fraction of eligible sites to replace.
  - `count` ([min, max], integers): Number of sites to replace.
  - `group` (array, optional): Limit replacement to atoms whose `group` value is in this list  (e.g., "a,c").

If `use` is omitted or unrecognized, all eligible sites are replaced.



#### How Ratios and Exact Work
- `Random`: for each selected site, samples a dopant species according to normalized `dopants` ratios.
- `Exact`: computes integer counts as `floor(ratio * N)` for each species, assigns leftovers to the largest‑ratio species, shuffles order for randomness.

#### Edge Cases and Validation
- If fewer eligible `target` sites exist than requested, the algorithm clamps to the available count.
- `group` matching is exact (case‑sensitive) against the structure’s `group` array values.
- Combining multiple rules applies them sequentially per structure; later rules see results of earlier substitutions.

#### Tips
- Use `Exact` for reproducible stoichiometry, `Random` for data augmentation.
- Keep concentration windows narrow for incremental variations (e.g., 2–10%).
- Prefer `group` when only specific sublattices or layers should be doped.

#### Algorithm
- For each rule, find candidate sites (optionally limited by `group`).
- If `use == concentration`, sample a fraction of candidates; if `use == count`, sample an integer range.
- Choose dopant species by probability (`Random`) or exact counts (`Exact`) and substitute symbols.

#### Caveats
- If candidates are fewer than requested, the algorithm clamps to available sites.
- Doping can break charge balance; validate for your physics/chemistry constraints.

#### Best Practices
- Prefer `Exact` to reproduce specific stoichiometries; use `Random` for augmentation.
- Use `group` to limit substitutions to sublattices or layers.


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

#### Rule Schema
- Rules is a list; each item is a JSON object with keys:
  - `element` (string): Element symbol to delete.
  - `count` ([min, max], integers): Number of atoms to remove per rule application.
  - `group` (array, optional): Restrict deletions to atoms with `group` in the list.

If `element` is missing or `count.max <= 0`, the rule is skipped.


#### Edge Cases and Validation
- Requested deletions are clamped to available eligible atoms.
- Multiple rules are applied sequentially; later rules operate on the already‑modified structure.

#### Tips
- Combine with Super Cell to maintain sufficient system size after deletions.
- For surface models, use `group` to target only top layers or named regions.

#### Algorithm
- For each rule, determines deletions by element and optional `group`, draws a random integer count in range, and deletes unique indices.

#### Caveats
- Excessive deletion can collapse periodic images; keep counts modest.

#### Best Practices
- Use alongside Super Cell to maintain adequate atoms after deletion.

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

#### Algorithm
- Enumerates Miller indices (h,k,l), layer counts, and vacuum values; for each combination builds an ASE slab `surface(...)`, wraps, and annotates.

#### Caveats
- (0,0,0) is skipped; degenerate or redundant orientations can arise for symmetric lattices.
- Large enumerations can explode combinatorially; trim ranges sensibly.

#### Best Practices
- Start with a few low‑index planes and small layer/vacuum ranges.
- Post‑filter by thickness, surface area, or descriptor distance.

### 2.9 Shear Angle Strain
Adjusts cell angles (alpha, beta, gamma) over specified ranges.

**Parameters**: Alpha/Beta/Gamma ranges (degrees), Identify organic.

#### Algorithm
- Convert cell to `[a,b,c,alpha,beta,gamma]`, add angle deltas, rebuild the lattice, and rescale atoms.

#### Caveats
- Large angle changes may yield ill‑conditioned cells; keep steps small.

#### Best Practices
- Combine with small lattice scaling to explore local angle neighborhoods.

### 2.10 Shear Matrix Strain
Applies a shear matrix to the lattice; optionally symmetric.

**Parameters**: XY/YZ/XZ strain percentages, Identify organic, Symmetric shear.

#### Algorithm
- Build shear matrix with off‑diagonal terms from percentages; if symmetric, also fill transposed entries; multiply with cell and rescale atoms.

#### Caveats
- Large shear may produce non‑physical cells; maintain moderate ranges (±5%).

#### Best Practices
- Use symmetric shear for balanced distortions unless specific directionality is desired.

### 2.11 Stacking Fault Generation
Generates stacking faults (or twins) along a specified Miller plane.

**Parameters**: (h,k,l), Step range (start, end, step), Layers.

#### Algorithm
- Compute plane normal from reciprocal lattice, select a layer split along a perpendicular direction, translate the upper part by multiples in the normal direction, wrap, and annotate.

#### Caveats
- If the normal becomes ill‑defined, the card skips modification.

#### Best Practices
- Use small step sizes and validate resulting interlayer distances.

### 2.12 Interstitial & Adsorbate Insertion
Introduces interstitial atoms inside the cell or adsorbates above a surface.

**Parameters**: Mode (Interstitial/Adsorption), Species list (optional weights), Atoms per structure, Minimum distance, Surface axis (adsorption), Offset distance (adsorption).

#### Algorithm
- Interstitial mode samples random fractional coordinates, converts to Cartesian positions, and enforces a minimum-distance constraint via the minimum image convention.
- Adsorption mode locates the topmost fractional coordinate along the selected axis, samples in-plane coordinates randomly, offsets the point outward by the requested distance, and validates separation before insertion.
- Species are drawn according to user-supplied weights (default uniform) and appended to the structure.

#### Caveats
- Offsets must remain within the available vacuum thickness; exceeding the cell length along the chosen axis will wrap under periodic boundary conditions.
- Dense host lattices may require either lower distance thresholds or higher trial counts to find feasible sites.
- Newly inserted atoms inherit zero-valued per-atom arrays (forces, magmoms, etc.); regenerate properties before training.

#### Best Practices
- Run multiple configurations with different random seeds to diversify interstitial placements.
- Combine with vacancy or stacking-fault cards inside a `CardGroup` to cover complementary defect families.
- Apply FPS or energy-based screening afterwards to curate physically relevant insertions.

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

#### Algorithm
- Runs NEP to compute structure descriptors; uses FPS to select up to `Max selected` with a minimum pairwise distance `Min distance` in descriptor space.

#### Caveats
- Descriptor generation requires a valid NEP file; large datasets can take time—progress is reported.

#### Best Practices
- Use a modest `Min distance` first (e.g., 1e-3–1e-2) and tune up.
- Cascade after generation cards to reduce redundancy and export only representative samples.

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

#### Caveats
- Group outputs can be large; consider an in‑group filter to control size.

#### Best Practices
- Use Card Group for parallel variants (e.g., different perturb cards) and a single filter at the bottom to unify selection criteria.

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
from NepTrainKit.core import CardManager
from NepTrainKit.custom_widget.card_widget import MakeDataCard
@CardManager.register_card
class CustomCard(MakeDataCard):
    # Required class attributes
    group = "Custom" # menu name
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
https://github.com/aboys-cb/NepTrainKit/tree/dev/src/NepTrainKit/ui/views/_card
