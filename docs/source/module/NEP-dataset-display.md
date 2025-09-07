# NEP Dataset Display



## 1.User Interface

As shown in the figure, the overall user interface of the software primarily includes the toolbar, result visualization area, structure display area, information display area, and working path display area.

![interface](../_static/image/interface.png)

## 2.Data transmission area

### Data Import

Users can import files in the following two ways:

- Click the import button located at the top left of the menu <img src="../_static/image/open.svg" alt="open" width='20' height='20' /> to import the file path.
- Drag and drop the file directly into the software interface for import.
:::{important}
software will automatically detect the `nep.txt` file type and import it, including the `norm`、`dipole` and `polarizability`.
Currently, the software supports:
- `train.xyz` and matching `*.out` files
- `nep.txt` (optional; falls back to NEP89 if absent) + `train.xyz`
- DeepMD training directory (auto‑detected)
:::

 
### Data Export

After completing the operation, the user can click the save button <img alt="save" height="20" src="../_static/image/save.svg" width="20"/>
to export the results as two files:

- **export_remove_model.xyz**: Contains information about the deleted structures.
- **export_good_model.xyz**: Contains information about the remaining structures.
In the export menu, you can click "Export Selected Structures" to export the currently selected structures.
## 3.Toolbar

In the drawing toolbar, we have integrated function buttons such as reset, zoom, selection and editing, undo, and delete. Users can perform basic operations on the image using these tools. Please note that these operation tools are only effective on the main chart.

<img src="../_static/image/toolbar.png" alt="toolbar" width='600' height='70' />

<img src="../_static/image/init.svg" alt="init" width='30' height='30'  /> **Reset Tool:** Restores the image to its initial state, clearing all modifications made.

<img src="../_static/image/pan.svg" alt="pan" width='30' height='30' /> **Zoom Tool:** Allows users to drag the image's axes or zoom in/out to adjust the view position.

<img src="../_static/image/find_max.svg" alt="find_max" width='30' height='30' /> **Error Max Point Selection Tool:** Automatically identifies a specified number of points with the largest errors, making it easier for users to process them.

<img src="../_static/image/sparse.svg" alt="sparse" width='30' height='30' /> **Farthest Point Sampling Tool:** Users can set the maximum number of training samples and the minimum sampling distance to filter structures.

<img src="../_static/image/pen.svg" alt="pen" width='30' height='30' /> **Selection and Editing Tool:** Use the left mouse button to draw a selection box or directly select a structure; right-click to deselect.

<img src="../_static/image/discovery.svg" alt="discovery" width='30' height='30' /> **Find non-physical structures Tool:** This tool can automatically identify non-physical structures based on the bond length threshold.

<img src="../_static/image/revoke.svg" alt="revoke" width='30' height='30' /> **Undo Tool:** If a structure is accidentally deleted or modified, the undo function can restore the previous state. It supports multiple consecutive undo actions.

<img src="../_static/image/delete.svg" alt="delete" width='30' height='30' /> **Delete Tool:** Removes the structure corresponding to the selected point.

<img src="../_static/image/export.svg" alt="export" width='30' height='30' /> **Export structure descriptor:** Exports the selected structure descriptor to a file.


<img alt="search" height="30" src="../_static/image/search.svg" width="30"/> **Search Tool:** Allows you to select a single **Config_type** and search for specific prefixes, suffixes, or strings.


:::{important}
After clicking the search button, the relevant structures will turn green to indicate the search results. However, at this point, the structures will not be selected. You will need to perform additional actions to complete the selection.
:::

<img alt="select" height="30" src="../_static/image/check.svg" width="30"/> **Select Button:** Selects the structures found by the search.

<img alt="cancel" height="30" src="../_static/image/uncheck.svg" width="30"/> **Deselect Button:** Deselects the selected structures.

:::{important}
If there are no **Config_type** available, clicking the **Select** button will select all visible items, and clicking the **Deselect** button will deselect all items.
:::
\n+### Search by formula

- Toggle the small "formula" checkbox next to the search box to switch between searching by `Config_type` (default) and searching by chemical formula.
- When formula mode is enabled, autocompletion updates to your available formulas.
## 4.Result Visualization and Structure Display

- The result visualization area consists of five subplots, displaying the descriptors, energy, force, pressure, and potential energy information of the dataset. We use the **pyqtgraph** library to encapsulate the plotting functions, and all five subplots support switching to the main plot by double-clicking.
- By clicking on a data point in the main plot, the corresponding crystal structure will be displayed in the right-side display area. The atom sizes and colors in the crystal structure are set based on the atomic radius and the CPK color scheme, respectively.

<img src="../_static/image/visualization.png" alt="visualization" width='810' height='450' />

**Plotting Details:** During the plotting process, **energy**, **force**, **pressure**, and **potential energy** data are all read from the NEP output files in the working path. For **descriptor projection**, we use NEP_CPU to obtain the descriptor of each atom and compute its average as the structure descriptor. Then, principal component analysis (PCA) is used to project the structure descriptors into a two-dimensional space for easier visualization.

### Vector field overlay

- If your structure data contains per-atom vector properties (e.g., dipoles), you can overlay arrows in the structure view.
- Click the arrow button in the structure toolbar, choose a vector property, colormap, and scale, and apply to show/hide the overlay.

## 5.Information Display Area

- In the information display area on the right, the system will show detailed information about the selected **xyz file**. By default, when you click on any data point in the subplots, the display area on the right will synchronize and show the detailed information of the selected structure.
- Below the display area, the **frame number** of the current structure in the original file will be shown. Users can adjust the frame number to view the corresponding structure and its detailed information.

<img src="../_static/image/information.png" alt="information" width='650' height='190' />

### Auto playback

- Use the play/pause button near the index to automatically step through currently visible/selected structures.
- Playback respects your current selection and stops at the end of the range.
