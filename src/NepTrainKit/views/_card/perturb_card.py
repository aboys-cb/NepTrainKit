from ..cards import *

class PerturbCard(MakeDataCard):
    card_name= "Atomic Perturb"
    menu_icon=r":/images/src/images/perturb.svg"
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Atomic Perturb")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("perturb_card_widget")
        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(True)
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))



        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("Å",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_radio_label=BodyLabel("Max distance:",self.setting_widget)
        self.scaling_condition_frame.set_input_value([0.3])
        self.scaling_radio_label.setToolTip("Maximum displacement distance")
        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_condition_frame.set_input_value([50])

        self.num_label=BodyLabel("Max num:",self.setting_widget)
        self.num_label.setToolTip("Number of structures to generate")

        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame,1, 1, 1, 2)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)

        self.settingLayout.addWidget(self.num_label,3, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame,3, 1, 1,2)

    def process_structure(self, structure):
        structure_list=[]
        engine_type=self.engine_type_combo.currentIndex()
        max_scaling=self.scaling_condition_frame.get_input_value()[0]
        max_num=self.num_condition_frame.get_input_value()[0]
        identify_organic=self.organic_checkbox.isChecked()
        n_atoms = len(structure)
        dim = n_atoms * 3  # 每个原子有 x, y, z 三个维度

        if engine_type == 0:

            sobol_engine = Sobol(d=dim, scramble=True)
            sobol_seq = sobol_engine.random(max_num)  # 生成 [0, 1] 的序列
            perturbation_factors = (sobol_seq - 0.5) * 2  # 转换为 [-1, 1]
        else:
            # 生成均匀分布的扰动因子，范围 [-1, 1]
            perturbation_factors = np.random.uniform(-1, 1, (max_num, dim))

            # 识别团簇和有机分子
        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        orig_positions = structure.positions
        for i in range(max_num):
            new_structure = structure.copy()

            # 提取当前结构的扰动因子并重塑为 (n_atoms, 3)
            delta = perturbation_factors[i].reshape(n_atoms, 3) * max_scaling
            if identify_organic:
                # 对每个团簇应用微扰
                new_positions = orig_positions.copy()

                for cluster_indices, is_organic in zip(clusters, is_organic_list):
                    if is_organic:
                        # 有机分子：整体平移，应用统一的偏移向量
                        # 从团簇的第一个原子的扰动因子中取偏移向量
                        cluster_delta = delta[cluster_indices[0]]
                        for idx in cluster_indices:
                            new_positions[idx] += cluster_delta
                    else:
                        # 非有机分子：逐原子微扰
                        for idx in cluster_indices:
                            new_positions[idx] += delta[idx]
            else:
                new_positions=orig_positions+delta

            # 更新新结构的坐标
            new_structure.set_positions(new_positions)
            new_structure.info["Config_type"] = new_structure.info.get("Config_type","") + f" Perturb(distance={max_scaling}, {'uniform' if engine_type == 1 else 'Sobol'})"
            structure_list.append(new_structure)

        return structure_list

    def to_dict(self):
        data_dict = super().to_dict()


        data_dict['engine_type'] = self.engine_type_combo.currentIndex()
        data_dict["organic"]=self.organic_checkbox.isChecked()
        data_dict['scaling_condition'] = self.scaling_condition_frame.get_input_value()

        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)

        self.engine_type_combo.setCurrentIndex(data_dict['engine_type'])

        self.scaling_condition_frame.set_input_value(data_dict['scaling_condition'])

        self.num_condition_frame.set_input_value(data_dict['num_condition'])
        self.organic_checkbox.setChecked(data_dict.get("organic", False))
