from NepTrainKit.core.structure import Structure,save_npy_structure



structures = Structure.read_multiple(r"D:\Master\dp\1000-6-4\train.xyz")
#我是按照config_type 来区分文件夹的  所以你这里需要遍历修改下 不然不同原子数的不能再一起
for structure in structures:
    structure.tag=structure.formula
save_npy_structure(r"D:\Desktop\perturb-aimd\dp",structures)

