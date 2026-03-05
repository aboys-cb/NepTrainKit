# 自定义卡片开发

## 放置目录

将自定义卡片放在用户配置目录下的 `cards/` 目录。

## 最小模板

- 继承 `MakeDataCard`
- 实现 `init_ui` / `process_structure` / `to_dict` / `from_dict`
- 使用 `@CardManager.register_card` 注册

## 约定

- `process_structure` 返回 `list[ase.Atoms]`
- `to_dict` 持久化所有行为参数
- `from_dict` 完整恢复状态
- 必要时写入 `Config_type` 标签
