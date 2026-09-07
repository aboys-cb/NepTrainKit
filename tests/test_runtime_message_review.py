"""Regression coverage for complete messages, runtime sinks and opaque values."""
from __future__ import annotations

from string import Formatter
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from NepTrainKit import i18n
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.ui.messages import translate_runtime_error, translate_runtime_message
from NepTrainKit.ui.runtime_error_catalog import RUNTIME_ERROR_TEMPLATES, DIAGNOSTIC_TEMPLATES


@pytest.fixture
def chinese_app():
    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, 'zh_CN')
    yield app
    i18n.install_translator(app, 'en_US')


def _values(template):
    return {name: ('已知原因' if name in ('error', 'reason', 'value1') and
                   ('Failed to import structures' in template or name != 'value1') else 'TOKEN_' + name)
            for _, name, _, _ in Formatter().parse(template) if name is not None}


@pytest.mark.parametrize('context,template', RUNTIME_ERROR_TEMPLATES)
def test_reviewed_whole_templates_survive_flattening_and_language_switch(chinese_app, context, template):
    values = _values(template)
    raw = template.format(**values)
    translated_template = QCoreApplication.translate(context, template)
    assert translated_template != template
    expected = translated_template.format(**values)
    assert translate_runtime_message(raw) == expected
    i18n.install_translator(chinese_app, 'en_US')
    assert translate_runtime_message(raw) == raw


@pytest.mark.parametrize('template', DIAGNOSTIC_TEMPLATES)
def test_reviewed_diagnostics_keep_raw_text_in_log(chinese_app, template, monkeypatch):
    from NepTrainKit.ui import messages
    logged = []
    monkeypatch.setattr(messages.logger, 'warning', lambda fmt, value: logged.append(value))
    raw = template.format(**_values(template))
    assert translate_runtime_message(raw) == '内部数据校验失败，详细信息已写入日志。'
    assert logged == [raw]
    i18n.install_translator(chinese_app, 'en_US')
    assert translate_runtime_message(raw) == raw


def test_paths_and_user_values_are_opaque(chinese_app):
    path = r'D:\计算文件\invalid empty must be failed\模型 {x}.txt'
    assert translate_runtime_message('NEP file does not exist: ' + path) == 'NEP 文件不存在：' + path
    error = CardOperationError('test', 'NEP file does not exist: {model_path}', model_path=path)
    # A raw FileNotFoundError follows the same complete-message path.
    assert translate_runtime_error(FileNotFoundError(str(error))) == 'NEP 文件不存在：' + path
    assert translate_runtime_message(path) == path


@pytest.mark.parametrize('case,expected', [
    ('percent', '百分比 不能为空。'),
    ('weight-negative', '插隙与表面吸附：元素 Fe 的权重必须是有限正数。'),
    ('weight-invalid', '插隙与表面吸附：元素 Fe 的权重无效。'),
    ('range', '扫描范围必须包含三个值：起点、终点、步长。'),
    ('dz', 'dz 表达式为空。'),
    ('replacement', '替换比例必须是有限的非负数。'),
    ('response', '响应扫描至少需要 1 个不同的坐标。'),
    ('deepmd', '无法导出空的 DeepMD 数据集。'),
    ('direction', '方向向量不能为零。'),
    ('site-rules', 'site_rules 必须是非空 JSON 对象。'),
])
def test_actual_invalid_inputs(chinese_app, tmp_path, case, expected):
    from NepTrainKit.core.cards.alloy import _range_pair, parse_replacements
    from NepTrainKit.core.cards.defect import _parse_insert_species
    from NepTrainKit.core.cards.magnetism import range_values
    from NepTrainKit.core.cards.structure import validate_dz_expr
    from NepTrainKit.core.magnetic_response import _parse_scan
    from NepTrainKit.core.structure import save_npy_structure
    from NepTrainKit.ui.widgets.parameter_inputs import DirectionInput
    from NepTrainKit.ui.widgets.alloy_site_rules import AlloySiteRulesEditor
    calls = {
        'percent': lambda: _range_pair([], label='percent'),
        'weight-negative': lambda: _parse_insert_species('Fe:-1'),
        'weight-invalid': lambda: _parse_insert_species('Fe:abc'),
        'range': lambda: range_values([0, 1]),
        'dz': lambda: validate_dz_expr('', allowed_names=set()),
        'replacement': lambda: parse_replacements('Fe:-1'),
        'response': lambda: _parse_scan(''),
        'deepmd': lambda: save_npy_structure(tmp_path / 'export', []),
        'direction': lambda: DirectionInput._normalized((0, 0, 0)),
        'site-rules': lambda: AlloySiteRulesEditor.validate_rule_mapping([]),
    }
    with pytest.raises(ValueError) as caught:
        calls[case]()
    raw = str(caught.value)
    assert translate_runtime_message(caught.value) == expected
    assert translate_runtime_message(raw) == expected
    i18n.install_translator(chinese_app, 'en_US')
    assert translate_runtime_message(caught.value) == raw


def test_audit_reason_uses_complete_model_error(chinese_app, tmp_path):
    from NepTrainKit.core.audit.nep_cutoff import parse_nep_cutoff
    from NepTrainKit.ui.pages.training_set_audit import TrainingSetAuditWidget
    path = tmp_path / 'nep.txt'
    path.write_text('nep4 1 Fe\ncutoff -1 2 0 0\n')
    with pytest.raises(ValueError) as caught:
        parse_nep_cutoff(path)
    widget = SimpleNamespace(tr=lambda text: QCoreApplication.translate('TrainingSetAuditWidget', text))
    assert TrainingSetAuditWidget._localized_dimension_reason(
        widget, SimpleNamespace(reason=str(caught.value))
    ) == 'NEP 截断半径必须是有限正数。'


def test_update_http_failure(chinese_app, monkeypatch):
    from NepTrainKit.ui.update import fetch_releases
    monkeypatch.setattr('requests.get', lambda *a, **kw: SimpleNamespace(status_code=503, json=lambda: {}))
    with pytest.raises(RuntimeError) as caught:
        fetch_releases()
    assert translate_runtime_message(caught.value) == '检查更新失败，HTTP 状态码为 503'


def test_background_failure_localizes_inner_reason(chinese_app):
    from NepTrainKit.ui.threads import BackgroundTask
    contents = []
    fake = SimpleNamespace(tip=SimpleNamespace(setContent=contents.append, setState=lambda state: None),
                           tr=lambda text: QCoreApplication.translate('BackgroundTask', text))
    BackgroundTask._BackgroundTask__failed_work(fake, 'Cannot export an empty DeepMD dataset.')
    assert contents == ['失败：无法导出空的 DeepMD 数据集。']
