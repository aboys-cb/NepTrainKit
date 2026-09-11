"""Exercise scanner failure modes with small sources, independent of the GUI."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load('audit_translations', ROOT / 'tools/audit_translations.py')
update = _load('update_translations_for_audit', ROOT / 'tools/update_translations.py')


def _scan(tmp_path, source, catalog=None, translate=lambda value: value):
    path = tmp_path / 'example.py'
    path.write_text(source, encoding='utf-8')
    return audit.audit_source(path, catalog or {}, translate)


@pytest.mark.parametrize('expression', [
    'f"Invalid {name}"', '"Invalid {}".format(name)', '"Invalid %s" % name',
])
def test_interpolation_before_translation_fails(tmp_path, expression):
    result = _scan(tmp_path, f'self.tr({expression})')
    assert [item['rule'] for item in result] == ['interpolated_before_translation']
    assert result[0]['severity'] == 'error'


def test_translation_before_interpolation_preserves_template(tmp_path):
    assert _scan(tmp_path, 'self.tr("Invalid {name}").format(name=name)',
                 {('Widget', 'Invalid {name}'): '名称 {name} 无效'}) == []


def test_structured_error_checks_template_instead_of_code(tmp_path):
    result = _scan(tmp_path, 'raise CardOperationError("code", "Invalid {name}", name=name)')
    assert len(result) == 1
    assert result[0]['source'] == 'Invalid {name}'
    assert result[0]['context'] == 'CardOperationError'
    assert result[0]['rule'] == 'missing_catalog_entry'


def test_explicit_context_cannot_use_another_contexts_entry(tmp_path):
    result = _scan(tmp_path, 'QCoreApplication.translate("One", "Count")', {('Two', 'Count'): '数量'})
    assert result[0]['rule'] == 'missing_catalog_entry'


def test_dynamic_error_is_a_candidate_and_never_executes(tmp_path):
    result = _scan(tmp_path, 'raise ValueError(f"Invalid name {dangerous()}")',
                  translate=lambda text: text.replace('Invalid', '无效'))
    assert result[0]['source'] == 'Invalid name ⟦value⟧'
    assert result[0]['rule'] == 'mixed_runtime_message'
    assert result[0]['severity'] == 'candidate'


def test_dynamic_values_and_technical_literals_are_not_english_prose(tmp_path):
    assert _scan(tmp_path, 'label.setText(f"{a} / {b}")\nlabel.setText("Cu")') == []
    assert _scan(tmp_path, 'label.setText("Count")')[0]['rule'] == 'ui_literal'


def test_catalog_checks_status_and_all_placeholder_forms(tmp_path):
    path = tmp_path / 'catalog.ts'
    path.write_text('''<TS><context><name>Widget</name>
      <message><source>A</source><translation type="unfinished">甲</translation></message>
      <message><source>B {value:.2f}</source><translation>乙 {value}</translation></message>
      <message><source>C %1</source><translation>丙 %2</translation></message>
      <message><source>D {}</source><translation>丁</translation></message>
      <message><source>E</source><translation type="vanished" /></message>
      <message><source>F</source><translation>己</translation></message>
    </context></TS>''', encoding='utf-8')
    catalog, issues = audit.audit_catalog(path)
    assert len(catalog) == 5
    assert [item['rule'] for item in issues] == ['unfinished_translation'] + ['placeholder_mismatch'] * 3
    assert audit.format_fields('{{literal}} {value} %1') == audit.format_fields('%1 {value} {{literal}}')


def test_extractor_covers_structured_errors_and_helpers_inside_fstrings(tmp_path, monkeypatch):
    path = tmp_path / 'example.py'
    path.write_text('''def _tr(text):
    return QCoreApplication.translate("Example", text)
value = f"<b>{_tr('Docs')}</b>"
raise CardOperationError("code", "Invalid {name}", name="a")
''', encoding='utf-8')
    monkeypatch.setattr(update, 'SRC', tmp_path)
    root = ET.fromstring('<TS />')
    update._sync_python_templates(root)
    first = ET.tostring(root)
    update._sync_python_templates(root)
    assert ET.tostring(root) == first
    found = {(c.findtext('name'), m.findtext('source')) for c in root.findall('context') for m in c.findall('message')}
    assert found == {('Example', 'Docs'), ('CardOperationError', 'Invalid {name}')}


def test_catalog_with_partial_translation_remains_a_review_candidate(tmp_path):
    path = tmp_path / 'catalog.ts'
    path.write_text('''<TS><context><name>Widget</name>
      <message><source>Invalid name</source><translation>无效 name</translation></message>
    </context></TS>''', encoding='utf-8')
    _, issues = audit.audit_catalog(path)
    assert issues[0]['rule'] == 'catalog_english_residue'
    assert issues[0]['severity'] == 'candidate'


def test_extractor_does_not_revive_an_obsolete_duplicate(tmp_path, monkeypatch):
    (tmp_path / 'example.py').write_text('raise CardOperationError("code", "Invalid")', encoding='utf-8')
    monkeypatch.setattr(update, 'SRC', tmp_path)
    root = ET.fromstring('''<TS><context><name>CardOperationError</name>
      <message><source>Invalid</source><translation type="vanished">旧译文</translation></message>
      <message><source>Invalid</source><translation>无效</translation></message>
    </context></TS>''')
    update._sync_python_templates(root)
    update._prune_obsolete_duplicates(root)
    messages = root.findall('.//message')
    assert len(messages) == 1
    assert messages[0].findtext('translation') == '无效'


def test_catalog_placeholder_names_are_not_english_residue(tmp_path):
    path = tmp_path / 'catalog.ts'
    path.write_text('''<TS><context><name>Widget</name>
      <message><source>Missing {labels}: {expected}</source>
        <translation>缺少 {labels}：{expected}</translation></message>
    </context></TS>''', encoding='utf-8')
    assert audit.audit_catalog(path)[1] == []


def test_extractor_recovers_translation_lupdate_marked_obsolete(tmp_path, monkeypatch):
    (tmp_path / 'example.py').write_text('''class Example:
    def render(self):
        return f"{self.tr('moment')}"
''', encoding='utf-8')
    monkeypatch.setattr(update, 'SRC', tmp_path)
    root = ET.fromstring('''<TS><context><name>Example</name>
      <message><source>moment</source><translation type="vanished">磁矩</translation></message>
      <message><source>moment</source><translation type="unfinished" /></message>
    </context></TS>''')
    update._sync_python_templates(root)
    update._prune_obsolete_duplicates(root)
    update._finalize_populated_translations(root)
    messages = root.findall('.//message')
    assert len(messages) == 1
    assert messages[0].findtext('translation') == '磁矩'
    assert messages[0].find('translation').get('type') is None


def test_review_requires_both_source_and_rendered_text(tmp_path):
    import json
    (tmp_path / 'tools').mkdir()
    entry = dict(path='src/a.py', source='field', context=None, rendered='字段 field',
                 review='technical', reason='Field identifier')
    (tmp_path / 'tools/translation_review.json').write_text(
        json.dumps({'retained_candidates': [entry]}))
    original = dict(severity='candidate', path='src/a.py', source='field', rendered='字段 field')
    changed_source = dict(original, source='new field')
    changed_translation = dict(original, rendered='字段 must field')
    hard_error = dict(original, severity='error')
    audit.annotate_review(tmp_path, [original, changed_source, changed_translation, hard_error])
    assert original['review'] == 'technical'
    assert 'review' not in changed_source
    assert 'review' not in changed_translation
    assert 'review' not in hard_error


def test_runtime_registry_templates_survive_lupdate_obsoletion(tmp_path, monkeypatch):
    (tmp_path / 'ui').mkdir()
    (tmp_path / 'ui/runtime_error_catalog.py').write_text(
        'RUNTIME_ERROR_TEMPLATES = (("RuntimeMessage", "Invalid {name}"),)')
    monkeypatch.setattr(update, 'SRC', tmp_path)
    root = ET.fromstring('''<TS><context><name>RuntimeMessage</name><message>
      <source>Invalid {name}</source><translation type="vanished">名称 {name} 无效</translation>
    </message></context></TS>''')
    update._sync_runtime_error_templates(root)
    update._finalize_populated_translations(root)
    translation = root.find('.//translation')
    assert translation.text == '名称 {name} 无效'
    assert translation.get('type') is None
