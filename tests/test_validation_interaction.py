"""Regression coverage for explicit scopes, immutable versions and truthful feedback."""
from unittest.mock import patch

from PySide6.QtTest import QTest
from test_validation_ui import make_service, scored, _wait_until
from test_validation_framework import MODEL_HEADER, _import, eos_frames, efv_frames
from test_validation_toolbar import toolbar
from NepTrainKit.core.dataset.services import ModelService
from NepTrainKit.ui.views.model_library import ModelLibraryWidget


def test_changed_model_bytes_create_version_and_preserve_baseline(scored, tmp_path):
    service = scored.service
    original = service.get_model(scored.old['id'])
    old_bytes = service.model_file(original['id']).read_bytes()
    old_run = service.latest_run(original['id'], scored.eos['id'])
    replacement = tmp_path / 'replacement.txt'
    replacement.write_text(MODEL_HEADER + 'replacement')
    saved = ModelService(service.db).modify_model(original['id'], model_path=str(replacement), tags=['new-tag'])
    assert saved.id != original['id'] and saved.parent_id == original['id']
    assert service.model_file(saved.id).read_bytes() == replacement.read_bytes()
    assert service.model_file(original['id']).read_bytes() == old_bytes
    assert service.get_model(original['id'])['model_path'] == original['model_path']
    assert service.get_suite(scored.suite['id'])['baseline_model_id'] == original['id']
    assert service.latest_run(original['id'], scored.eos['id'])['id'] == old_run['id']
    assert service.latest_run(saved.id, scored.eos['id']) is None
    assert saved.energy is None and not saved.starred


def test_identical_relocation_preserves_version_and_invalid_path_is_atomic(scored, tmp_path):
    import pytest
    service = scored.service
    old = scored.old
    copied = tmp_path / 'moved.txt'
    copied.write_bytes(service.model_file(old['id']).read_bytes())
    api = ModelService(service.db)
    saved = api.modify_model(old['id'], model_path=str(copied), name='moved')
    assert saved.id == old['id'] and saved.content_hash == old['content_hash']
    assert saved.source_path == str(copied.resolve())
    with pytest.raises(OSError):
        api.modify_model(old['id'], model_path=str(tmp_path/'missing'), name='must not change')
    assert service.get_model(old['id'])['name'] == 'moved'
    copied.write_text(MODEL_HEADER + 'overwritten in place')
    updated = api.modify_model(old['id'], model_path=str(copied))
    assert updated.id != old['id']  # Same path does not imply same model.


def test_star_preserves_plot_identity_focus_zoom_and_hidden_series(scored):
    w = ModelLibraryWidget(service=scored.service)
    w.open_scorecard(scored.suite['id'], scored.new['id'])
    p = w.detail.eos_plots[0]
    p.setRange(xRange=(33, 34), yRange=(.1, .2), padding=0)
    before = p.viewRange()
    selection = w.table.selectionModel().selectedRows()
    with patch.object(w, 'refresh') as refresh, patch.object(w, '_update_detail') as render:
        w.toggle_star()
        refresh.assert_not_called(); render.assert_not_called()
    assert w.detail.eos_plots[0] is p and p.viewRange() == before
    assert w.table.selectionModel().selectedRows() == selection
    assert w.star_action.isChecked()
    w.close()


def test_scope_matches_visible_counts_and_each_run_action(scored, tmp_path):
    s = scored.service
    another = _import(s, tmp_path, eos_frames(curvature=.9), 'second EOS')
    suite = s.create_suite('mixed', [scored.eos['id'], another['id'], scored.efv['id']])
    w = ModelLibraryWidget(service=s)
    w.open_scorecard(suite['id'], scored.new['id'])
    w.comparison_picker.set_selected_ids([scored.old['id']])
    w.table.selectRow(0)
    # Zoom/focus is a view operation; default scope explicitly says all EOS groups.
    ids = w.selected_model_ids()
    assert len(w.run_jobs(model_ids=ids)) == 4
    assert '2 个势 · 2 项 · 4 次验证' == w.scope_hint.text()
    w.scope_combo.setCurrentIndex(w.scope_combo.findData('selection'))
    assert len(w.run_jobs(model_ids=ids)) == 2
    with patch.object(s, 'evaluate', wraps=s.evaluate) as evaluate:
        w.run_selected()
        assert _wait_until(lambda: w._run_task is None)
        assert {c.args for c in evaluate.call_args_list} == {(i, scored.eos['id']) for i in ids}
    w.scope_combo.setCurrentIndex(w.scope_combo.findData('suite'))
    assert len(w.run_jobs(model_ids=ids)) == 6
    assert len(w.run_jobs(scored.new['id'])) == 3
    assert len(w.run_jobs()) == 9
    w.close()


def test_all_failed_reports_failures_and_retry_uses_original_jobs(scored):
    s = scored.service
    good_factory = s.predictor_factory
    s.predictor_factory = lambda *_: (_ for _ in ()).throw(RuntimeError('inference failed'))
    w = ModelLibraryWidget(service=s)
    w.open_scorecard(scored.suite['id'], scored.fresh['id'])
    w.scope_combo.setCurrentIndex(w.scope_combo.findData('suite'))
    task = w.run_current()
    assert task.tip is None  # No generic green "Success" on worker completion.
    assert not w.import_action.isEnabled() and not w.refresh_button.isEnabled()
    assert not w.model_combo.isEnabled() and not w.scope_combo.isEnabled()
    assert _wait_until(lambda: w._run_task is None)
    assert '计算成功 0 / 失败 2 / 不支持 0' in w.run_summary.text()
    assert 'inference failed' in w.run_summary.text()
    assert len(w._failed_jobs) == 2 and not w.retry_button.isHidden()
    # Changing the view must not retarget a retry to other models/experiments.
    w.model_combo.setCurrentIndex(w.model_combo.findData(scored.old['id']))
    w.scope_combo.setCurrentIndex(w.scope_combo.findData('property'))
    s.predictor_factory = good_factory
    with patch.object(s, 'evaluate', wraps=s.evaluate) as evaluate:
        w.retry_failed()
        assert _wait_until(lambda: w._run_task is None)
        assert {c.args for c in evaluate.call_args_list} == {(scored.fresh['id'], b['id']) for b in (scored.eos, scored.efv)}
    assert '计算成功 2 / 失败 0' in w.run_summary.text() and w.retry_button.isHidden()
    w.close()


def test_top_save_targets_editor_then_temporary_plan_then_model(toolbar, monkeypatch):
    from ase.build import bulk
    host, w = toolbar.host, toolbar.library
    calls = []
    monkeypatch.setattr(w, 'save_current', lambda: calls.append('model'))
    w.show_diagnostic_editor()
    assert not w.save_action.isEnabled() and not w.import_action.isEnabled()
    assert not host.save_dir_button.button.isEnabled()
    w.set_diagnostic_seeds([bulk('PbTe', 'rocksalt', a=6.46)])
    w.diagnostic_editor.name.setText('toolbar plan')
    assert host.save_dir_button.button.isEnabled()
    assert '诊断方案' in host.save_dir_button.toolTip()
    host.save_dir_button.button.click()
    assert w.editor_scroll.isHidden() and w.current_suite_id() > 0
    assert calls == []
    assert '势版本' in host.save_dir_button.toolTip()
    w.show_diagnostic_editor()
    w.diagnostic_editor.name.setText('trial plan')
    w.diagnostic_editor.temporary_button.click()
    assert w.current_suite_id() < 0 and '临时诊断方案' in host.save_dir_button.toolTip()
    from NepTrainKit.ui.widgets import GetStrMessageBox
    monkeypatch.setattr(GetStrMessageBox, 'exec', lambda *_: 1)
    host.save_dir_button.button.click()
    assert w.current_suite_id() > 0 and calls == []


def test_overview_no_result_reads_and_property_selector_is_available(scored):
    w = ModelLibraryWidget(service=scored.service)
    with patch.object(scored.service, 'run_details') as details, patch.object(scored.service, 'evaluate') as evaluate:
        w.refresh(scored.suite['id'], scored.new['id'])
        assert '已有结果' in w.overview_label.text()
        assert w.property_combo.parent() is w.controls_panel
        w.property_combo.setCurrentIndex(w.property_combo.findData('efv'))
        assert len(w.run_jobs(scored.new['id'])) == 1
        details.assert_not_called(); evaluate.assert_not_called()
    w.close()


def test_failed_unsupported_and_success_are_reported_separately(scored):
    s = scored.service
    w = ModelLibraryWidget(service=s)
    w.open_scorecard(scored.suite['id'], scored.fresh['id'])
    w.comparison_picker.set_selected_ids([scored.old['id'], scored.new['id']])
    def result(model_id, set_id):
        status = {scored.old['id']: 'succeeded', scored.new['id']: 'unsupported', scored.fresh['id']: 'failed'}[model_id]
        return {'status': status, 'error': 'unsupported test' if status == 'unsupported' else 'failure test'}
    with patch.object(s, 'evaluate', side_effect=result):
        w.run_selected()
        assert _wait_until(lambda: w._run_task is None)
    assert '计算成功 1 / 失败 1 / 不支持 1' in w.run_summary.text()
    assert w._failed_jobs == [(scored.fresh['id'], scored.eos['id'])]
    w.close()


def test_version_edit_selects_new_child_and_updates_project_count(toolbar, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from NepTrainKit.ui.views import dataset_widget as module
    from test_validation_toolbar import _project_index
    page = toolbar.page
    project_id = toolbar.scored.old['project_id']
    page.show_projects()
    tree, models = page.project_widget, page.data_item_widget
    idx = _project_index(tree, project_id)
    tree._view.setCurrentIndex(idx); tree.item_clicked(idx)
    from PySide6.QtCore import Qt
    matches = models._model.match(models._model.index(0, 0), Qt.DisplayRole, toolbar.scored.old['id'], 1, Qt.MatchExactly | Qt.MatchRecursive)
    models._view.setCurrentIndex(matches[0])
    before = tree.project_item_dict[project_id].model_num
    path = tmp_path / 'new-version.txt'; path.write_text(MODEL_HEADER + 'changed')
    box = MagicMock(); box.exec_.return_value = 1
    box.get_dict.return_value = dict(name='retrained', model_type='NEP', model_path=str(path), tags=[])
    monkeypatch.setattr(module, 'ModelInfoMessageBox', lambda *_: box)
    models.create_model(modify=True)
    selected = models._view.currentIndex()
    new_id = selected.data()
    assert new_id != toolbar.scored.old['id'] and selected.parent().isValid()
    assert models._view.isExpanded(selected.parent())
    assert tree.project_item_dict[project_id].model_num == before + 1
    assert page.library_widget.model_combo.findData(new_id) >= 0
