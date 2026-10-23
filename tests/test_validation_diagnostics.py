"""Independent energy functions exercise derivative signs, units and session persistence."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from sqlalchemy import select, func

from NepTrainKit.core.magnetism import set_initial_magmoms_safe, existing_moment_vectors
from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.core.validation.service import ModelDraft, compare
from NepTrainKit.core.dataset.models import BenchmarkSuite, BenchmarkSet, BenchmarkRun
from NepTrainKit.core.validation.protocols import read_frames
from NepTrainKit.core.validation.protocols.diagnostic import DiagnosticSpec, DiagnosticProtocol, UnsupportedDiagnostic, encode_experiment, experiment, scan_frames


@pytest.fixture
def seed():
    atoms = Atoms('Fe2', positions=[[0,0,0],[1.2,.4,.1]], cell=[5,5,5], pbc=True)
    set_initial_magmoms_safe(atoms, [[.2,.3,2], [.1,.5,1.8]])
    return atoms


class AnalyticPredictor:
    """E = |r1-r0|²/2 + 0.02 V² + sum(|m|⁴)/4 - m0·m1."""
    is_spin_model = True
    backend = 'analytic'
    wrong_force = False
    missing_mforce = False

    def predict(self, frames):
        energy, forces, virials, mforces = [], [], [], []
        for a in frames:
            assert a.calc is None and 'energy' not in a.info and 'forces' not in a.arrays
            r = a.positions[1] - a.positions[0]
            m = existing_moment_vectors(a, lift_scalar=False)
            e = .5 * r @ r + .02 * a.get_volume()**2
            force = np.array([r, -r])
            virial = -np.outer(r, r) - np.eye(3) * .04 * a.get_volume()**2
            e += .25 * np.sum(np.sum(m*m,axis=1)**2) - m[0] @ m[1]
            mf = -m * np.sum(m*m,axis=1)[:,None] + m[::-1]
            energy.append(e)
            forces.extend(force * (-1 if self.wrong_force else 1))
            virials.append((virial / len(a)).reshape(9))
            mforces.extend(mf)
        return SimpleNamespace(energy=np.array(energy), forces=np.array(forces), structure_virials=np.array(virials),
                               mforces=None if self.missing_mforce else np.array(mforces))


def frames(seed, kind, **kwargs):
    spec = DiagnosticSpec(kind=kind,low=.8 if kind=='spin_magnitude' else -.2,
                          high=1.2 if kind=='spin_magnitude' else .2,
                          derivative_step=.01, abs_tolerance=1e-6, rel_tolerance=1e-4,
                          variable_moments=True, **kwargs)
    return read_frames(encode_experiment(seed,spec))


@pytest.mark.parametrize('kind', ['volume','displacement','spin_angle','spin_magnitude'])
def test_total_energy_derivative_and_normalized_response(seed,kind):
    ev=DiagnosticProtocol().evaluate(AnalyticPredictor(),frames(seed,kind))
    check=ev.details['checks'][0]
    assert check['status']=='pass'
    assert check['numeric'][-1] == pytest.approx(check['analytic'], rel=1e-4, abs=1e-6)
    e=np.array(ev.details['energy'])
    np.testing.assert_allclose(ev.details['relative_energy'], (e-e[ev.details['origin_index']])*1000/len(seed))
    assert ev.details['reference'] is None
    assert compare(1,2,'none',.05) is None


def test_wrong_force_and_missing_mforce_are_not_passes(seed):
    p=AnalyticPredictor();p.wrong_force=True
    assert DiagnosticProtocol().evaluate(p,frames(seed,'displacement')).details['checks'][0]['status']=='fail'
    p.missing_mforce=True
    assert DiagnosticProtocol().evaluate(p,frames(seed,'spin_angle')).details['checks'][0]['status']=='not_applicable'


def test_invariance_conditions_are_explicit(seed):
    p=DiagnosticProtocol()
    ev=p.evaluate(AnalyticPredictor(),frames(seed,'spin_angle'))
    assert [c['status'] for c in ev.details['checks'][-2:]]==['not_applicable','not_applicable']
    ev=p.evaluate(AnalyticPredictor(),frames(seed,'spin_angle',zero_field=True,no_soc=True))
    assert all(c['status']=='pass' for c in ev.details['checks'])


def test_unconverged_difference_is_undetermined(seed):
    f=frames(seed,'spin_magnitude')
    a,s=experiment(f)
    s=replace(s,derivative_step=.4,abs_tolerance=1e-9,rel_tolerance=1e-9)
    ev=DiagnosticProtocol().evaluate(AnalyticPredictor(),read_frames(encode_experiment(a,s)))
    assert ev.details['checks'][0]['status']=='undetermined'


def test_unsupported_inputs_and_no_fake_labels(seed):
    seed.calc=SinglePointCalculator(seed,energy=123,forces=np.zeros((2,3)))
    seed.info['energy']=123
    seed.arrays['force']=np.ones((2,3))
    a,s=experiment(frames(seed,'spin_angle'))
    assert a.calc is None and not a.info and 'force' not in a.arrays
    p=AnalyticPredictor();p.is_spin_model=False
    with pytest.raises(UnsupportedDiagnostic):
        DiagnosticProtocol().evaluate(p,frames(seed,'spin_angle'))
    with pytest.raises(ValueError,match='三维磁矩'):
        encode_experiment(Atoms('Fe',cell=[3]*3),s)
    with pytest.raises(ValueError):
        encode_experiment(seed,replace(s,axis=(0,0,0)))
    with pytest.raises(UnsupportedDiagnostic,match='幅值'):
        DiagnosticProtocol().evaluate(AnalyticPredictor(),read_frames(encode_experiment(seed,replace(s,kind='spin_magnitude',low=.8,high=1.2,variable_moments=False))))


def test_same_structures_and_only_selected_spin_changes(seed):
    a,s=experiment(frames(seed,'spin_angle'))
    generated=scan_frames(a,s,[-.1,0,.1])
    for f in generated:
        np.testing.assert_array_equal(f.positions,seed.positions)
        np.testing.assert_array_equal(f.arrays['spin'][1],seed.arrays['spin'][1])
    np.testing.assert_allclose(generated[1].arrays['spin'],seed.arrays['spin'])


@pytest.fixture
def service(tmp_path):
    return ValidationService(Library(tmp_path/'library'),predictor_factory=lambda *_:AnalyticPredictor())


def counts(service):
    with service.db.session() as s:
        return tuple(s.scalar(select(func.count()).select_from(cls)) for cls in (BenchmarkSuite,BenchmarkSet,BenchmarkRun))


def test_temporary_plan_save_roundtrip_and_cache_identity(service,seed):
    spec=DiagnosticSpec(kind='displacement',low=-.1,high=.1,derivative_step=.01)
    suite=service.create_diagnostic_plan('Fe',[(seed,spec)])
    benchmark=service.suite_sets(suite['id'])[0]
    assert counts(service)==(0,0,0)
    model=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),temporary=True)
    run=service.evaluate(model['id'],benchmark['id'])
    assert run['status']=='succeeded',run
    assert counts(service)==(0,0,0)
    assert service.evaluate(model['id'],benchmark['id'])['id']==run['id']
    changed=service.create_diagnostic_plan('changed',[(seed,replace(spec,high=.2))])
    assert service.suite_sets(changed['id'])[0]['id']!=benchmark['id']
    saved=service.save_diagnostic_plan(suite['id'],'Fe saved')
    assert counts(service)==(1,1,1)
    reopened=ValidationService(service.library)
    saved_set=reopened.suite_sets(saved['id'])[0]
    assert reopened.get_suite(saved['id'])['baseline_model_id'] is None
    assert reopened.run_details(reopened.latest_run(model['id'],saved_set['id']))==service.run_details(run)
    assert experiment(reopened.set_frames(saved_set['id']))[1]==experiment(service.set_frames(benchmark['id']))[1]
    with pytest.raises(ValueError,match='同名'):
        service.save_diagnostic_plan(suite['id'],'Fe saved')
    assert counts(service)==(1,1,1)
    service.backend='other'
    assert service.evaluate(model['id'],benchmark['id'])['id']!=run['id']


def test_unsupported_run_is_reported_without_hiding_other_experiments(service,seed):
    suite=service.create_diagnostic_plan('mixed',[(seed,DiagnosticSpec()),(seed,DiagnosticSpec(kind='spin_magnitude',low=.8,high=1.2))])
    model=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),temporary=True)
    assert [r['status'] for r in service.run_suite(model['id'],suite['id'])]==['succeeded','unsupported']


def test_saved_plan_keeps_explicit_baseline_and_old_plans(service,seed):
    model=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),auto=False)
    old=service.create_suite('old')
    service.set_baseline(old['id'],model['id'],'original')
    suite=service.create_diagnostic_plan('new',[(seed,DiagnosticSpec())])
    service.set_baseline(suite['id'],model['id'],'explicit')
    saved=service.save_diagnostic_plan(suite['id'],'new saved')
    assert saved['baseline_model_id']==model['id'] and saved['baseline_note']=='explicit'
    assert service.get_suite(old['id'])['baseline_note']=='original'
    from NepTrainKit.core.validation.package import build_package
    with pytest.raises(ValueError,match='v1'):
        build_package(service,saved['id'])


def test_workbench_editor_draft_run_overlays_focus_and_save(service,seed,tmp_path):
    from PySide6.QtWidgets import QApplication
    from qfluentwidgets import TransparentToolButton
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget
    from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive
    from test_validation_ui import _wait_until
    keep_created_styles_alive()
    widget=ModelLibraryWidget(service=service)
    widget.resize(1060,820)
    widget.show()
    draft=ModelDraft(b'nep4_spin3 1 Fe\nnew','/test/new.txt')
    widget.open_scorecard(None,draft=draft)
    assert not service.models()
    widget.set_diagnostic_seeds([seed])
    widget.show_diagnostic_editor()
    editor=widget.diagnostic_editor
    assert editor.use_button.isEnabled() and not widget.controls_panel.isEnabled()
    assert widget.controls_panel.isHidden()
    editor.rows['spin_angle'][0].setChecked(True)
    editor.use_button.click()
    assert widget.current_suite_id()<0 and counts(service)==(0,0,0)
    assert not widget.editor_scroll.isVisible() and widget.controls_panel.isEnabled()
    assert not widget.controls_panel.isHidden()
    assert len(widget.detail.eos_plots)==3
    assert not widget.detail.offset_checkbox.isVisible() and None not in widget.detail.legend_buttons
    assert widget.save_plan_action.isEnabled() and not widget.export_package_action.isEnabled()
    task=widget.run_current()
    assert task and _wait_until(lambda:widget._run_task is None)
    assert counts(service)==(0,0,0)
    assert len(service.models())==1 and service.models()[0]['project_id'] is None
    current=widget.current_model_row()['model_id']
    assert len(widget.detail._curves[current])==3
    second=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\nold','/test/old.txt'),auto=False)
    widget.refresh(model_id=current)
    widget.comparison_picker.set_options([(second['id'],'old','not run')],[second['id']])
    widget._on_selection_changed()
    task=widget.run_selected()
    assert task and _wait_until(lambda:widget._run_task is None)
    assert len(widget.detail._curves[second['id']])==3
    widget.detail.legend_buttons[second['id']].click()
    assert not any(c.isVisible() for c in widget.detail._curves[second['id']])
    focus=widget.detail.eos_cards[0].findChild(TransparentToolButton,'toggleDiagnosticFocus')
    focus.click()
    assert len(widget.detail.eos_plots)==1
    plot=widget.detail.eos_plots[0]
    plot.setXRange(100,101)
    widget.detail.eos_cards[0].findChild(TransparentToolButton,'resetDiagnosticView').click()
    assert plot.getViewBox().viewRange()[0][0]<1
    widget.detail.eos_cards[0].findChild(TransparentToolButton,'toggleDiagnosticFocus').click()
    assert len(widget.detail.eos_plots)==3
    saved=service.save_diagnostic_plan(widget.current_suite_id(),'saved')
    widget.refresh(saved['id'],current)
    assert not widget.save_plan_action.isEnabled()
    assert len(widget.detail.eos_plots)==3 and len(widget.detail._curves[current])==3
    widget.close()


def test_editor_replacement_clears_stale_preview(seed,monkeypatch):
    from NepTrainKit.ui.widgets import diagnostic_editor as module
    widget=module.DiagnosticEditor()
    widget.set_seeds([seed])
    assert widget.use_button.isEnabled()
    monkeypatch.setattr(module.QFileDialog,'getOpenFileName',lambda *_:('/missing.xyz',''))
    widget.load_file()
    assert not widget.use_button.isEnabled() and not widget._seeds
    widget.set_seeds([seed])
    widget.rows['spin_angle'][0].setChecked(True)
    widget.axis.setCurrentIndex(2)
    assert widget.use_button.isEnabled()  # tilted, not exactly collinear
    straight=seed.copy();set_initial_magmoms_safe(straight,[[0,0,2],[0,0,2]])
    widget.set_seeds([straight])
    assert not widget.use_button.isEnabled() and '平行' in widget.preview.text()
    widget.close()


def test_resolved_backend_and_runtime_change_invalidate_cache(service,seed):
    suite=service.create_diagnostic_plan('cache',[(seed,DiagnosticSpec())])
    model=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),temporary=True)
    key=service.suite_sets(suite['id'])[0]['id']
    before=service.evaluate(model['id'],key)
    predictor=AnalyticPredictor();predictor.backend='different'
    service.predictor_factory=lambda *_:predictor
    after=service.evaluate(model['id'],key)
    assert before['id']!=after['id']
    assert service.evaluate(model['id'],key)['id']==after['id']


def test_viewer_passes_selected_structures_without_copying_labels_into_plan(seed):
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget
    captured=[]
    stub=SimpleNamespace(refresh_validation_chip=lambda:None,_validation_status=None,
        _validation_model_id=None,_validation_draft=ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),
        _parent=SimpleNamespace(open_model_scorecard=lambda *args,**kwargs:captured.append((args,kwargs))),
        nep_result_data=SimpleNamespace(load_flag=True,select_index={2,1},get_atoms=lambda i:seed),
        struct_index_spinbox=SimpleNamespace(value=lambda:0))
    ShowNepWidget.open_validation_scorecard(stub)
    assert len(captured[0][1]['seeds'])==2 and captured[0][1]['seeds'][0] is not seed
    assert captured[0][1]['draft'] is stub._validation_draft


def test_edit_saved_experiment_creates_copy_and_preserves_original(service,seed):
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget
    spec=DiagnosticSpec(kind='displacement',low=-.2,high=.2,derivative_step=.01)
    suite=service.create_diagnostic_plan('original',[(seed,spec)])
    saved=service.save_diagnostic_plan(suite['id'],'original')
    original_set=service.suite_sets(saved['id'])[0]
    widget=ModelLibraryWidget(service=service)
    widget.show();widget.open_scorecard(saved['id'])
    widget.edit_diagnostic_plan()
    editor=widget.diagnostic_editor
    assert editor.isVisible() and editor.rows['displacement'][2].value()==.2
    editor.rows['displacement'][2].setValue(.3)
    editor.use_button.click()
    assert widget.current_suite_id()<0
    assert service.suite_sets(widget.current_suite_id())[0]['id']!=original_set['id']
    assert service.suite_sets(saved['id'])[0]==original_set
    widget.close()


def test_global_response_mode_preserves_hidden_curves_and_evidence(service,seed):
    from PySide6.QtWidgets import QTextBrowser
    from qfluentwidgets import TransparentPushButton
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget
    from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive
    keep_created_styles_alive()
    model=service.capture_model(ModelDraft(b'nep4_spin3 1 Fe\n','/test/nep.txt'),temporary=True)
    suite=service.create_diagnostic_plan('response',[(seed,DiagnosticSpec()),(seed,DiagnosticSpec(kind='displacement',low=-.1,high=.1,derivative_step=.01))])
    service.run_suite(model['id'],suite['id'])
    widget=ModelLibraryWidget(service=service);widget.show();widget.open_scorecard(suite['id'],model['id'])
    panel=widget.detail
    from test_validation_ui import _wait_until
    assert _wait_until(lambda: all(card.isVisible() for card in panel.eos_cards))
    panel.legend_buttons[model['id']].click()
    panel.diagnostic_mode.setCurrentIndex(1)
    for curve, benchmark in zip(panel._curves[model['id']],service.suite_sets(suite['id'])):
        details=service.run_details(service.latest_run(model['id'],benchmark['id']))['details']
        np.testing.assert_allclose(curve.getData()[1],details['derivative'])
        assert not curve.isVisible()
    panel.eos_cards[0].findChild(TransparentPushButton,'diagnosticEvidenceButton').click()
    evidence=panel.eos_cards[0].findChild(QTextBrowser,'diagnosticEvidence')
    assert evidence.isVisible() and '解析导数' in evidence.toPlainText() and '差分步长' in evidence.toPlainText()
    widget.table.selectRow(1)
    assert panel.diagnostic_mode.currentData()=='derivative'
    assert not panel._curves[model['id']][0].isVisible()
    panel.diagnostic_mode.setCurrentIndex(0)
    assert 'meV/atom' in panel.eos_plots[0].getAxis('left').labelText
    widget.close()


def test_editor_selection_preserves_parameters_and_keeps_creation_visible(seed):
    from PySide6.QtWidgets import QApplication
    from NepTrainKit.ui.widgets.diagnostic_editor import DiagnosticEditor
    widget = DiagnosticEditor()
    widget.resize(920, 640)
    widget.set_seeds([seed])
    widget.show()
    widget.rows['displacement'][1].setValue(-.25)
    widget.rows['displacement'][0].setChecked(False)
    assert widget.target_panel.isHidden()
    assert widget.experiment_panels['displacement'].isHidden()
    assert [spec.kind for _, spec in widget.experiments()] == ['volume']
    widget.rows['displacement'][0].setChecked(True)
    assert widget.rows['displacement'][1].value() == -.25
    assert not widget.target_panel.isHidden()
    for kind in ('spin_angle', 'spin_magnitude'):
        widget.rows[kind][0].setChecked(True)
    assert not widget.magnetic_panel.isHidden()
    assert not widget.variable.isChecked()  # Never infer a model's magnitude support.
    assert '跳过' in widget.preview.text()
    widget.variable.setChecked(True)
    widget.advanced_button.setChecked(True)
    QApplication.processEvents()
    assert widget.body_scroll.horizontalScrollBar().maximum() == 0
    assert widget.use_button.mapTo(widget, widget.use_button.rect().bottomRight()).y() < widget.height()
    widget.body_scroll.verticalScrollBar().setValue(widget.body_scroll.verticalScrollBar().maximum())
    QApplication.processEvents()
    assert widget.use_button.isVisible()
    assert len(widget.experiments()) == 4
    plain = seed.copy()
    plain.arrays.pop('spin', None)
    plain.arrays.pop('initial_magmoms', None)
    widget.set_seeds([plain])
    assert not widget.use_button.isEnabled()
    assert '未提供矢量磁矩' in widget.seed_hint.text()
    for kind in ('spin_angle', 'spin_magnitude'):
        widget.rows[kind][0].setChecked(False)
    assert widget.use_button.isEnabled()
    assert widget.magnetic_panel.isHidden()
    widget.close()


_ABACUS_CIF = '''data_none
_audit_creation_method generated by ABACUS
_cell_length_a 5.30799
_cell_length_b 5.30799
_cell_length_c 5.30799
_cell_angle_alpha 33.5573
_cell_angle_beta 33.5573
_cell_angle_gamma 33.5573
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe 0 0 0
Fe 0.51 0.51 0.51
'''


def test_editor_imports_abacus_identity_cif_without_altering_source(tmp_path, monkeypatch):
    from NepTrainKit.ui.widgets import diagnostic_editor as module
    path = tmp_path / 'STRU.cif'
    path.write_text(_ABACUS_CIF)
    before = path.read_bytes()
    widget = module.DiagnosticEditor()
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *_: (str(path), ''))
    try:
        widget.load_file()
        assert widget.use_button.isEnabled(), widget.preview.text()
        atoms = widget._seeds[0]
        assert atoms.get_chemical_formula() == 'Fe2'
        np.testing.assert_allclose(atoms.cell.cellpar(), [5.30799]*3 + [33.5573]*3)
        np.testing.assert_allclose(atoms.get_scaled_positions(), [[0,0,0],[.51,.51,.51]], atol=1e-12)
        assert existing_moment_vectors(atoms, lift_scalar=False) is None
        widget.rows['spin_angle'][0].setChecked(True)
        assert not widget.use_button.isEnabled() and '三维磁矩' in widget.preview.text()
        assert path.read_bytes() == before
    finally:
        widget.close()


def test_identity_cif_fallback_keeps_other_blocks_and_rejects_ambiguous_symmetry(tmp_path):
    from NepTrainKit.core.validation.protocols.diagnostic import read_seed_structures
    path = tmp_path / 'multiple.cif'
    standard = _ABACUS_CIF.replace('data_none', 'data_standard\n_space_group_IT_number 1')
    path.write_text(standard + '\n' + _ABACUS_CIF)
    assert [len(a) for a in read_seed_structures(path)] == [2, 2]
    path.write_text(_ABACUS_CIF.replace("'x,y,z'", "'x,y,z'\n'-x,-y,-z'"))
    with pytest.raises(ValueError, match='CIF 空间群'):
        read_seed_structures(path)
    from ase.spacegroup.spacegroup import SpacegroupNotFoundError
    path.write_text(_ABACUS_CIF.replace('data_none', 'data_none\n_space_group_name_H-M_alt invalid'))
    with pytest.raises(SpacegroupNotFoundError):
        read_seed_structures(path)


@pytest.mark.parametrize('filename,format', [('seed.xyz','xyz'), ('seed.extxyz','extxyz'), ('POSCAR','vasp'), ('seed.cif','cif')])
def test_seed_reader_preserves_supported_standard_formats(seed, tmp_path, filename, format):
    from ase.io import write
    from NepTrainKit.core.validation.protocols.diagnostic import read_seed_structures
    path = tmp_path / filename
    write(path, seed, format=format)
    actual, = read_seed_structures(path)
    np.testing.assert_allclose(actual.positions, seed.positions)
    assert actual.get_chemical_symbols() == seed.get_chemical_symbols()
    if format != 'xyz':
        np.testing.assert_allclose(actual.cell, seed.cell)
    if format == 'extxyz':
        np.testing.assert_allclose(existing_moment_vectors(actual), existing_moment_vectors(seed))
