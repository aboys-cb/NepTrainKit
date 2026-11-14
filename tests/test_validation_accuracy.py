"""Independent label coverage, cache provenance and magnetic-force accuracy."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.core.validation.accuracy import accuracy_labels
from NepTrainKit.core.validation.protocols import get_protocol


@pytest.fixture
def accuracy_case(tmp_path):
    a = Atoms('Fe2', positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    a.info['energy'] = 2.0
    a.info['virial'] = np.arange(9, dtype=float).reshape(3, 3) * 0.02
    a.new_array('forces', np.array([[.1, .2, .3], [.4, .5, .6]]))
    a.new_array('force_mag', np.array([[0., 0., 0.], [.2, 0., .1]]))
    b = Atoms('Fe', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    b.new_array('mforce', np.array([[.3, -.1, .2]]))
    c = Atoms('Fe2', positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    c.info['energy'] = -3.0
    frames = [a, b, c]
    for frame in frames:
        frame.set_initial_magnetic_moments(np.tile([0, 0, 2.0], (len(frame), 1)))
    forces = np.zeros((5, 3)); forces[:2] = a.arrays['forces'] + .002
    mforces = np.zeros((5, 3)); mforces[:2] = a.arrays['force_mag'] + .004
    mforces[2] = b.arrays['mforce'][0] + .004
    virials = np.zeros((3, 9)); virials[0] = a.info['virial'].reshape(9) / 2 + .003
    prediction = SimpleNamespace(energy=np.array([2.002, .5, -2.998]), forces=forces,
                                 structure_virials=virials, mforces=mforces)
    calls = []

    class Predictor:
        backend = 'cpu'
        def predict(self, structures):
            calls.append(True)
            assert all(frame.calc is None for frame in structures)
            assert all('force_mag' not in frame.arrays and 'mforce' not in frame.arrays for frame in structures)
            return prediction
        def close(self):
            pass

    service = ValidationService(Library(tmp_path / 'library'), predictor_factory=lambda *args: Predictor())
    model_path = tmp_path / 'nep.txt'; model_path.write_text('nep4_spin 1 Fe\n')
    data = tmp_path / 'independent.xyz'; write(data, frames, format='extxyz')
    model = service.capture_model(model_path)
    benchmark = service.import_set(data, name='mixed labels', protocol='efv', provenance={})
    return SimpleNamespace(service=service, model=model, benchmark=benchmark, frames=frames,
                           data=data, prediction=prediction, calls=calls, model_path=model_path)


def assert_metrics(run):
    assert run['status'] == 'succeeded', run['error']
    for quantity, expected in [('energy', 1), ('force', 2), ('virial', 3), ('mforce', 4)]:
        assert run['metrics'][quantity+'_rmse'] == pytest.approx(expected)
        assert run['metrics'][quantity+'_mae'] == pytest.approx(expected)


def test_partial_labels_keep_each_quantity_and_record_identity_scope_and_counts(accuracy_case):
    case = accuracy_case
    case.model_path.unlink(); case.data.unlink()
    run = case.service.evaluate(case.model['id'], case.benchmark['id'])
    assert_metrics(run)
    details = case.service.run_details(run)['details']
    assert details['label_counts']['energy'] == dict(structures=2, atoms=4, components=2,
                                                    missing_structures=1, evaluated_components=2)
    assert details['label_counts']['force'] == dict(structures=1, atoms=2, components=6,
                                                   missing_structures=2, evaluated_components=6)
    assert details['label_counts']['virial']['components'] == 6
    assert details['label_counts']['mforce'] == dict(structures=2, atoms=3, components=9,
                                                    missing_structures=1, evaluated_components=9)
    assert details['energy']['frame_indices'] == [0, 2]
    assert details['evaluation_scope']['frame_indices'] == [0, 1, 2]
    assert details['evaluation_scope']['atom_counts'] == [2, 1, 2]
    assert details['model_sha256'] == run['environment']['model_sha256'] == case.model['content_hash']
    assert details['dataset_sha256'] == run['environment']['dataset_sha256'] == case.benchmark['object_hash']
    assert details['prediction_source']['kind'] == 'recalculated'
    same = case.service.evaluate(case.model['id'], case.benchmark['id'])
    assert same['id'] == run['id'] and len(case.calls) == 1
    forced = case.service.evaluate(case.model['id'], case.benchmark['id'], force=True)
    assert forced['id'] != same['id'] and len(case.calls) == 2


def make_cache(case, directory):
    directory.mkdir()
    labels = accuracy_labels(case.frames)
    pred = case.prediction
    predicted = {'energy': pred.energy / labels.atom_counts, 'force': pred.forces,
                 'virial': pred.structure_virials[:, [0, 4, 8, 1, 5, 6]], 'mforce': pred.mforces}
    outputs = []
    for key in predicted:
        filename = key+'_independent.out'
        left, right = predicted[key], labels.values[key]
        if key == 'energy': left, right = left[:, None], right[:, None]
        # Historical outputs can contain predictions in unlabelled reference rows.
        right = np.where(np.isfinite(right), right, left)
        path = directory / filename
        np.savetxt(path, np.column_stack([left, right]), fmt='%.17g')
        outputs.append(dict(name=filename, size=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record = {'model': {'sha256': case.model['content_hash']},
              'dataset': {'sha256': case.benchmark['object_hash'], 'structures': 3, 'atoms': 5},
              'outputs': outputs, 'backend': {'resolved': 'cpu'}, 'nep_adapters_version': 'test'}
    manifest = {'schema_version': 1, 'predictions': {'independent.xyz': record}}
    (directory / 'prediction.meta.json').write_text(json.dumps(manifest))
    return manifest


def test_verified_cache_uses_shared_metrics_without_loading_native_model(accuracy_case, tmp_path):
    case = accuracy_case
    directory = tmp_path / 'cache'; make_cache(case, directory)
    def no_native(*args):
        pytest.fail('Verified output import must not require a native model backend')
    case.service.predictor_factory = no_native
    run = case.service.evaluate(case.model['id'], case.benchmark['id'], cache_directory=directory)
    assert_metrics(run)
    details = case.service.run_details(run)['details']
    assert details['prediction_source']['kind'] == 'verified_cache'
    assert len(details['prediction_source']['outputs']) == 4
    assert details['label_counts']['force']['components'] == 6
    assert details['label_counts']['mforce']['evaluated_components'] == 9
    assert not case.calls


def test_verified_cache_is_reused_by_normal_runs_when_runtime_matches(accuracy_case, tmp_path):
    case = accuracy_case
    native = case.service.evaluate(case.model['id'], case.benchmark['id'])
    directory = tmp_path / 'cache'; manifest = make_cache(case, directory)
    manifest['predictions']['independent.xyz']['nep_adapters_version'] = native['environment']['nep_adapters']
    (directory / 'prediction.meta.json').write_text(json.dumps(manifest))
    imported = case.service.evaluate(case.model['id'], case.benchmark['id'], cache_directory=directory)
    assert imported['environment'] == native['environment']
    again = case.service.evaluate(case.model['id'], case.benchmark['id'])
    assert again['id'] == imported['id'] and len(case.calls) == 1


@pytest.mark.parametrize('change', ['model', 'dataset', 'count', 'old_manifest', 'missing_manifest',
                                    'tampered_output', 'wrong_reference', 'partial', 'malformed_manifest'])
def test_unverified_or_stale_outputs_recompute_without_overwriting_sources(accuracy_case, tmp_path, change):
    case = accuracy_case
    directory = tmp_path / 'cache'; manifest = make_cache(case, directory)
    record = manifest['predictions']['independent.xyz']
    if change in ('model', 'dataset'):
        record[change]['sha256'] = 'wrong'
    elif change == 'count': record['dataset']['atoms'] = 6
    elif change == 'old_manifest':
        for entry in record['outputs']: entry.pop('sha256')
    elif change == 'tampered_output':
        (directory / 'force_independent.out').write_text('replaced')
    elif change == 'wrong_reference':
        path = directory / 'mforce_independent.out'
        array = np.loadtxt(path); array[0, 3] = 99; np.savetxt(path, array)
        next(row for row in record['outputs'] if row['name'] == path.name)['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif change == 'partial':
        (directory / 'virial_independent.out').unlink()
    file = directory / 'prediction.meta.json'
    file.write_text(json.dumps(manifest))
    if change == 'missing_manifest': file.unlink()
    if change == 'malformed_manifest': file.write_text('[]')
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    run = case.service.evaluate(case.model['id'], case.benchmark['id'], cache_directory=directory)
    assert_metrics(run)
    assert len(case.calls) == 1
    assert case.service.run_details(run)['warnings']
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize('key', ['energy', 'force', 'virial', 'mforce'])
def test_each_label_quantity_is_independently_sufficient_and_zero_is_valid(accuracy_case, key):
    frames = accuracy_case.frames
    frame = frames[0].copy()
    frame.info.clear()
    for name in ('forces', 'force_mag'): frame.arrays.pop(name, None)
    if key == 'energy': frame.info['energy'] = 0.0
    elif key == 'virial': frame.info['virial'] = np.zeros(9)
    else: frame.new_array('force_mag' if key == 'mforce' else 'forces', np.zeros((2, 3)))
    protocol = get_protocol('efv')
    assert not protocol.check([frame])
    summary = protocol.summarize_reference([frame])
    assert summary['label_counts'][key]['components'] == (1 if key == 'energy' else 6)
    assert sum(value['components'] > 0 for value in summary['label_counts'].values()) == 1


@pytest.mark.parametrize('key', ['forces', 'force_mag', 'mforce'])
def test_nonfinite_and_malformed_labels_fail_instead_of_disappearing(accuracy_case, key):
    frame = accuracy_case.frames[0].copy()
    frame.arrays.pop('force_mag', None)
    frame.arrays[key] = np.full((len(frame), 3), np.nan)
    assert any(issue.level == 'error' and 'finite' in issue.message for issue in get_protocol('efv').check([frame]))
    frame.arrays[key] = np.zeros((len(frame), 2))
    assert any(issue.level == 'error' and 'shape' in issue.message for issue in get_protocol('efv').check([frame]))


def test_missing_magnetic_prediction_is_na_and_conflicting_aliases_are_rejected(accuracy_case):
    case = accuracy_case
    del case.prediction.mforces
    run = case.service.evaluate(case.model['id'], case.benchmark['id'])
    assert run['status'] == 'succeeded' and run['metrics']['mforce_rmse'] is None
    result = case.service.run_details(run)
    assert result['details']['label_counts']['mforce']['evaluated_components'] == 0
    assert result['warnings']
    frame = case.frames[0].copy()
    frame.new_array('mforce', np.ones((len(frame), 3)))
    assert any('disagree' in issue.message for issue in get_protocol('efv').check([frame]))


def test_dataset_corruption_is_checked_even_when_a_run_is_cached(accuracy_case):
    case = accuracy_case
    first = case.service.evaluate(case.model['id'], case.benchmark['id'])
    path = case.service.library.object_path(case.benchmark['object_hash'])
    path.chmod(0o600); path.write_text('corrupt')
    second = case.service.evaluate(case.model['id'], case.benchmark['id'])
    assert second['status'] == 'failed' and 'corrupt' in second['error']
    assert second['id'] != first['id']


def test_accuracy_controls_report_counts_and_route_recalculation_and_cache(accuracy_case, tmp_path, monkeypatch):
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget
    from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog

    case = accuracy_case
    suite = case.service.create_suite('Accuracy', [case.benchmark['id']])
    run = case.service.evaluate(case.model['id'], case.benchmark['id'])
    widget = ModelLibraryWidget(service=case.service)
    widget.open_scorecard(suite['id'], case.model['id'])
    assert 'mforce_rmse' in widget.metrics_model._metric_keys
    details = case.service.run_details(run)['details']
    widget.detail.show_runs(case.benchmark, [(case.model['id'], 'Spin model', run, details)])
    text = widget.detail.info_label.text()
    assert 'MF 3' in text and '2 / 6 / 6 / 9' in text
    assert case.model['content_hash'][:12] in text
    calls = []
    monkeypatch.setattr(widget, '_run_validation', lambda *args, **kwargs: calls.append((args, kwargs)))
    widget.recalculate_action.trigger()
    assert calls[-1][1] == {'force': True}
    assert calls[-1][0] == ([case.model['id']],)
    monkeypatch.setattr('NepTrainKit.ui.views.model_library.QFileDialog.getExistingDirectory', lambda *args: str(tmp_path))
    widget.output_cache_action.trigger()
    assert calls[-1][1] == {'jobs': [(case.model['id'], case.benchmark['id'])], 'cache_directory': str(tmp_path)}
    dialog = BenchmarkImportDialog(widget, suites=[suite], suite_name=suite['name'])
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData('efv'))
    dialog.load_file(case.data)
    assert not dialog.validation_errors()
    assert 'MF 3' in dialog.preview_summary.text()
    widget.close()


def test_unified_spin_header_has_correct_capabilities():
    from NepTrainKit.core.validation.service import parse_model_header
    assert parse_model_header(b'nep 2 Fe O\nformat_version 1\nmodel nep_spin\n') == ('nep_spin', ('Fe', 'O'))


def test_model_and_dataset_import_do_not_load_or_predict_with_a_native_backend(accuracy_case, tmp_path):
    def forbidden(*args):
        pytest.fail('Import must not instantiate a predictor or evaluate the model')
    service = ValidationService(Library(tmp_path / 'another-library'), predictor_factory=forbidden)
    model = service.capture_model(accuracy_case.model_path)
    benchmark = service.import_set(accuracy_case.data, name='Reference', protocol='efv', provenance={})
    assert service.latest_run(model['id'], benchmark['id']) is None
    assert not list(service.library.results_dir.iterdir())
    assert benchmark['summary']['label_counts']['mforce']['atoms'] == 3


@pytest.mark.parametrize('outputs', [None, {}, [None], [{'name': '../outside.out', 'sha256': 'x'}]])
def test_shared_cache_checks_reject_malformed_output_metadata(tmp_path, outputs):
    from NepTrainKit.core.prediction_cache import prediction_record_error
    error = prediction_record_error({'outputs': outputs}, model_hash=None, dataset_hash='x',
                                    structures=1, atoms=1, directory=tmp_path)
    assert error and 'prediction.meta.json' in error


def test_existing_output_preview_verifies_full_cache_and_keeps_partial_historical(accuracy_case, tmp_path):
    from NepTrainKit.core.prediction_cache import find_output_groups, inspect_output_groups
    case = accuracy_case
    directory = tmp_path / 'preview'; make_cache(case, directory)
    (directory / 'independent.xyz').write_bytes(case.data.read_bytes())
    def preview():
        return inspect_output_groups(find_output_groups(directory), case.model['content_hash'])['independent']
    summary = preview()
    assert summary['status'] == 'verified'
    for key, value in [('energy', 1), ('force', 2), ('virial', 3), ('mforce', 4)]:
        assert summary['metrics'][key + '_rmse'] == pytest.approx(value)
    assert summary['counts']['mforce']['components'] == 9
    assert summary['dataset_sha256'] == case.benchmark['object_hash']
    (directory / 'mforce_independent.out').unlink()
    assert preview()['status'] == 'unverified'
    with (directory / 'force_independent.out').open('a') as f:
        f.write('\n0 0 0 0 0 0')
    assert preview()['status'] == 'mismatch'
    assert not case.calls


def test_existing_output_stream_masks_missing_vectors_and_reports_bad_properties(tmp_path):
    from NepTrainKit.core.prediction_cache import find_output_groups, inspect_output_groups, sha256_file
    path = tmp_path / 'force_train.out'
    path.write_text('# output\n' + '0.002 0.002 0.002 0 0 0\n' * 8200 + '1 1 1 nan nan nan\n')
    (tmp_path / 'energy_train.out').write_text('0 0 0\n')
    (tmp_path / 'mforce_test.out').write_text('0 0 0 0 nan 0\n')
    (tmp_path / 'notes.out').write_text('ignore')
    groups = find_output_groups(tmp_path)
    summary = inspect_output_groups(groups, 'model')
    train = summary['train']
    assert train['metrics']['force_rmse'] == pytest.approx(2)
    assert train['counts']['force'] == {'rows': 8201, 'components': 24600}
    assert train['outputs'][path.name] == sha256_file(path)
    assert 'energy' in train['errors'] and 'energy_rmse' not in train['metrics']
    assert 'mforce' in summary['test']['errors']
    assert inspect_output_groups(groups, 'model', lambda: True) == {}
    (tmp_path / 'prediction.meta.json').write_text('{"schema_version":1,"predictions":[]}')
    assert inspect_output_groups(groups, 'model')['train']['status'] == 'mismatch'
