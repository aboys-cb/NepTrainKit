"""Portable suites preserve scientific inputs and never replace local decisions."""
import io
import json
import zipfile

import numpy as np
import pytest

from test_validation_framework import eos_frames, write_xyz, MODEL_HEADER
from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.core.validation.package import build_package, export_package, inspect_package, import_package


@pytest.fixture
def package_source(tmp_path):
    service = ValidationService(Library(tmp_path / 'source'))
    frames = []
    for group in ('A', 'B'):
        batch = eos_frames()
        for frame in batch:
            frame.info['eos_group'] = group
            frame.arrays['spin'] = np.ones((len(frame), 3))
        frames.extend(batch)
    path = write_xyz(tmp_path / 'groups.xyz', frames)
    refs = service.import_eos_groups(path, name='EOS', provenance={'notes': 'PBE'}, group_field='eos_group', group_names=['B'])
    suite = service.create_suite('HEA', [ref['id'] for ref in refs])
    model_path = tmp_path / 'nep.txt'
    model_path.write_text(MODEL_HEADER)
    model = service.capture_model(model_path, name='shared version', auto=False)
    service.set_baseline(suite['id'], model['id'])
    return service, suite, model


def test_roundtrip_selected_group_and_optional_model(package_source, tmp_path):
    source, suite, model = package_source
    target = ValidationService(Library(tmp_path / 'target'))
    preview = build_package(source, suite['id'])
    assert preview.manifest['models'] == []
    assert preview.manifest['suite']['baseline'] is None
    assert preview.references[0]['frame_count'] == 7
    assert str(tmp_path).encode() not in json.dumps(preview.manifest).encode()
    path = tmp_path / 'test.ntkpack'
    export_package(preview, path)
    preview = inspect_package(path)
    # A later replacement on disk cannot alter the confirmed preview.
    path.write_text('replaced')
    imported = import_package(target, preview, name='HEA')
    assert imported['baseline_model_id'] is None
    assert target.models() == []
    frames = target.set_frames(target.suite_sets(imported['id'])[0]['id'])
    originals = source.set_frames(source.suite_sets(suite['id'])[0]['id'])
    assert len(frames) == 7 and all(f.info['eos_group'] == 'B' for f in frames)
    for actual, expected in zip(frames, originals):
        np.testing.assert_array_equal(actual.positions, expected.positions)
        np.testing.assert_array_equal(actual.arrays['spin'], expected.arrays['spin'])
        assert actual.get_potential_energy() == expected.get_potential_energy()
    full = build_package(source, suite['id'], [model['id']])
    imported2 = import_package(target, full, name='HEA', project_name='Shared')
    assert imported2['name'] == 'HEA (2)'
    assert target.get_suite(imported['id']) == imported
    assert target.get_model(imported2['baseline_model_id'])['name'] == 'shared version'
    assert target.latest_run(imported2['baseline_model_id'], target.suite_sets(imported2['id'])[0]['id']) is None


def test_existing_models_and_baseline_unchanged(package_source, tmp_path):
    service, suite, model = package_source
    preview = build_package(service, suite['id'], [model['id']])
    service.star_model(model['id'])
    before = service.get_model(model['id'])
    project = next(p for p in service.projects() if p['id'] == before['project_id'])
    first = import_package(service, preview, name='HEA', project_name=project['name'])
    second = import_package(service, preview, name='HEA', project_name='Different project')
    assert first['baseline_model_id'] == model['id']
    assert second['baseline_model_id'] != model['id']
    assert service.get_model(model['id']) == before
    assert service.get_suite(suite['id']) == suite | {'baseline_model_id': model['id']}
    assert service.model_file(second['baseline_model_id']) == service.model_file(model['id'])


def rewrite(preview, change=None, extra=None, corrupt=False):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(preview.data)) as src, zipfile.ZipFile(output, 'w') as dst:
        manifest = json.loads(src.read('manifest.json'))
        if change:
            change(manifest)
        dst.writestr('manifest.json', json.dumps(manifest))
        for name in src.namelist():
            if name != 'manifest.json':
                dst.writestr(name, b'corrupt' if corrupt else src.read(name))
        if extra:
            dst.writestr(extra, b'bad')
    return output.getvalue()


@pytest.mark.parametrize('change,extra,corrupt', [
    (lambda m: m.update(version=999), None, False),
    (lambda m: m['references'][0].update(protocol_version=999), None, False),
    (lambda m: m['suite'].update(baseline='missing'), None, False),
    (None, '../outside', False), (None, None, True),
    (lambda m: m.update(references=None), None, False),
    (lambda m: m["references"].append(m["references"][0]), None, False),
])
def test_reject_invalid_without_mutation(package_source, tmp_path, change, extra, corrupt):
    source, suite, _ = package_source
    target = ValidationService(Library(tmp_path / 'target'))
    path = tmp_path / 'invalid.ntkpack'
    path.write_bytes(rewrite(build_package(source, suite['id']), change, extra, corrupt))
    with pytest.raises(ValueError):
        inspect_package(path)
    assert target.suites() == [] and target.sets() == [] and target.models() == []


def test_fitting_warnings_are_importable(package_source, tmp_path):
    source, _, _ = package_source
    frames = eos_frames(minimum=2.0)
    ref = source.import_set(write_xyz(tmp_path / 'warning.xyz', frames), name='Warning', protocol='eos', provenance={})
    suite = source.create_suite('Warning', [ref['id']])
    preview = build_package(source, suite['id'])
    assert preview.warnings
    assert import_package(source, preview, name='Warning')['name'] == 'Warning (2)'


def test_failed_import_rolls_back_all_records(package_source, tmp_path, monkeypatch):
    source, suite, model = package_source
    target = ValidationService(Library(tmp_path / 'target'))
    preview = build_package(source, suite['id'], [model['id']])
    original = target.library.put_bytes
    calls = 0
    def fail_reference(data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('disk full')
        return original(data)
    monkeypatch.setattr(target.library, 'put_bytes', fail_reference)
    with pytest.raises(OSError, match='disk full'):
        import_package(target, preview, name='HEA', project_name='New project')
    assert target.suites() == [] and target.sets() == [] and target.models() == [] and target.projects() == []


def test_export_failure_preserves_existing_file(package_source, tmp_path, monkeypatch):
    source, suite, _ = package_source
    preview = build_package(source, suite['id'])
    target = tmp_path / 'existing.ntkpack'
    target.write_bytes(b'keep existing')
    from pathlib import Path
    def denied(*_):
        raise PermissionError('read only')
    monkeypatch.setattr(Path, 'replace', denied)
    with pytest.raises(PermissionError):
        export_package(preview, target)
    assert target.read_bytes() == b'keep existing'
