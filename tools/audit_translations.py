#!/usr/bin/env python
"""Audit UI catalogs and late-bound messages without modifying repository files.

Run ``python tools/audit_translations.py --check --json /tmp/i18n.json``.
Errors fail --check; candidates need human review and never fail it. See
``tools/translation-audit.md`` for coverage and limitations.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
from string import Formatter
import sys
import tempfile
from typing import Callable
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MESSAGE_METHODS = {
    'send_error_message', 'send_warning_message', 'send_info_message',
    'send_success_message', 'send_message_box',
}
UI_METHODS = {
    'setText', 'setToolTip', 'setPlaceholderText', 'setWindowTitle',
    'setTitle', 'setContent', 'setStatusTip',
}
ENGLISH_PROSE = re.compile(
    r'\b(?:requires?|required|must|cannot|could not|failed|unsupported|invalid|expected|'
    r'does not|do not|not found|please|missing|should|only supports?|is empty)\b', re.I,
)
HAN = re.compile(r'[\u4e00-\u9fff]')
# Exact technical/UI literals, not a blanket exemption for ASCII strings.
TECHNICAL_TEXT = {'NepTrainKit', 'all', 'A', 'B', 'Cu', 'Au', 'Ni,Co', 'Ni:1,Co:0', 'Ni:0,Co:1', 'Co,Cr,Ni'}
TECHNICAL_WORDS = {'spin', 'energy', 'force', 'forces', 'virial', 'stress', 'mforce', 'descriptor',
                   'descriptors', 'float', 'int', 'dtype', 'shape', 'axis', 'seed', 'count', 'percent',
                   'vispy', 'pyqtgraph', 'nep', 'adapters'}


def prose_text(text: str) -> str:
    """Exclude format values and filenames from language heuristics."""
    text = text.replace('⟦value⟧', '')
    text = re.sub(r'\{[^{}]*\}', '', text)
    return re.sub(r'[\w-]+\.(?:out|txt|xyz|json|csv|npy)\b', '', text)


def has_english_residue(text: str) -> bool:
    """Flag lowercase prose, allowing common scientific fields and file names."""
    text = prose_text(text)
    return any(word not in TECHNICAL_WORDS for word in re.findall(r'\b[a-z]{2,}\b', text))


def format_fields(text: str) -> Counter:
    """Count Python fields (including format specs) and Qt argument markers."""
    fields = Counter()
    for _, field, spec, conversion in Formatter().parse(text):
        if field is not None:
            fields[(field, spec, conversion)] += 1
    fields.update(re.findall(r'%(?:L?\d+|n)\b', text))
    return fields


def literal_template(node: ast.AST) -> tuple[str, bool] | None:
    """Read literals/f-strings without executing code or guessing dynamic values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        chunks = []
        for child in node.values:
            chunks.append(child.value if isinstance(child, ast.Constant) else '⟦value⟧')
        return ''.join(chunks), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal_template(node.left), literal_template(node.right)
        if left and right:
            return left[0] + right[0], left[1] or right[1]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'format':
        source = literal_template(node.func.value)
        if source:
            return re.sub(r'\{[^{}]*\}', '⟦value⟧', source[0]), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        source = literal_template(node.left)
        if source:
            return re.sub(r'%[-+0-9.#]*[sdrfg]', '⟦value⟧', source[0]), True
    return None


def audit_catalog(path: Path) -> tuple[dict, list[dict]]:
    catalog = {}
    issues = []
    for context in ET.parse(path).getroot().findall('context'):
        name = context.findtext('name') or ''
        for message in context.findall('message'):
            translation = message.find('translation')
            status = translation.get('type') if translation is not None else 'missing'
            if status in {'obsolete', 'vanished'}:
                continue
            source = message.findtext('source') or ''
            targets = translation.findall('numerusform') if translation is not None else []
            values = [(target.text or '') for target in targets] or [message.findtext('translation') or '']
            catalog[(name, source)] = values[0]
            reason = None
            if status in {'unfinished', 'missing'} or any(not value.strip() for value in values):
                reason = 'unfinished_translation'
            else:
                try:
                    if any(format_fields(source) != format_fields(value) for value in values):
                        reason = 'placeholder_mismatch'
                except ValueError:
                    reason = 'invalid_format_string'
            if reason:
                issues.append(dict(severity='error', rule=reason, path=str(path), line=0,
                                   context=name, source=source, rendered=values))
            elif any(ENGLISH_PROSE.search(prose_text(value)) or (
                HAN.search(value) and ENGLISH_PROSE.search(source) and has_english_residue(value)
            ) for value in values):
                issues.append(dict(severity='candidate', rule='catalog_english_residue', path=str(path), line=0,
                                   context=name, source=source, rendered=values))
    return catalog, issues


def audit_source(path: Path, catalog: dict, translate: Callable[[str], str]) -> list[dict]:
    """Inspect visible sinks, exception candidates and structured error templates."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    issues = []
    for node in ast.walk(tree):
        arg = None
        kind = ''
        context = None
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
            arg, kind = node.exc.args[0], 'exception'
        if isinstance(node, ast.Call):
            method = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            if method == 'CardOperationError' and len(node.args) >= 2:
                arg, kind, context = node.args[1], 'translation', 'CardOperationError'
            elif method == 'translate' and len(node.args) >= 2 and isinstance(node.func, ast.Attribute):
                owner = getattr(node.func.value, 'id', '')
                if owner in {'QCoreApplication', 'QApplication'}:
                    arg, kind = node.args[1], 'translation'
                    ctx = literal_template(node.args[0])
                    context = ctx[0] if ctx and not ctx[1] else None
            elif method in {'tr', '_tr'} and node.args:
                arg, kind = node.args[0], 'translation'
            elif method in MESSAGE_METHODS and node.args:
                arg, kind = node.args[0], 'message'
            elif method in UI_METHODS and node.args:
                arg, kind = node.args[0], 'ui_literal'
        if arg is None:
            continue
        value = literal_template(arg)
        if value is None:
            continue
        source, dynamic = value
        if not source.strip():
            continue
        issue = dict(severity='candidate', rule=kind, path=str(path), line=node.lineno, source=source)
        if kind == 'translation':
            if dynamic:
                issue.update(severity='error', rule='interpolated_before_translation')
            elif context and (context, source) not in catalog:
                issue.update(severity='error', rule='missing_catalog_entry', context=context)
            elif not context and not any(key[1] == source for key in catalog):
                issue.update(severity='error', rule='missing_catalog_entry')
            else:
                continue
        elif kind == 'ui_literal':
            prose = source.replace('⟦value⟧', '')
            if not re.search(r'[A-Za-z\u4e00-\u9fff]', prose) or source in TECHNICAL_TEXT:
                continue
        else:
            # Raises carrying a structured error have already been audited at the call.
            if isinstance(node, ast.Raise) and getattr(node.exc.func, 'id', '') == 'CardOperationError':
                continue
            rendered = translate(source)
            partial = HAN.search(rendered) and ENGLISH_PROSE.search(source) and has_english_residue(rendered)
            if not ENGLISH_PROSE.search(rendered) and not partial:
                continue
            issue.update(rule='mixed_runtime_message' if HAN.search(rendered) else 'untranslated_runtime_message',
                         origin=kind, rendered=rendered)
        issues.append(issue)
    return issues


def annotate_review(root: Path, issues: list[dict]) -> None:
    """Recognize only the exact reviewed source AND rendered result."""
    path = root / 'tools/translation_review.json'
    if not path.exists():
        return
    approved = json.loads(path.read_text(encoding='utf-8')).get('retained_candidates', [])
    for issue in issues:
        if issue['severity'] != 'candidate':
            continue
        match = next((entry for entry in approved if all(
            entry.get(key) == issue.get(key) for key in ('path', 'source', 'context', 'rendered')
        )), None)
        if match:
            issue['review'] = match['review']
            issue['review_reason'] = match['reason']


def audit(root: Path, translate: Callable[[str], str]) -> dict:
    catalog, issues = audit_catalog(root / 'src/NepTrainKit/translations/neptrainkit_zh_CN.ts')
    paths = sorted((root / 'src').rglob('*.py'))
    for path in paths:
        issues.extend(audit_source(path, catalog, translate))
    registry = root / 'src/NepTrainKit/ui/runtime_error_catalog.py'
    if registry.exists():
        tree = ast.parse(registry.read_text(encoding='utf-8'))
        entries = next(ast.literal_eval(node.value) for node in tree.body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == 'RUNTIME_ERROR_TEMPLATES'
                           for target in node.targets))
        for context, template in entries:
            if (context, template) not in catalog:
                issues.append(dict(severity='error', rule='missing_catalog_entry',
                                   path=str(registry), line=0, context=context, source=template))
    for issue in issues:
        issue['path'] = Path(issue['path']).relative_to(root).as_posix()
    annotate_review(root, issues)
    return dict(python_files=len(paths), catalog_entries=len(catalog),
                errors=sum(item['severity'] == 'error' for item in issues),
                candidates=sum(item['severity'] == 'candidate' for item in issues),
                unreviewed=sum(item['severity'] == 'candidate' and 'review' not in item for item in issues),
                issues=issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail on confirmed catalog/translation-call errors only.')
    parser.add_argument('--json', type=Path, help='Write the complete report to this path.')
    parser.add_argument('--require-reviewed', action='store_true', help='Fail if a candidate differs from the reviewed source and rendered text.')
    args = parser.parse_args(argv)
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    sys.path.insert(0, str(ROOT / 'src'))
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, QTranslator

    # Import the message formatter without activating or repairing a user's
    # managed runtime. Avoid i18n.py, whose Config import opens the user database.
    with tempfile.TemporaryDirectory(prefix='ntk-translation-audit-') as runtime:
        previous_runtime = os.environ.get('NEPTRAINKIT_RUNTIME_ROOT')
        os.environ['NEPTRAINKIT_RUNTIME_ROOT'] = runtime
        try:
            from NepTrainKit.ui.messages import translate_runtime_message
        finally:
            if previous_runtime is None:
                os.environ.pop('NEPTRAINKIT_RUNTIME_ROOT', None)
            else:
                os.environ['NEPTRAINKIT_RUNTIME_ROOT'] = previous_runtime

    app = QApplication.instance() or QApplication([])
    translator = QTranslator(app)
    loaded = translator.load(str(ROOT / 'src/NepTrainKit/translations/neptrainkit_zh_CN.qm'))
    app.installTranslator(translator)
    try:
        report = audit(ROOT, translate_runtime_message)
        # Detect a missing/stale compiled catalog, too. TS alone is insufficient.
        catalog, _ = audit_catalog(ROOT / 'src/NepTrainKit/translations/neptrainkit_zh_CN.ts')
        if not loaded:
            report['issues'].append(dict(severity='error', rule='compiled_catalog_unavailable',
                path='src/NepTrainKit/translations/neptrainkit_zh_CN.qm', line=0, source=''))
            report['errors'] += 1
        for (context, source), target in (catalog.items() if loaded else []):
            if '%n' in source:
                continue  # Numerus forms are checked structurally above.
            if QCoreApplication.translate(context, source) != target:
                report['issues'].append(dict(severity='error', rule='compiled_catalog_mismatch',
                    path='src/NepTrainKit/translations/neptrainkit_zh_CN.qm', line=0,
                    context=context, source=source))
                report['errors'] += 1
    finally:
        app.removeTranslator(translator)
    print(f"{report['python_files']} Python files, {report['catalog_entries']} catalog entries: "
          f"{report['errors']} errors, {report['candidates']} candidates ({report['unreviewed']} unreviewed)")
    for issue in report['issues']:
        if issue['severity'] == 'error':
            print(f"{issue['path']}:{issue['line']}: {issue['rule']}: {issue['source']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int((args.check and report['errors'] > 0) or (args.require_reviewed and report['unreviewed'] > 0))


if __name__ == '__main__':
    raise SystemExit(main())
