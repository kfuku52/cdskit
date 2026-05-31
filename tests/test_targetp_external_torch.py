from cdskit.localize_model import LOCALIZATION_CLASSES
from cdskit import targetp_external_torch as external_torch


def _row(class_name, idx, source='unit'):
    return {
        'source': source,
        'accession': '{}{}'.format(class_name, idx),
        'sequence': 'M{}AAAAAA'.format('K' * (idx + 1)),
        'organism_group': 'plant' if class_name in ['cTP', 'lTP'] else 'non_plant',
        'localization': class_name,
        'peroxisome': 'no',
        'cleavage_site': '3' if class_name != 'noTP' else '',
        'fold_id': str(idx % 2),
    }


def test_fit_external_augmented_torch_runtime_model_uses_external_calibration(monkeypatch):
    target_rows = [
        _row(class_name=class_name, idx=10 + class_i, source='targetp')
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
    ]
    external_rows = [
        _row(class_name=class_name, idx=idx, source='external')
        for class_name in LOCALIZATION_CLASSES
        for idx in range(4)
    ]
    captured = {}

    monkeypatch.setattr(
        external_torch,
        'read_tsv',
        lambda path: list(target_rows),
    )
    monkeypatch.setattr(
        external_torch,
        'build_external_augmented_training_rows',
        lambda **kwargs: (list(external_rows), {'sampled_rows': len(external_rows)}),
    )

    def fake_fit_targetp2_torch_model(**kwargs):
        captured.update(kwargs)
        return {
            'config': {
                'seq_len': kwargs['x_train'].shape[1],
                'cleavage_loss_weight': kwargs['cleavage_loss_weight'],
                'balanced_batch': kwargs['balanced_batch'],
                'selection_metric': kwargs['selection_metric'],
                'type_class_weight': kwargs['type_class_weight'],
            },
            'state_dict': {},
            'best_metrics': {
                'val_macro_f1': 0.75,
                'val_thresholds': {
                    class_name: 1.0
                    for class_name in LOCALIZATION_CLASSES
                },
            },
            'final_val_metrics': {'macro_f1': 0.74},
            'seed': kwargs['seed'],
        }

    monkeypatch.setattr(
        external_torch,
        'fit_targetp2_torch_model',
        fake_fit_targetp2_torch_model,
    )

    result = external_torch.fit_external_augmented_torch_runtime_model(
        training_tsv='targetp.tsv',
        uniprot_tsv='uniprot.tsv',
        calibration_fraction=0.25,
        seq_len=12,
        device='cpu',
    )

    assert captured['x_train'].shape[0] == len(target_rows) + 15
    assert captured['x_val'].shape[0] == 5
    assert captured['seq_len'] == 12
    assert captured['cleavage_loss_weight'] == 0.0
    assert captured['balanced_batch'] == 'yes'
    assert captured['selection_metric'] == 'val_macro_f1'
    assert captured['type_class_weight'] == 'none'
    assert result['report']['num_external_train_rows'] == 15
    assert result['report']['num_external_calibration_rows'] == 5
    assert result['model']['metadata']['model_arch'] == 'targetp_torch_v1_external_augmented'


def test_fit_external_augmented_torch_runtime_model_can_reuse_external_tsv(monkeypatch):
    target_rows = [
        _row(class_name=class_name, idx=10 + class_i, source='targetp')
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
    ]
    external_rows = [
        _row(class_name=class_name, idx=idx, source='external')
        for class_name in LOCALIZATION_CLASSES
        for idx in range(4)
    ]
    calls = {'builder': 0}

    def fake_read_tsv(path):
        if path == 'precomputed.tsv':
            return list(external_rows)
        return list(target_rows)

    monkeypatch.setattr(external_torch, 'read_tsv', fake_read_tsv)

    def fake_builder(**kwargs):
        calls['builder'] += 1
        return list(), {}

    monkeypatch.setattr(
        external_torch,
        'build_external_augmented_training_rows',
        fake_builder,
    )
    monkeypatch.setattr(
        external_torch,
        'fit_targetp2_torch_model',
        lambda **kwargs: {
            'config': {'seq_len': kwargs['x_train'].shape[1]},
            'state_dict': {},
            'best_metrics': {'val_macro_f1': 0.75},
            'final_val_metrics': {'macro_f1': 0.74},
            'seed': kwargs['seed'],
        },
    )

    result = external_torch.fit_external_augmented_torch_runtime_model(
        training_tsv='targetp.tsv',
        uniprot_tsv='unused.tsv',
        external_tsv='precomputed.tsv',
        calibration_fraction=0.25,
        seq_len=12,
        device='cpu',
    )

    assert calls['builder'] == 0
    assert result['report']['external_tsv'] == 'precomputed.tsv'
    assert result['report']['external_report']['source'] == 'external_tsv'
    assert result['model']['metadata']['precomputed_external_tsv'] == 'precomputed.tsv'


def test_fit_external_augmented_torch_runtime_model_requires_all_calibration_classes(monkeypatch):
    target_rows = [
        _row(class_name=class_name, idx=10 + class_i, source='targetp')
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
    ]
    external_rows = [
        _row(class_name=class_name, idx=idx, source='external')
        for class_name in LOCALIZATION_CLASSES
        for idx in range(2)
        if class_name != 'lTP'
    ]

    monkeypatch.setattr(
        external_torch,
        'read_tsv',
        lambda path: list(target_rows),
    )
    monkeypatch.setattr(
        external_torch,
        'build_external_augmented_training_rows',
        lambda **kwargs: (list(external_rows), {'sampled_rows': len(external_rows)}),
    )

    import pytest

    with pytest.raises(ValueError, match='missing classes: lTP'):
        external_torch.fit_external_augmented_torch_runtime_model(
            training_tsv='targetp.tsv',
            uniprot_tsv='uniprot.tsv',
            calibration_fraction=0.5,
            device='cpu',
        )
