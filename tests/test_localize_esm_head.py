from cdskit.localize_esm_head import _resolve_model_source


def test_resolve_model_source_uses_huggingface_when_local_empty():
    source, local_only, source_type = _resolve_model_source(
        model_name='facebook/esm2_t6_8M_UR50D',
        model_local_dir='',
    )
    assert source == 'facebook/esm2_t6_8M_UR50D'
    assert local_only is False
    assert source_type == 'huggingface'


def test_resolve_model_source_prefers_local_dir():
    source, local_only, source_type = _resolve_model_source(
        model_name='facebook/esm2_t6_8M_UR50D',
        model_local_dir='/tmp/my_esm',
    )
    assert source == '/tmp/my_esm'
    assert local_only is True
    assert source_type == 'local'
