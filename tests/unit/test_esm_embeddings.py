import numpy as np

from cdskit.esm_embeddings import ESMEmbeddingCache, encoder_cache_key


def test_embedding_cache_is_bounded_and_owns_its_arrays():
    vector = np.arange(4, dtype=np.float32)
    cache = ESMEmbeddingCache(max_bytes=2 * (vector.nbytes + 512))
    cache.put("encoder", "A", vector)
    cache.put("encoder", "B", vector)
    cache.get("encoder", "A")
    cache.put("encoder", "C", vector)
    vector[:] = 0
    assert cache.get("encoder", "B") is None
    np.testing.assert_array_equal(cache.get("encoder", "A"), np.arange(4))
    assert cache.size_bytes <= cache.max_bytes
    assert not cache.get("encoder", "C").flags.writeable


def test_local_encoder_changes_invalidate_embedding_identity(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"original weights")
    before = encoder_cache_key(str(tmp_path), None, 200, "mean")
    weights.write_bytes(b"changed weights with different size")
    assert encoder_cache_key(str(tmp_path), None, 200, "mean") != before
