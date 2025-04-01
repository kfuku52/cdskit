import sys
import pytest

def test_python_version():
    """サポートする Python バージョンをチェック"""
    assert sys.version_info >= (3, 8), "Python 3.8 以上が必要です"

@pytest.mark.skipif(sys.version_info < (3,9), reason="Python 3.9 未満では実行しない")
def test_typing_annotations():
    """Python 3.9 以上で追加された型ヒントのテスト"""
    from typing import Literal
    x: Literal["hello", "world"] = "hello"
    assert x == "hello"

@pytest.mark.skipif(sys.version_info < (3,10), reason="Python 3.10 未満では実行しない")
def test_match_case():
    """Python 3.10 以上で追加された match-case 文のテスト"""
    def match_test(value):
        match value:
            case 1:
                return "one"
            case 2:
                return "two"
            case _:
                return "other"
    
    assert match_test(1) == "one"
    assert match_test(3) == "other"

# Python 3.11 以上でのみ ExceptionGroup をインポート
if sys.version_info >= (3, 11):
    from exceptiongroup import ExceptionGroup  # Python 3.11 以上のみ
else:
    ExceptionGroup = None  # それ以前のバージョンでは None を設定

@pytest.mark.skipif(sys.version_info < (3,11), reason="Python 3.11 未満では実行しない")
def test_exception_group():
    """Python 3.11 以上で追加された ExceptionGroup のテスト"""
    if ExceptionGroup is None:
        pytest.skip("ExceptionGroup は Python 3.11 以上でのみ利用可能")

    try:
        raise ExceptionGroup("multiple errors", [ValueError("error1"), TypeError("error2")])
    except ExceptionGroup as e:
        assert len(e.exceptions) == 2
