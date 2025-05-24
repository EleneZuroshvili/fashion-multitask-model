import pytest
import sys
from evaluation import eval

def test_eval_parser_handles_arguments(monkeypatch):
    # Fake command-line args
    test_args = [
        "eval.py",
        "--task", "classification",
        "--checkpoint", "fake_model.pth",
        "--img-dir", "fake/img",
        "--anno-dir", "fake/annotations"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(FileNotFoundError):
        # We expect failure due to missing model, but want to confirm it *runs*
        eval.main()

import sys
import pytest
from evaluation import eval

def test_eval_parser_handles_arguments(monkeypatch):
    test_args = [
        "eval.py",
        "--task", "classification",
        "--checkpoint", "fake_model.pth",
        "--img-dir", "fake/img",
        "--anno-dir", "fake/annotations"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(FileNotFoundError):
        eval.main()


def test_eval_classification_runs(monkeypatch):
    test_args = [
        "eval.py",
        "--task", "classification",
        "--checkpoint", "src/tests/fake_checkpoints/fake_model.pth",
        "--img-dir", "src/tests/fake_data/img",
        "--anno-dir", "src/tests/fake_data/annotations",
        "--batch-size", "1"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Should run and print accuracy/f1 (won’t raise any error)
    eval.main()
