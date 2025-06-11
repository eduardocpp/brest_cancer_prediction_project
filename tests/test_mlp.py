import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import train_mlp


def test_train_mlp_raises_when_no_configuration_converges(monkeypatch):
    def failing_cv_score(*args, **kwargs):
        raise ValueError("fail")

    monkeypatch.setattr('models.mlp.cross_val_score', failing_cv_score)

    X_train = [[0], [1]]
    y_train = [0, 1]

    with pytest.raises(RuntimeError, match="No MLP configuration converged"):
        train_mlp(X_train, y_train, size_options=[1], max_layers=1, cv=2)


def test_train_mlp_returns_model(monkeypatch):
    def dummy_cv_score(*args, **kwargs):
        return [1.0, 1.0]

    monkeypatch.setattr('models.mlp.cross_val_score', dummy_cv_score)

    X_train = [[0], [1]]
    y_train = [0, 1]

    model, config, score = train_mlp(X_train, y_train, size_options=[1], max_layers=1, cv=2, max_iter=10)

    assert config == (1,)
    assert score == 1.0
    assert hasattr(model, 'predict')
