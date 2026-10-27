import pytest

from xray.metrics import clinical_metrics, confusion, threshold_for_sensitivity

Y = [1, 1, 1, 1, 0, 0, 0, 0]
P = [0.9, 0.8, 0.6, 0.3, 0.7, 0.2, 0.1, 0.05]


def test_confusion_counts():
    assert confusion(Y, P, 0.5) == {"tp": 3, "tn": 3, "fp": 1, "fn": 1}


def test_clinical_metrics():
    m = clinical_metrics(Y, P, 0.5)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["ppv"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75)


def test_threshold_reaches_target_sensitivity_with_best_specificity():
    t = threshold_for_sensitivity(Y, P, target=1.0)
    assert t == pytest.approx(0.3)
    m = clinical_metrics(Y, P, t)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == pytest.approx(0.75)


def test_empty_classes_do_not_divide_by_zero():
    m = clinical_metrics([0, 0], [0.1, 0.2], 0.5)
    assert m["sensitivity"] == 0.0 and m["specificity"] == 1.0
