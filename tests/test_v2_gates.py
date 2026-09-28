"""S0/S1/S2 gate helpers."""

from __future__ import annotations

import pytest

from evaluation.v2_gates import GateFail, s0_pass, s1_language_alive, s2_semantic_alive, assert_no_identity


def test_s0_gate_clean_pack():
    report = s0_pass(loss_start=4.0, loss_end=3.2, grad_norm=0.4, packed_ids=[1, 19, 20, 18])
    assert report["loss_moved"]


def test_s0_rejects_eot():
    with pytest.raises(GateFail):
        s0_pass(loss_start=1.0, loss_end=0.9, grad_norm=0.1, packed_ids=[1, 2, 19, 18])


def test_identity_rejected():
    with pytest.raises(GateFail):
        assert_no_identity("I am SHINRA today")


def test_s1_language_alive():
    s1_language_alive(2.0, 2.1, 5.0)
    with pytest.raises(GateFail):
        s1_language_alive(9.0, 9.0, 3.0)


def test_s2_semantic_alive():
    s2_semantic_alive({"negation_acc": 0.4, "relation_acc": 0.3, "coref_acc": 0.2})
    with pytest.raises(GateFail):
        s2_semantic_alive({"negation_acc": 0.0, "relation_acc": 0.3, "coref_acc": 0.2})
