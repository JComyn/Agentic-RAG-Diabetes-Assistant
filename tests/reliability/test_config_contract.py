from src import config


def test_reliability_threshold_contract():
    assert hasattr(config, "EVIDENCE_SCORE_THRESHOLD")
    assert hasattr(config, "MAX_RETRIEVAL_RECOVERY_ATTEMPTS")
    assert hasattr(config, "CONSERVATIVE_MEDICAL_MODE")
    assert 0.0 <= config.EVIDENCE_SCORE_THRESHOLD <= 1.0
