from src import config


def test_reliability_threshold_contract():
    assert hasattr(config, "EVIDENCE_SCORE_THRESHOLD")
    assert hasattr(config, "MAX_RETRIEVAL_RECOVERY_ATTEMPTS")
    assert hasattr(config, "CONSERVATIVE_MEDICAL_MODE")
    assert hasattr(config, "JUDGE_UNSUPPORTED_CLAIM_THRESHOLD")
    assert 0.0 <= config.EVIDENCE_SCORE_THRESHOLD <= 1.0
    assert 0.0 <= config.JUDGE_UNSUPPORTED_CLAIM_THRESHOLD <= 1.0
