from src import config


def test_reliability_threshold_contract():
    assert hasattr(config, "EVIDENCE_SCORE_THRESHOLD")
    assert hasattr(config, "MAX_RETRIEVAL_RECOVERY_ATTEMPTS")
    assert hasattr(config, "CONSERVATIVE_MEDICAL_MODE")
    assert hasattr(config, "JUDGE_UNSUPPORTED_CLAIM_THRESHOLD")
    # type/domain checks
    # MAX_RETRIEVAL_RECOVERY_ATTEMPTS must be a plain int (not bool) and non-negative
    assert type(config.MAX_RETRIEVAL_RECOVERY_ATTEMPTS) is int
    assert config.MAX_RETRIEVAL_RECOVERY_ATTEMPTS >= 0
    assert isinstance(config.CONSERVATIVE_MEDICAL_MODE, bool)
    # EVIDENCE_SCORE_THRESHOLD and JUDGE_UNSUPPORTED_CLAIM_THRESHOLD must be numeric (int/float) but not bool
    assert (isinstance(config.EVIDENCE_SCORE_THRESHOLD, (int, float)) and not isinstance(config.EVIDENCE_SCORE_THRESHOLD, bool))
    assert (isinstance(config.JUDGE_UNSUPPORTED_CLAIM_THRESHOLD, (int, float)) and not isinstance(config.JUDGE_UNSUPPORTED_CLAIM_THRESHOLD, bool))
    # existing numeric range checks
    assert 0.0 <= config.EVIDENCE_SCORE_THRESHOLD <= 1.0
    assert 0.0 <= config.JUDGE_UNSUPPORTED_CLAIM_THRESHOLD <= 1.0
