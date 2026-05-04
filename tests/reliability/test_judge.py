from src.reliability.judge import classify_reliability


def test_unsupported_claim_detected():
    answer = "La vitamina D cura la diabetes."
    evidence = ["La insulina ayuda a controlar la glucosa."]

    result = classify_reliability(answer, evidence)
    assert result == "unsupported_claim", f"Expected 'unsupported_claim' but got {result}"
