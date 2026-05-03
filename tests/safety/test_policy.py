from src.safety.policy import conservative_response


def test_conservative_fallback_on_low_confidence():
    answer = "La vitamina D cura la diabetes."
    fallback = conservative_response('low', answer)
    assert fallback == "Lo siento, no puedo responder con confianza a esa pregunta en este momento."


def test_policy_preserves_high_confidence():
    answer = "La vitamina D ayuda a regular los niveles de calcio."
    preserved = conservative_response('high', answer)
    assert preserved == answer
