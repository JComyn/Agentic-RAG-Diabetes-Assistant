from typing import Literal


def conservative_response(confidence: Literal['high', 'low'], answer: str) -> str:
    """Return a conservative fallback if confidence is low, otherwise return answer.

    This implements a strict conservative policy used to avoid presenting
    potentially misleading information when the model is not confident.
    """
    if confidence == 'low':
        return "Lo siento, no puedo responder con confianza a esa pregunta en este momento."
    return answer
