"""Map RWIS pavement sensor condition codes to the three classes the model predicts.

Codes follow the IEM explanation of NTCIP ESS surface status:
  Dry                                -> dry
  Wet, Trace Moisture, Chemical Wet  -> wet
  Ice Warning, Ice Watch, Frost,
  Snow Warning                       -> frozen  (moisture at or below freezing)
Other, Error, NA or anything unknown is not a usable label and is dropped.
"""

from __future__ import annotations

CLASSES: tuple[str, ...] = ("dry", "wet", "frozen")

_CODE_TO_CLASS = {
    "dry": "dry",
    "wet": "wet",
    "trace moisture": "wet",
    "chemical wet": "wet",
    "ice warning": "frozen",
    "ice watch": "frozen",
    "frost": "frozen",
    "snow warning": "frozen",
    "snow watch": "frozen",
}


def to_class(code: str | None) -> str | None:
    if code is None:
        return None
    return _CODE_TO_CLASS.get(str(code).strip().lower())


def pick_label(primary: str | None, secondary: str | None) -> str | None:
    """Prefer pavement sensor 0, fall back to sensor 1 when 0 has no usable code."""
    return to_class(primary) or to_class(secondary)
