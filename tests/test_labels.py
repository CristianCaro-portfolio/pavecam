import pytest

from pavecam.labels import pick_label, to_class


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("Dry", "dry"),
        ("Trace Moisture", "wet"),
        ("Chemical Wet", "wet"),
        ("Ice Warning", "frozen"),
        ("ice watch", "frozen"),
        ("Frost", "frozen"),
        ("Snow Warning", "frozen"),
        ("Other", None),
        ("Error", None),
        (None, None),
    ],
)
def test_to_class(code, expected):
    assert to_class(code) == expected


def test_pick_label_falls_back_to_second_sensor():
    assert pick_label("Other", "Wet") == "wet"
    assert pick_label("Dry", "Ice Warning") == "dry"
    assert pick_label(None, None) is None
