from pact.alfworld_official import detect_repetitive_actions


def test_detect_repeated_single_action():
    actions = ["go to desk 1", "look", "look", "look", "look", "look"]

    assert detect_repetitive_actions(actions, consecutive_limit=5) == "repeated_action:look"


def test_repeated_single_action_below_limit_does_not_stop():
    actions = ["look", "look", "look", "look"]

    assert detect_repetitive_actions(actions, consecutive_limit=5) is None


def test_short_action_cycle_does_not_stop():
    actions = [
        "take laptop 1 from dresser 1",
        "move laptop 1 to dresser 1",
        "take laptop 1 from dresser 1",
        "move laptop 1 to dresser 1",
        "take laptop 1 from dresser 1",
        "move laptop 1 to dresser 1",
    ]

    assert detect_repetitive_actions(actions, consecutive_limit=5) is None
