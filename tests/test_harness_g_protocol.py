from harness_g.protocol import parse_harness_g_action


def test_protocol_parser_actions():
    assert parse_harness_g_action("INIT", initialized=False)["is_valid"]

    action_map = {
        "A0": {"type": "SELECT"},
        "A1": {"type": "LOOKUP"},
        "A2": {"type": "ANSWER_WITH"},
        "A3": {"type": "ANSWER"},
    }
    assert parse_harness_g_action("A0", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("A1", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("A2", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("A3", action_map, initialized=True)["is_valid"]
    # no action accepts a || short query: the env builds the retrieval query
    assert not parse_harness_g_action("A0 || Ada", action_map, initialized=True)["is_valid"]
    assert not parse_harness_g_action("A1 || Ada Lovelace birthplace", action_map, initialized=True)["is_valid"]
    assert not parse_harness_g_action("A2 || Ada", action_map, initialized=True)["is_valid"]
    assert not parse_harness_g_action("A3 || Ada", action_map, initialized=True)["is_valid"]
    # natural-language queries and unknown ids are invalid
    assert not parse_harness_g_action("Ada Lovelace birthplace", action_map, initialized=True)["is_valid"]
    assert parse_harness_g_action("Ada Lovelace birthplace", action_map, initialized=True)["is_natural_query"]
    assert not parse_harness_g_action("A99", action_map, initialized=True)["is_valid"]
    assert not parse_harness_g_action("INIT", action_map, initialized=True)["is_valid"]
