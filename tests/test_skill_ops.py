import json

from pact.skill_ops import (
    apply_skill_ops,
    canonicalize_skill_text,
    parse_skill_rules,
    render_for_agent,
    render_for_critic,
)


def ops_payload(ops):
    return json.dumps({"ops": ops})


def sample_skill():
    return (
        "[RULE 001] Quote workbook paths\n"
        "content: Always quote workbook file placeholders.\n"
        "why: Unquoted placeholders can become invalid Python.\n"
        "\n"
        "[RULE 002] Write direct values\n"
        "content: Write computed literal values when recalculation is unavailable.\n"
        "why: Evaluators may inspect saved values without recalculating formulas.\n"
    )


def test_parse_empty_skills_md():
    assert parse_skill_rules("") == []


def test_parse_render_canonical_3line_round_trip():
    rules = parse_skill_rules(sample_skill())
    assert [rule.id for rule in rules] == ["001", "002"]
    assert render_for_critic(rules) == sample_skill()


def test_render_for_agent_omits_why():
    rules = parse_skill_rules(sample_skill())
    rendered = render_for_agent(rules)
    assert "content:" in rendered
    assert "why:" not in rendered


def test_render_for_critic_includes_why():
    rules = parse_skill_rules(sample_skill())
    rendered = render_for_critic(rules)
    assert "why:" in rendered


def test_add_assigns_001_for_empty_skill():
    result = apply_skill_ops(
        "",
        ops_payload(
            [
                {
                    "op": "add",
                    "rule": {
                        "title": "Check headers",
                        "content": "Read table headers before choosing columns.",
                        "why": "Student used the wrong column while teacher matched the header.",
                    },
                }
            ]
        ),
    )

    assert result.ok, result.error
    assert result.text.startswith("[RULE 001] Check headers\n")


def test_add_assigns_next_id_after_existing_max():
    result = apply_skill_ops(
        sample_skill(),
        ops_payload(
            [
                {
                    "op": "add",
                    "rule": {
                        "title": "Save outputs",
                        "content": "Save the modified workbook before exiting.",
                        "why": "The evaluator checks the saved output file.",
                    },
                }
            ]
        ),
    )

    assert result.ok, result.error
    assert "[RULE 003] Save outputs" in result.text


def test_update_keeps_id_and_position():
    result = apply_skill_ops(
        sample_skill(),
        ops_payload(
            [
                {
                    "op": "update",
                    "id": "001",
                    "rule": {
                        "title": "Quote file placeholders",
                        "content": "Always quote input and output workbook placeholders in Python APIs.",
                        "why": "The placeholders are literal paths, not Python variables.",
                    },
                }
            ]
        ),
    )

    assert result.ok, result.error
    rules = parse_skill_rules(result.text)
    assert [rule.id for rule in rules] == ["001", "002"]
    assert rules[0].title == "Quote file placeholders"


def test_delete_removes_rule_without_renumbering():
    result = apply_skill_ops(sample_skill(), ops_payload([{"op": "delete", "id": "001"}]))

    assert result.ok, result.error
    assert "[RULE 001]" not in result.text
    assert "[RULE 002] Write direct values" in result.text


def test_multiple_ops_apply_in_order():
    result = apply_skill_ops(
        sample_skill(),
        ops_payload(
            [
                {"op": "delete", "id": "001"},
                {
                    "op": "add",
                    "rule": {
                        "title": "Save outputs",
                        "content": "Save the modified workbook to the requested output path.",
                        "why": "Unsaved changes are not visible to file-based evaluators.",
                    },
                },
                {
                    "op": "update",
                    "id": "002",
                    "rule": {
                        "title": "Write direct values",
                        "content": "Write literal computed values whenever recalculation is unavailable or uncertain.",
                        "why": "The benchmark may inspect saved values without recalculating formulas.",
                    },
                },
            ]
        ),
    )

    assert result.ok, result.error
    rules = parse_skill_rules(result.text)
    assert [rule.id for rule in rules] == ["002", "003"]
    assert rules[0].content.startswith("Write literal computed values")


def test_add_after_deleting_max_id_does_not_reuse_deleted_id():
    result = apply_skill_ops(
        sample_skill(),
        ops_payload(
            [
                {"op": "delete", "id": "002"},
                {
                    "op": "add",
                    "rule": {
                        "title": "Save outputs",
                        "content": "Save the modified workbook to the requested output path.",
                        "why": "Unsaved changes are not visible to file-based evaluators.",
                    },
                },
            ]
        ),
    )

    assert result.ok, result.error
    rules = parse_skill_rules(result.text)
    assert [rule.id for rule in rules] == ["001", "003"]


def test_empty_ops_returns_canonical_original_skill():
    result = apply_skill_ops(sample_skill(), ops_payload([]))

    assert result.ok, result.error
    assert result.text == sample_skill()


def test_reject_malformed_json():
    result = apply_skill_ops("", "not json")
    assert not result.ok
    assert "JSON object" in result.error


def test_reject_missing_title_content_or_why():
    result = apply_skill_ops(
        "",
        ops_payload([{"op": "add", "rule": {"title": "Missing fields", "content": "Only content"}}]),
    )
    assert not result.ok
    assert "rule.why" in result.error


def test_reject_update_or_delete_for_nonexistent_id():
    update_result = apply_skill_ops(
        "",
        ops_payload(
            [
                {
                    "op": "update",
                    "id": "001",
                    "rule": {
                        "title": "No target",
                        "content": "This cannot update a missing rule.",
                        "why": "There is no existing rule.",
                    },
                }
            ]
        ),
    )
    delete_result = apply_skill_ops("", ops_payload([{"op": "delete", "id": "001"}]))

    assert not update_result.ok
    assert "missing rule 001" in update_result.error
    assert not delete_result.ok
    assert "missing rule 001" in delete_result.error


def test_reject_critic_provided_id_on_add():
    result = apply_skill_ops(
        "",
        ops_payload(
            [
                {
                    "op": "add",
                    "rule": {
                        "id": "001",
                        "title": "Bad add",
                        "content": "The critic should not choose ids.",
                        "why": "Ids are assigned by the system.",
                    },
                }
            ]
        ),
    )

    assert not result.ok
    assert "must not provide rule.id" in result.error


def test_reject_update_where_rule_id_does_not_match_target_id():
    result = apply_skill_ops(
        sample_skill(),
        ops_payload(
            [
                {
                    "op": "update",
                    "id": "001",
                    "rule": {
                        "id": "002",
                        "title": "Bad update",
                        "content": "Rule ids cannot change during an update.",
                        "why": "Updates must preserve the target rule identity.",
                    },
                }
            ]
        ),
    )

    assert not result.ok
    assert "rule.id must match target id" in result.error


def test_legacy_bullets_canonicalize_to_rules():
    text = "## SpreadsheetBench Rules\n\n- Save the workbook.\n- Write direct values.\n"
    canonical = canonicalize_skill_text(text)

    assert "[RULE 001] Save the workbook" in canonical
    assert "why: Migrated from the legacy initial skill list." in canonical


def test_apply_ops_returns_rules_before_and_after():
    result = apply_skill_ops(sample_skill(), ops_payload([{"op": "delete", "id": "001"}]))
    assert result.ok
    assert [r.id for r in result.rules_before] == ["001", "002"]
    assert [r.id for r in result.rules_after] == ["002"]
