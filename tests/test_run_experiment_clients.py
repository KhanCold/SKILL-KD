from argparse import Namespace

from pact import run_experiment


def _args(**overrides):
    values = {
        "mode": "evolve+test",
        "base_url": "https://base.example/v1",
        "student_base_url": "https://student.example/v1",
        "teacher_base_url": "https://teacher.example/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "student_model": "student-model",
        "teacher_model": "critic-model",
    }
    values.update(overrides)
    return Namespace(**values)


def test_evolve_mode_initializes_student_teacher_and_critic(monkeypatch):
    constructed = []

    class FakeClient:
        def __init__(self, config):
            constructed.append((config.model, config.base_url, config.api_key_env))

    monkeypatch.setattr(run_experiment, "LLMClient", FakeClient)

    student, teacher, critic = run_experiment.make_llm_clients(_args())

    assert student is not None
    assert teacher is critic
    assert constructed == [
        ("student-model", "https://student.example/v1", "DASHSCOPE_API_KEY"),
        ("critic-model", "https://teacher.example/v1", "DASHSCOPE_API_KEY")
    ]


def test_test_mode_initializes_only_student_without_teacher(monkeypatch):
    constructed = []

    class FakeClient:
        def __init__(self, config):
            constructed.append((config.model, config.base_url, config.api_key_env))

    monkeypatch.setattr(run_experiment, "LLMClient", FakeClient)

    student, teacher, critic = run_experiment.make_llm_clients(_args(mode="test", teacher_model=None))

    assert student is not None
    assert teacher is None
    assert critic is None
    assert constructed == [
        ("student-model", "https://student.example/v1", "DASHSCOPE_API_KEY")
    ]


def test_make_llm_clients_uses_skill_kd_base_url(monkeypatch):
    constructed = []

    class FakeClient:
        def __init__(self, config):
            constructed.append((config.model, config.base_url, config.api_key_env))

    monkeypatch.setenv("SKILL_KD_BASE_URL", "https://skill-kd.example/v1")
    monkeypatch.setattr(run_experiment, "LLMClient", FakeClient)

    student, teacher, critic = run_experiment.make_llm_clients(
        _args(base_url=None, student_base_url=None, teacher_base_url=None)
    )

    assert student is not None
    assert teacher is critic
    assert constructed == [
        ("student-model", "https://skill-kd.example/v1", "DASHSCOPE_API_KEY"),
        ("critic-model", "https://skill-kd.example/v1", "DASHSCOPE_API_KEY"),
    ]
