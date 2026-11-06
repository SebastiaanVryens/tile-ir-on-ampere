from tilebench.__main__ import plan_for
from tilebench.targets import get_target


def winner(target: str, workload: str, size: list[int]) -> dict[str, object]:
    return {"target": target, "workload": workload, "size": size, "config": {"block": 1024}}


def test_a_plan_holds_only_the_winners_of_that_target_and_workload() -> None:
    winners = [
        winner("cutile", "layernorm", [16, 1024]),
        winner("triton-ptx", "layernorm", [16, 1024]),
        winner("cutile", "vector_add", [1024]),
    ]
    plan = plan_for(winners, get_target("cutile"), "layernorm")
    assert plan == [[[16, 1024], {"block": 1024}]]


def test_a_target_without_winners_gets_an_empty_plan() -> None:
    assert plan_for([], get_target("torch"), "layernorm") == []
