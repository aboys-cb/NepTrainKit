from tools.benchmark_card_operations import build_cases, build_count50_cases


def test_card_benchmark_profiles_follow_current_operation_params() -> None:
    default_names = {case.name for case in build_cases()}
    count50_names = {case.name for case in build_count50_cases()}

    assert "FPSFilter:20->5" in default_names
    assert "FPSFilter:50->10" in count50_names
