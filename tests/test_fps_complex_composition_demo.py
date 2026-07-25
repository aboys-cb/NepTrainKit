from tools.demo_fps_complex_composition import median_metrics, run_repeated, win_rate


def test_stratified_local_fps_improves_minority_environment_coverage():
    results = run_repeated(seed=20260721, repeats=12)
    baseline = median_metrics(results, "global mean FPS")
    proposed = median_metrics(results, "stratified local FPS")

    assert proposed.rmse < 0.95 * baseline.rmse
    assert proposed.minority_extreme_rmse < 0.8 * baseline.minority_extreme_rmse
    assert proposed.local_coverage_radius_p95 < 0.9 * baseline.local_coverage_radius_p95
    assert win_rate(results, "stratified local FPS", "minority_extreme_rmse") == 1.0
