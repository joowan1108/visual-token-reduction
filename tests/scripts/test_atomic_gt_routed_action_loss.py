from experiments.atomic.eval_gt_routed_action_loss import summarize


def test_summarize_reports_skill_macro_and_sample_micro_means():
    result = summarize({0: [1.0, 3.0], 1: [2.0]})

    assert result["per_skill"]["pick"]["count"] == 2
    assert result["per_skill"]["pick"]["mean"] == 2.0
    assert result["macro_mean"] == 2.0
    assert result["micro_mean"] == 2.0
    assert result["total_samples"] == 3
