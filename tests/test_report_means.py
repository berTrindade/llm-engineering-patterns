from evals.run import TaskOutcome, build_report


def _outcome(task_id, failure_type, latency, prompt, completion, calls, iterations):
    return TaskOutcome(
        task_id=task_id,
        category="simple",
        success=failure_type is None,
        failure_type=failure_type,
        reason=None,
        tool_call_count=calls,
        iterations=iterations,
        recovered_error_count=0,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=latency,
        final_message=None,
        stopped_reason=None,
        trace=[],
    )


def test_model_error_tasks_are_left_out_of_the_means():
    report = build_report(
        [
            _outcome("t1", None, 1000.0, 400, 40, 2, 3),
            _outcome("t2", "model_error", 0.0, 0, 0, 0, 0),
        ]
    )

    assert report["mean_latency_ms"] == 1000.0
    assert report["mean_prompt_tokens"] == 400
    assert report["mean_completion_tokens"] == 40
    assert report["mean_tool_calls"] == 2
    assert report["mean_iterations"] == 3
    assert report["success_count"] == 1
    assert report["failures_by_type"] == {"model_error": 1}
