from src.progress import ProgressTracker


def test_progress_tracker_tracks_overall_and_current_tasks():
    tracker = ProgressTracker()

    overall_task = tracker.add_overall_task("Overall", total=10)
    current_task = tracker.add_current_task("Current", total=4)

    assert overall_task is not None
    assert current_task is not None

    tracker.update_overall(3)
    tracker.update_current(2)

    assert tracker.progress.tasks[0].completed == 3
    assert tracker.progress.tasks[1].completed == 2


def test_status_line_is_created_once_and_updated():
    tracker = ProgressTracker()
    tracker.add_overall_task("Overall", total=10)
    tracker.set_status("first")
    tracker.set_status("second: 3 pts/round")
    statuses = [t for t in tracker.progress.tasks if t.id == tracker.status_task_id]
    assert len(statuses) == 1 and statuses[0].description == "second: 3 pts/round"
