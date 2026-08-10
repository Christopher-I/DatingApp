import random
from datetime import datetime

from copilot.brain.scheduler import ShadowbanMonitor, SwipeScheduler
from copilot.config import SchedulerConfig


def _at(hour, minute=0):
    return datetime(2026, 8, 10, hour, minute, 0)


def test_active_window():
    sched = SwipeScheduler(SchedulerConfig(active_start_hour=9, active_end_hour=23))
    assert sched.is_active_window(_at(14)) is True
    assert sched.is_active_window(_at(3)) is False   # overnight -> never
    assert sched.is_active_window(_at(23)) is False  # end is exclusive


def test_daily_cap_enforced():
    cfg = SchedulerConfig(daily_cap=3, active_start_hour=0, active_end_hour=23)
    sched = SwipeScheduler(cfg)
    now = _at(12)
    for _ in range(3):
        assert sched.can_swipe_now(now) is True
        sched.record_swipe(now, liked=False)
    assert sched.can_swipe_now(now) is False
    assert sched.swipes_remaining_today(now) == 0


def test_daily_counters_reset_next_day():
    cfg = SchedulerConfig(daily_cap=2, active_start_hour=0, active_end_hour=23)
    sched = SwipeScheduler(cfg)
    day1 = datetime(2026, 8, 10, 12)
    sched.record_swipe(day1, liked=True)
    sched.record_swipe(day1, liked=False)
    assert sched.swipes_remaining_today(day1) == 0
    day2 = datetime(2026, 8, 11, 12)
    assert sched.swipes_remaining_today(day2) == 2


def test_like_ratio_guard():
    cfg = SchedulerConfig(max_like_ratio=0.5, active_start_hour=0, active_end_hour=23)
    sched = SwipeScheduler(cfg)
    now = _at(12)
    assert sched.can_like(now) is True     # first swipe always allowed
    sched.record_swipe(now, liked=True)    # 1 like / 1 swipe
    # projected ratio after another like = 2/2 = 1.0 > 0.5 -> blocked
    assert sched.can_like(now) is False
    sched.record_swipe(now, liked=False)   # 1/2
    # projected = 2/3 = 0.66 > 0.5 -> still blocked
    assert sched.can_like(now) is False
    sched.record_swipe(now, liked=False)   # 1/3
    # projected = 2/4 = 0.5 <= 0.5 -> allowed
    assert sched.can_like(now) is True


def test_delay_within_bounds():
    cfg = SchedulerConfig(delay_min_seconds=2.0, delay_max_seconds=5.0)
    sched = SwipeScheduler(cfg, rng=random.Random(0))
    for _ in range(50):
        d = sched.next_delay_seconds()
        assert 2.0 <= d <= 5.0


def test_session_plan_within_bounds():
    cfg = SchedulerConfig(session_min_swipes=30, session_max_swipes=50,
                          session_span_min_minutes=15, session_span_max_minutes=25)
    sched = SwipeScheduler(cfg, rng=random.Random(0))
    plan = sched.plan_session()
    assert 30 <= plan.swipe_count <= 50
    assert 15 <= plan.span_minutes <= 25


def test_shadowban_monitor_detects_collapse():
    cfg = SchedulerConfig(shadowban_drop_ratio=0.25, shadowban_window_days=3)
    mon = ShadowbanMonitor(cfg)
    # Three healthy days at ~10% match rate.
    mon.record_day("2026-08-07", swipes=50, matches=5)
    mon.record_day("2026-08-08", swipes=50, matches=5)
    mon.record_day("2026-08-09", swipes=50, matches=5)
    # Then a collapse: 50 swipes, 0 matches.
    mon.record_day("2026-08-10", swipes=50, matches=0)
    assert mon.is_shadowbanned() is True


def test_shadowban_monitor_not_triggered_when_healthy():
    cfg = SchedulerConfig(shadowban_drop_ratio=0.25, shadowban_window_days=3)
    mon = ShadowbanMonitor(cfg)
    mon.record_day("2026-08-08", swipes=50, matches=5)
    mon.record_day("2026-08-09", swipes=50, matches=5)
    mon.record_day("2026-08-10", swipes=50, matches=4)
    assert mon.is_shadowbanned() is False
