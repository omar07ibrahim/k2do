from pathlib import Path

import pytest

import k2do.cron.service as cron_service
from k2do.cron.service import CronService
from k2do.cron.types import CronJob, CronJobState, CronPayload, CronSchedule


@pytest.mark.asyncio
async def test_every_schedule_uses_previous_slot_anchor(monkeypatch, tmp_path: Path) -> None:
    # start_ms=10000, updated_at_ms=10050, now_ms for scheduling=10050
    times = [10000, 10050, 10050]

    def fake_now_ms() -> int:
        return times.pop(0) if times else 10050

    monkeypatch.setattr(cron_service, "_now_ms", fake_now_ms)

    async def on_job(_: CronJob) -> str | None:
        return "ok"

    service = CronService(store_path=tmp_path / "jobs.json", on_job=on_job)
    service._store = cron_service.CronStore(jobs=[])

    job = CronJob(
        id="job1",
        name="every-1s",
        enabled=True,
        schedule=CronSchedule(kind="every", every_ms=1000),
        payload=CronPayload(kind="agent_turn", message="ping"),
        state=CronJobState(next_run_at_ms=10000),
        created_at_ms=0,
        updated_at_ms=0,
    )

    await service._execute_job(job)

    # No drift: next run is exactly previous slot + interval.
    assert job.state.next_run_at_ms == 11000


def test_add_job_rejects_non_positive_every_interval(tmp_path: Path) -> None:
    service = CronService(store_path=tmp_path / "jobs.json")
    with pytest.raises(ValueError):
        service.add_job(
            name="bad",
            schedule=CronSchedule(kind="every", every_ms=-1),
            message="x",
        )
