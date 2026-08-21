import json

from cs336_basics.experiment import ExperimentLogger


class _FakeRun:
    def __init__(self):
        self.defined = []
        self.logged = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.kwargs = None
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def test_experiment_logger_records_local_and_wandb_metrics(tmp_path):
    fake_wandb = _FakeWandb()
    logger = ExperimentLogger(
        tmp_path,
        {"batch_size": 4},
        wandb_project="assignment1",
        wandb_run_name="test-run",
        wandb_mode="offline",
        wandb_module=fake_wandb,
    )
    metric = logger.log("train", 7, 1.25, learning_rate=3e-4)
    logger.log("validation", 7, 1.5)
    logger.finish()

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["gradient_step"] == 7
    assert records[0]["loss"] == 1.25
    assert records[0]["wallclock_seconds"] >= 0
    assert metric["learning_rate"] == 3e-4
    assert fake_wandb.kwargs["config"] == {"batch_size": 4}
    assert fake_wandb.run.logged[0][0]["train/loss"] == 1.25
    assert fake_wandb.run.logged[1][0]["validation/loss"] == 1.5
    assert fake_wandb.run.finished
