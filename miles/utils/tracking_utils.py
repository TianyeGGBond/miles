import logging

from miles.utils.tensorboard_utils import _TensorboardAdapter

from . import wandb_utils
from .prometheus_utils import get_prometheus, init_prometheus

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)

    if args.use_prometheus:
        init_prometheus(args, start_server=primary)


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        # Lazy import: ambient protobuf 3.20.x (pinned by rlix) breaks the
        # eager wandb import chain (`wandb.proto.wandb_telemetry_pb2.Imports`
        # was generated against a newer protobuf). Importing wandb only
        # inside the gated branch keeps the module loadable when nothing
        # actually opted into wandb tracking.
        import wandb  # noqa: PLC0415

        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    if args.use_prometheus:
        prom = get_prometheus()
        assert prom is not None, (
            "Prometheus collector is not initialized; ensure init_tracking(..., primary=...) ran on the "
            "driver and workers can resolve the miles_prometheus_collector Ray actor."
        )
        prom.update.remote(metrics)
