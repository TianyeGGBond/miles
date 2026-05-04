import asyncio
import os

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import init_tracking


def _assert_standalone_entry(args) -> None:
    """F11 standalone fail-fast — refuse to run in RLix mode via this entry point.

    The RLix entry driver is `examples/rlix/run_miles_rlix.py`. Standalone code path
    (this file) MUST NOT be reached when `RLIX_CONTROL_PLANE=rlix` is set; doing so
    would silently bypass scheduler-managed sleep/wake / partial overlap and degrade
    to full-broadcast weight sync.

    Also fail fast on the architectural precondition `train_devices ⊂ infer_devices`
    when the user has accidentally configured a partial-overlap topology under the
    standalone entry: that combination requires C20 router admission + cache-owner
    sync, neither of which standalone has.
    """
    if os.environ.get("RLIX_CONTROL_PLANE") == "rlix":
        raise RuntimeError(
            "RLIX_CONTROL_PLANE=rlix is set but train_async.py is the standalone "
            "entry. Use `examples/rlix/run_miles_rlix.py` for RLix-managed "
            "scheduling, or unset RLIX_CONTROL_PLANE to run standalone."
        )

    # Partial-overlap topology guard. Standalone path assumes either fully colocated
    # (`actor_train == actor_infer`) or fully disjoint allocation; partial overlap
    # without RLix is unsafe.
    actor_gpus = int(getattr(args, "actor_num_nodes", 1)) * int(getattr(args, "actor_num_gpus_per_node", 0))
    rollout_gpus = getattr(args, "rollout_num_gpus", None)
    if rollout_gpus is not None and actor_gpus > 0 and 0 < int(rollout_gpus) < actor_gpus:
        raise RuntimeError(
            "Partial-overlap topology detected (actor_train ⊂ actor_infer-equivalent) "
            "without RLIX_CONTROL_PLANE=rlix. This configuration requires the RLix "
            "scheduler entry driver — silent OOM via full-broadcast otherwise."
        )


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
async def train(args):
    _assert_standalone_entry(args)
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    # async train loop.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_curr_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.use_critic:
                await critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                await rollout_manager.save.remote(rollout_id)

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            await actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train(args))
