from miles.utils.misc import get_current_node_ip, get_free_port


class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    # F26 / scope F26: dynamic NCCL master addr+port lookup used by
    # MilesModelUpdateService._build_plan when the coordinator drives a
    # selective sync against the cache_owner actor. Distinct from
    # ``get_master_addr_and_port`` which exposes the actor's pre-existing
    # static MASTER_ADDR/PORT — those belong to the actor's own
    # init_process_group and reusing them for a dynamic broadcast group
    # would collide. ``start_port=20000`` keeps the dynamic range above
    # the train-side rendezvous default (10000) by convention.
    def get_node_ip(self) -> str:
        return get_current_node_ip()

    def get_free_port(self, start_port: int = 20000, consecutive: int = 1) -> int:
        return get_free_port(start_port=start_port, consecutive=consecutive)
