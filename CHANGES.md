# Current
- rt_eqcorrscan.reactor.scaling_relations:
  - Add context-manager to set scaling relationship used
    for region look-ups.
- rt_eqcorrscan.reactor:
  - Checks for trigger_events in already running template
    and does not trigger on events already running
  - Controls stopping of completed spin_ups
- rt_eqcorrscan.streaming:
  - Add a lock on `on_data` method and `stream` property for threadsafety.

# 0.0.2
- First full release. Everything is new!
