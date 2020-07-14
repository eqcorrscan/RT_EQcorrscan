# 0.1.2
- Added rteqcorrscan-bench script to provide a means of benchmarking the users
  set configuration - provides a way to check the limits of real-time processing
  on your machine.
- rt_eqcorrscan.reactor.scaling_relations:
  - Add context-manager to set scaling relationship used
    for region look-ups.
- rt_eqcorrscan.reactor:
  - Checks for trigger_events in already running template
    and does not trigger on events already running
  - Controls stopping of completed spin_ups
- rt_eqcorrscan.streaming:
  - Add a lock on `on_data` method and `stream` property for threadsafety.
  - Add wait-loop for access to hdf5 file underlying WaveBank for process IO
    safety.
- rt_eqcorrscan.listener:
  - Add lock on `old_events` property for threadsafe access.
- rt_eqcorrscan.rt_match_filter:
  - Backfilling now done in a separate concurrent process to avoid slowing
    down the real-time processing.
  - Templates are reshaped prior to detection use to avoid time-cost associated
    with re-doing this every iteration.

# 0.0.2
- First full release. Everything is new!
