# simulation-testing
- Fix threadlock when backfillers attempt to decluster in multithreaded mode
  (requires update to EQcorrscan)
- Share backfiller stream as file rather than in-memory to reduce memory use in backfilling
- Backfillers operate in subprocesses rather than multiprocessing Processes

# email-support
- Attempted to add email support, but gmail authentification kept breaking
- Switched to using notifiers

# 0.1.4
- Keep track of data stream coming in from streamer, and restart if stream 
  goes stale (See PR #12).
- Add threading exit event to streamers to allow internal stopping of threads
  (See PR #12).
- Move streamers to processes for Linux (not yet supported for Windows and 
  MacOS) and migrate wavebank handling to RTTribe (#13)

# 0.1.3
- Support Windows
- Handle detections more regularly while backfilling
- Deploy scripts as "entry-points"

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
  - Added a hypocentral_separation parameter for declustering - this also has
    a default config value set for it (30km)

# 0.0.2
- First full release. Everything is new!
