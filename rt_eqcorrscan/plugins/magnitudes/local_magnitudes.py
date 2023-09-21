"""
Script and functions to compute local magnitudes for events.

Steps:
1. Watch directory for detections
2. If new detections, read in to memory (event and waveform)
3. Run mag-calc.amp_pick_event
4. Compute magnitude according to some defined scale
5. Output to csv and json catalogue

Designed to be run as continuously running subprocess managed by rt_eqcorrscan
"""


def local_magnitudes():
    pass


if __name__ == "__main__":
    local_magntiudes()