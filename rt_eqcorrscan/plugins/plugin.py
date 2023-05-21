"""
Default handling of rt_eqcorrscan plugins.
"""

import logging
import subprocess

# Dict of registered plugins - no other plugins will be callable
# entry point must point to script to run the plugin. Plugin should run as a
# continuously running while loop until killed.

# Plugin death is communicated by a kill_file

REGISTERED_PLUGINS = {
    "reloc": "relocation/relocator.py"
}

Logger = logging.getLogger(__name__)

def run_plugin(
    plugin: str,
    plugin_args: list,
):
    plugin_path = REGISTERED_PLUGINS.get(plugin)
    if plugin_path is None:
        raise FileNotFoundError(f"plugin: {plugin} is not registered")
    # Start plugin subprocess
    plugin_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), plugin_path)
    _call = [
        "python", script_path,
    ]
    _call.extend(plugin_args)

    Logger.info("Running `{call}`".format(call=" ".join(_call)))
    proc = subprocess.Popen(_call)

    return proc
