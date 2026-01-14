""" Basic notification handling. Only tested for pushover """

import logging
import platform
import datetime

from notifiers import get_notifier

Logger = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        service: str = "pushover",
        default_args: dict = {"user": "USER", "token": "TOKEN"},
        *args, **kwargs
    ):
        self.service = get_notifier(service)
        self.default_args = default_args

    def notify(self, content: str):
        content = (f"RTEQcorrscan message at {datetime.datetime.now()} "
                   f"from {platform.node()}: {content}")
        try:
            self.service.notify(message=content, **self.default_args)
        except Exception as e:
            Logger.error(f"Could not send notification due to {e}")
        else:
            Logger.info(f"Notification sent to {self.service}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
