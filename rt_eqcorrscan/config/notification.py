"""
Notification of specific events.

Author
    Calum J Chamberlain
License
    GPL v3.0
"""

import notifiers
import logging


Logger = logging.getLogger(__name__)


class Notifier(object):
    """
    Class to notify by email of certain events.

    Parameters
    ----------
    level
        Level from 0-5, notification events at a level higher than `level`
        will result in emails. Set to 5 by default, results in no emails.
    service
        Service supported by notifiers
    formatter
        Format string for message
    service_kwargs
        Arguments as required for your chosen service.
    """
    def __init__(
        self,
        level: int = 5,
        service: str = None,
        formatter: str = None,
        service_kwargs: dict = None,
    ):
        if level > 5:
            level = 5
        self.level = level
        self.service = notifiers.get_notifier(service)
        self.formatter = formatter
        self.service_kwargs = service_kwargs

    def __repr__(self):
        return "Notifier(level={0}, service={1}, formatter={2})".format(
            self.level, self.service, self.formatter)

    def notify(self, message: str, level: int) -> None:
        if level >= self.level and self.service is not None:
            try:
                self.service.notify(message=message, **self.service_kwargs)
            except Exception as e:
                Logger.error(e)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
