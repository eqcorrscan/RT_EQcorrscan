"""
Basic email handling - used for critical alerts.

Note that this mailer isn't very secure! Do not use your main email address!
Passwords are handled by the default keychain on your computer.
"""

import keyring
import getpass
import smtplib
import ssl


class Mailer:
    def __init__(
        self,
        address: str = None,
        username: str = None,
        sendto: str = None,
        server: str = "smtp.gmail.com",
    ):
        self.server = server
        self.address = address
        self.username = username
        self.sendto = sendto
        if address and username and sendto and server:
            # Allow a dummy client
            self._can_send = True
        # Check for password on init to ensure user doesn't get prompted all the time!
        _ = self.password
        del _

    @property
    def password(self):
        if not self._can_send:
            return None
        _password = keyring.get_password(
            service_name=self.address, username=self.username)
        if _password is None:
            print("No stored password, enter password below:")
            _password = getpass.getpass()
            # Store password for later use
            keyring.set_password(
                service_name=self.address, username=self.username,
                password=_password)
        return _password

    def sendmail(self, content: str, type: str = "INFO"):
        """ Send via SSL """
        if not self._can_send:
            # Allow an empty dummy.
            return
        # Add in a subject line
        content = f"Subject: RTEQcorrscan {type}\n\n" + content
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.server, port=465, context=context) as server:
            server.login(self.address, self.password)
            server.sendmail(self.address, self.sendto, content)
        return