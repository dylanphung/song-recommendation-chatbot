"""Remembers the user's name for the session."""
import re

NAME_PATTERN = re.compile(r"(?:my name is|i am|i'm|call me)\s+(\w+)", re.IGNORECASE)


class IdentityManager:
    def __init__(self):
        self.name = None

    def set_name(self, user_input):
        """Take the name from phrases like "my name is Dylan", or treat the whole input as the name."""
        match = NAME_PATTERN.search(user_input)
        self.name = match.group(1) if match else user_input.strip()

    def get_name(self):
        return self.name or "Guest"
