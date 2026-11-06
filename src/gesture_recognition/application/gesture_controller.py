"""Decide whether a prediction should become a real keyboard command.

The decision logic (should we act?) is kept separate from the action
(pressing keys), so the safety rules can be tested without a webcam.
"""


class GestureController:
    def __init__(self, command_keys, confidence_threshold=0.8,
                 cooldown_seconds=1.5, start_enabled=False, dry_run=True):
        self.command_keys = command_keys
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.enabled = start_enabled
        self.dry_run = dry_run
        self.last_command_time = None

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def should_act(self, command, confidence, now):
        """Return (True, reason) if the command should run, else (False, reason)."""
        if not self.enabled:
            return False, "disabled"
        if command == "no_action":
            return False, "no gesture"
        if not self.command_keys.get(command):
            return False, "no keys mapped"
        if confidence < self.confidence_threshold:
            return False, "low confidence"
        if (self.last_command_time is not None
                and now - self.last_command_time < self.cooldown_seconds):
            return False, "cooldown"
        return True, "ok"

    def handle(self, command, confidence, now):
        """Check the rules and, if they pass, run the command."""
        act, reason = self.should_act(command, confidence, now)
        if act:
            self.last_command_time = now
            self.execute(command)
        return act, reason

    def execute(self, command):
        keys = self.command_keys[command]
        if self.dry_run:
            print(f"[dry run] {command} -> {'+'.join(keys)}")
            return
        import pyautogui  # imported here so tests do not need a display

        pyautogui.hotkey(*keys)
        print(f"[sent] {command} -> {'+'.join(keys)}")
