"""Decide whether a prediction should become a real keyboard command.

The decision logic (should we act?) is kept separate from the action
(pressing keys), so the safety rules can be tested without a webcam.

Safety rules, checked in this order:
1. Commands must be switched on (E key).
2. "No gesture" never fires.
3. Re-arm: after a command fires, the hand must rest first. Nothing else
   fires until the model has seen "No gesture" for `rearm_seconds` in a
   row. This stops the return movement after a swipe from counting as a
   new swipe, and stops a held gesture (like Stop Sign) from refiring.
4. The command must have keys mapped.
5. Confidence must be at or above the threshold.
6. Cooldown: a minimum gap between any two commands.
"""


class GestureController:
    def __init__(self, command_keys, confidence_threshold=0.8,
                 cooldown_seconds=1.5, start_enabled=False, dry_run=True,
                 rearm_seconds=0.0):
        self.command_keys = command_keys
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.enabled = start_enabled
        self.dry_run = dry_run
        self.rearm_seconds = rearm_seconds  # 0 turns the re-arm rule off
        self.last_command_time = None
        self.armed = True        # ready to fire
        self.rest_start = None   # when the current run of "No gesture" began

    def toggle(self):
        self.enabled = not self.enabled
        self.armed = True        # start fresh when switched on or off
        self.rest_start = None
        return self.enabled

    def update_rest(self, command, now):
        """Track how long the model has seen "No gesture" without a break."""
        if command != "no_action":
            self.rest_start = None
            return
        if self.rest_start is None:
            self.rest_start = now
        if now - self.rest_start >= self.rearm_seconds:
            self.armed = True

    def should_act(self, command, confidence, now, simulated=False):
        """Return (True, reason) if the command should run, else (False, reason)."""
        if not self.enabled:
            return False, "disabled"
        if command == "no_action":
            return False, "no gesture"
        if not simulated and self.rearm_seconds > 0 and not self.armed:
            return False, "waiting for rest"
        if not self.command_keys.get(command):
            return False, "no keys mapped"
        if confidence < self.confidence_threshold:
            return False, "low confidence"
        if (self.last_command_time is not None
                and now - self.last_command_time < self.cooldown_seconds):
            return False, "cooldown"
        return True, "ok"

    def handle(self, command, confidence, now, simulated=False):
        """Check the rules and, if they pass, run the command.

        simulated=True is used for the number keys. Those skip the re-arm
        rule, because a fake key press has no real hand that can rest.
        """
        if not simulated:
            self.update_rest(command, now)
        act, reason = self.should_act(command, confidence, now, simulated)
        if act:
            self.last_command_time = now
            if not simulated and self.rearm_seconds > 0:
                self.armed = False   # wait for the hand to rest again
                self.rest_start = None
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
