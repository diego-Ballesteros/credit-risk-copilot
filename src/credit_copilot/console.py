"""Making the standard streams able to carry the characters this project prints.

**Why this is a module and not three lines in whatever script needs it.** A Windows console
defaults to the cp1252 code page, which can encode about 250 characters. Anything outside
that set - an emoji, a box-drawing character, a Greek letter - raises `UnicodeEncodeError`
at the moment it is *written*, not when it is computed. That timing is the whole problem:
the work is already done, the numbers are already correct, and the process dies on the
line that was supposed to report them. An exit code produced that way says "this failed"
about something that succeeded.

**Why it does not live in `models/tracking.py` any more.** It was first written there,
because MLflow was what exposed the bug - it prints two emoji-prefixed lines at the end of
every run and offers no way to turn them off. Putting the fix inside `configure_mlflow`
tied it to MLflow twice over: a script that printed a non-ASCII character without talking
to a tracking server never got the fix, and a script that wanted the fix had to import a
module that imports `mlflow`. This module imports nothing but `sys` and `contextlib`, so
any script can call it for the price of a standard-library import.

The failure it prevents is not hypothetical here: `scripts/measure_null_distribution.py`
was written with a `±` in its output and, being the one script that never touches MLflow,
was the one script whose output came out in a different encoding from every other.
"""

import contextlib
import sys

__all__ = ["enable_unicode_console"]


def enable_unicode_console() -> None:
    """Reconfigure stdout and stderr to UTF-8, so printing cannot fail on a character.

    Idempotent, and a no-op on a console that already speaks UTF-8, so calling it at the
    top of every script costs nothing and removes a class of failure entirely. Call it
    before the first print rather than defensively around one - the point is that no caller
    should have to know which characters are safe.

    If a stream refuses to be reconfigured at all - it is not a text wrapper, or it is
    already detached - the fallback only makes encoding errors non-fatal. A character that
    cannot be represented then arrives mangled instead of raising, which is the right
    trade: a report with a wrong glyph is readable, and a report that raised is not.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="backslashreplace")
