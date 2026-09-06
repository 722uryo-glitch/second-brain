import contextvars
import inspect

_reporter = contextvars.ContextVar("second_brain_progress_reporter", default=None)


def set_reporter(reporter):
    """Install a per-task progress reporter and return a token for reset."""
    return _reporter.set(reporter)


def reset_reporter(token):
    _reporter.reset(token)


async def report(step: str, message: str = "", data=None):
    reporter = _reporter.get()
    if reporter is None:
        return
    try:
        result = reporter(str(step or ""), str(message or ""), data or {})
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        # Progress reporting must never break the actual job.
        print(f"[PROGRESS] reporter failed: {e}")
