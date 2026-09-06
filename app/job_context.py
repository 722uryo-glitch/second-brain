"""Task-local durable execution hooks; ordinary chat has no job context."""
import asyncio
import contextvars
import inspect

current = contextvars.ContextVar("job_execution", default=None)


class JobStopped(asyncio.CancelledError):
    """Must bypass broad Exception fallback handlers in research/model code."""


class YieldForUser(JobStopped):
    pass


class BudgetExceeded(JobStopped):
    pass


async def checkpoint(key, inputs, operation, kind="checkpoint"):
    context = current.get()
    if context is not None:
        return await context.step(key, inputs, operation, kind)
    value = operation()
    return await value if inspect.isawaitable(value) else value


def reserve_call(*, retry=False, output_tokens=0):
    context = current.get()
    if context is not None:
        context.reserve(retry=retry, output_tokens=output_tokens)


async def gather_owned(*operations):
    """Cancel and await siblings even when one child fails or cancels itself."""
    tasks = [asyncio.create_task(op) for op in operations]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
