"""ChannelAgent entrypoint.

Currently brings up configuration and the database, which is enough to
prove the application boots correctly inside the container (fails fast
on a missing ENCRYPTION_KEY, creates DB tables on first run). Channel
adapters (Telegram/Email/Matrix)
will be wired in here once they exist; there is nothing to route
messages through yet.
"""

import asyncio
import logging
import sys
from collections.abc import Coroutine
from pathlib import Path

from app.api.deps import MIN_API_KEY_LENGTH, api_key_is_acceptable
from app.config import get_settings
from app.db.bootstrap import bootstrap_admin_from_env
from app.db.session import init_db, session_scope
from app.graph import close_graph, get_graph
from app.health import heartbeat
from app.logging_setup import add_file_logging, configure_logging, quiet_job_polls
from app.security.permissions import harden_process, warn_about_loose_application_files
from app.settings_rules import api_exposure_problem

configure_logging()
logger = logging.getLogger("channelagent")


async def _supervised(name: str, hint: str, component: Coroutine) -> None:
    """Run one component (an adapter, the Admin API). If it fails it is logged
    once, with what to check, and stays stopped: the others keep running.
    A Telegram token that Telegram rejects used to end the whole process, API
    included. uvicorn ends with SystemExit when it cannot bind its port, which
    is a failure of that component too, so it is caught here. A shutdown
    (CancelledError, KeyboardInterrupt) is not a failure and goes through.
    """
    try:
        await component
    except (Exception, SystemExit):
        logger.error(
            "%s stopped and will stay stopped; the other components keep running. %s",
            name,
            hint,
            exc_info=True,
        )


# Files a container runtime creates (Docker, Podman): inside the image the API listens on
# 0.0.0.0 by design, and what is reachable is the published port, which start.sh checks.
CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))


def exposure_refusal(settings) -> str | None:
    """Why the Admin API must not start on API_SERVER_HOST, or None: a non-loopback
    address outside a container needs API_REMOTE=tls-proxy."""
    if not settings.api_server_key or any(marker.exists() for marker in CONTAINER_MARKERS):
        return None
    return api_exposure_problem("API_SERVER_HOST", settings.api_server_host, settings.api_remote)


async def main() -> int:
    harden_process()
    settings = get_settings()
    refusal = exposure_refusal(settings)
    if refusal:
        logger.error("Not started: %s.", refusal)
        return 2
    await init_db()
    async with session_scope() as session:
        await bootstrap_admin_from_env(session)
    await get_graph()  # opens (and creates) the checkpoint database now, not on the first message
    warn_about_loose_application_files()
    logger.info("ChannelAgent started. LLM gateway: %s", settings.llama_server_url)

    tasks = []
    if settings.telegram_bot_token:
        from app.channels.telegram import run_telegram_adapter

        tasks.append(
            asyncio.create_task(
                _supervised(
                    "Telegram adapter",
                    "Check TELEGRAM_BOT_TOKEN (if it was revoked, issue a new one with BotFather "
                    "and run ./start.sh --config TELEGRAM_BOT_TOKEN=...) and the network.",
                    run_telegram_adapter(),
                )
            )
        )
    else:
        logger.info("TELEGRAM_BOT_TOKEN not set — Telegram adapter disabled.")

    if settings.email_imap_host and settings.email_username and settings.email_password:
        from app.channels.email import run_email_adapter

        tasks.append(
            asyncio.create_task(
                _supervised(
                    "Email adapter",
                    "Check EMAIL_IMAP_HOST, EMAIL_USERNAME, EMAIL_PASSWORD and EMAIL_TRIGGER_TAG.",
                    run_email_adapter(),
                )
            )
        )
    else:
        logger.info("Email IMAP/SMTP settings not fully set — Email adapter disabled.")

    # Matrix adapter joins `tasks` here once it exists.

    if not settings.api_server_key:
        logger.info("API_SERVER_KEY not set — Admin API disabled.")
    elif not api_key_is_acceptable(settings.api_server_key):
        logger.error(
            "API_SERVER_KEY is shorter than %s characters — Admin API NOT started. "
            "Generate a long random key, for example: openssl rand -hex 32",
            MIN_API_KEY_LENGTH,
        )
    else:
        import uvicorn

        from app.server import root

        # The Admin API, and the admin UI under /ui.
        config = uvicorn.Config(
            root,
            host=settings.api_server_host,
            port=settings.api_server_port,
            log_level="info",
        )
        # uvicorn has just set its own handlers: its lines go to the log file too.
        add_file_logging(["uvicorn", "uvicorn.access"])
        quiet_job_polls()
        tasks.append(
            asyncio.create_task(
                _supervised(
                    "Admin API",
                    "Check API_SERVER_PORT (already in use?) and API_SERVER_HOST.",
                    uvicorn.Server(config).serve(),
                )
            )
        )
        logger.info(
            "Admin API starting on %s:%s.", settings.api_server_host, settings.api_server_port
        )

    # The scheduled backup runs beside the components, like the heartbeat: it does
    # not count as one, so a process whose adapters and API have all stopped still exits
    # with 1 instead of staying up for its backups alone.
    background = []
    from app.db.session import sqlite_file_path

    if sqlite_file_path(settings.database_url) is not None:
        from app.admin.backup_schedule import run_scheduler

        background.append(
            asyncio.create_task(
                _supervised(
                    "Backup scheduler",
                    "Check the backups directory (space, permissions) and GET /backups/schedule.",
                    run_scheduler(),
                )
            )
        )
    # The scheduled tasks run beside the components too, for the same reason.
    from app.tasks import run_scheduler as run_task_scheduler

    background.append(
        asyncio.create_task(
            _supervised(
                "Task scheduler",
                "Check GET /tasks (last_status, last_error) and GET /tasks/pause.",
                run_task_scheduler(),
            )
        )
    )
    beating = asyncio.create_task(heartbeat(tasks))
    try:
        if not tasks:
            logger.info("No channel adapters are enabled.")
            while True:
                await asyncio.sleep(3600)
        else:
            await asyncio.gather(*tasks)
            # Every component has ended (a failure was logged by _supervised). Ending
            # with 0 would look like a clean stop to a process manager, and the
            # container would sit idle instead of being restarted or flagged.
            logger.error("Every component has stopped (see the errors above): exiting.")
            return 1
    finally:
        await _stop_components(tasks)
        await _stop_components(background)
        beating.cancel()
        # Open MCP connections: closed here, not left to the interpreter's
        # shutdown, which printed an anyio "cancel scope in a different task" traceback.
        from app.mcp.manager import manager as mcp_manager

        await mcp_manager.reset()
        await close_graph()


async def _stop_components(tasks: list[asyncio.Task]) -> None:
    """Cancel every component and wait until each has finished its own cleanup.

    On Ctrl+C, uvicorn catches SIGINT, shuts the API down, then raises SIGINT
    again from inside its still-running task, and asyncio.run cancels main().
    gather then cancels the API task too, and it gives up as soon as one child
    ends cancelled, without waiting for the others: the Telegram adapter was
    still stopping its poller when asyncio.run cancelled every leftover task,
    and its interrupted cleanup was logged as "Telegram adapter stopped".

    A task that is already cancelling is left alone: a second cancel() would
    interrupt the very cleanup this function waits for.
    """
    for task in tasks:
        if not task.done() and not task.cancelling():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        # Ctrl+C: main() has already shut every component down (the finally above);
        # asyncio.run re-raises the interrupt only afterwards, which printed a traceback
        # after a clean stop. 130 is the shell's exit code for SIGINT.
        sys.exit(130)
