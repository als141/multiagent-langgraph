"""Logging configuration for the multi-agent system."""

import structlog
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_rich: bool = True
) -> None:
    """Set up structured logging with rich formatting."""
    
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    
    # Ensure log directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                rich_tracebacks=True,
                markup=True
            ) if enable_rich else logging.StreamHandler(),
            logging.FileHandler(file_path, encoding='utf-8')
        ]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback
            ) if settings.debug_mode else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given name."""
    return structlog.get_logger(name)


# Initialize logging on import
setup_logging()