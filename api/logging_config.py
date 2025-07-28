import logging
import os
from pathlib import Path

class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()

def setup_logging(format: str = None):
    """
    Configure logging for the application.
    Reads LOG_LEVEL and LOG_FILE_PATH from environment (defaults: INFO, logs/application.log).
    Ensures log directory exists, and configures both file and console handlers with different formats.
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level and file path from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get(
        "LOG_FILE_PATH", str(default_log_file)))

    # ensure log_file_path is within the project's logs directory to prevent path traversal
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatters for different output types
    detailed_format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    console_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create formatters
    file_formatter = logging.Formatter(detailed_format)
    console_formatter = logging.Formatter(console_format)
    
    # Create handlers
    file_handler = logging.FileHandler(resolved_path)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(IgnoreLogChangeDetectedFilter())
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured - Level: {log_level_str}, File: {resolved_path.name}")
    logger.debug(f"Detailed logging path: {resolved_path}")
