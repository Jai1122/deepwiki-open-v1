import logging
import os
from pathlib import Path

class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()

class IgnoreVerboseTextSplittingFilter(logging.Filter):
    """Filter out verbose text splitting messages from adalflow components."""
    def filter(self, record: logging.LogRecord):
        message = record.getMessage()
        # Filter out common verbose text splitting messages
        verbose_patterns = [
            "Text split by",
            "Splitting text with",
            "split by '' into",
            "Splitting text with ''",
            "Created chunk",
            "Split into chunks"
        ]
        return not any(pattern in message for pattern in verbose_patterns)

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
    file_handler.addFilter(IgnoreVerboseTextSplittingFilter())
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(IgnoreLogChangeDetectedFilter())
    console_handler.addFilter(IgnoreVerboseTextSplittingFilter())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure specific loggers to reduce verbosity
    # Reduce adalflow TextSplitter verbosity (moves "Text split by..." and "Splitting text with" logs to WARNING)
    adalflow_data_process_logger = logging.getLogger("adalflow.components.data_process")
    adalflow_data_process_logger.setLevel(logging.WARNING)
    
    # Reduce adalflow TextSplitter specifically (more targeted)
    adalflow_text_splitter_logger = logging.getLogger("adalflow.components.data_process.text_splitter")
    adalflow_text_splitter_logger.setLevel(logging.WARNING)
    
    # Reduce other adalflow component verbosity
    adalflow_logger = logging.getLogger("adalflow")
    if log_level > logging.DEBUG:  # Only if we're not in DEBUG mode
        adalflow_logger.setLevel(logging.WARNING)

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging configured - Level: {log_level_str}, File: {resolved_path.name}")
    logger.debug(f"Detailed logging path: {resolved_path}")
    logger.debug("🔇 Adalflow verbose logging reduced to WARNING level")
