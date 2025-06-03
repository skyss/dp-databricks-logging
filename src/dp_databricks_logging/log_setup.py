"""Log setup functions for databricks workflows / jobs."""

from __future__ import annotations

import inspect
import logging
import os
import socket
import sys
import traceback
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import requests
from loguru import logger

if TYPE_CHECKING:
    import loguru
    from databricks.sdk.runtime.dbutils_stub import dbutils as dbutils_type
    from pyspark.sql.session import SparkSession


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect it to Loguru."""

    valid_levels = frozenset(
        ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
    )

    def emit(self, record: logging.LogRecord) -> None:
        """Get corresponding Loguru level if it exists."""
        level: str | int
        if record.levelname in self.valid_levels:
            level = logger.level(record.levelname).name
        else:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class LogWrapper:
    """Logs to Datadog."""

    DD_API_TOKEN: str = ""
    cluster_name: str | None = ""
    user_name: str = ""
    logger_id: int | None = None

    @classmethod
    def write_events_to_datadog(cls: type[LogWrapper], event: dict[str, Any]) -> None:
        """Write events to datadog."""
        url = "https://http-intake.logs.datadoghq.eu/api/v2/logs"
        headers = {"DD-API-KEY": LogWrapper.DD_API_TOKEN}
        data = event
        data["service"] = "dp-workflows"
        data["hostname"] = LogWrapper.cluster_name
        data["username"] = LogWrapper.user_name
        env = os.environ.get("environment")  # noqa: SIM112
        data["ddtags"] = f"env:{env},status:{event['level']}"
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

    @classmethod
    def serialize(cls: type[LogWrapper], record: loguru.Record) -> dict[str, Any]:
        """Serialize the JSON log."""
        log = {
            "elapsed": f"{record['elapsed']}",
            "file.name": record["file"].name,
            "file.path": record["file"].path,
            "function": record["function"],
            "level": record["level"].name,
            "line": record["line"],
            "module": record["module"],
            "name": record["name"],
            "process.id": record["process"].id,
            "process.name": record["process"].name,
            "thread.name": record["thread"].name,
            "thread.id": record["thread"].id,
            "time": f"{record['time']}",
            "status": record["level"].name,
            "message": record["message"],
            "logger.thread_name": record["thread"].name,
            "hostname": socket.gethostname(),
        }

        if record["extra"]:
            for key, value in record["extra"].items():
                log[f"extra.{key}"] = value

        if record["exception"] is not None:
            error_data = {
                "error.stack": "".join(
                    traceback.format_exception(
                        record["exception"].type,
                        record["exception"].value,
                        record["exception"].traceback,
                    ),
                ),
                "error.kind": getattr(record["exception"].type, "__name__", "None"),
                "error.message": str(record["exception"].value),
            }
            log |= error_data

        return log


def __dd_sink(message: loguru.Message) -> None:
    serialized = LogWrapper.serialize(message.record)
    LogWrapper.write_events_to_datadog(serialized)


def update_root_handler() -> None:
    """Update the root logger to log only INFO and above."""
    try:
        fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> |{level: <8} | {message}"
        logger.remove(0)
        logger.add(sys.stderr, level="INFO", format=fmt)
    except ValueError:
        pass


def _get_dbutils() -> dbutils_type:
    """Iterate through the stack to find the dbutils object."""
    for stack in inspect.stack():
        if "dbutils" in stack[0].f_globals:
            return stack[0].f_globals["dbutils"]  # type: ignore [no-any-return]

    msg = "dbutils not found in the stack."
    raise RuntimeError(msg)


def _get_spark() -> SparkSession:
    """Iterate through the stack to find the dbutils object."""
    for stack in inspect.stack():
        if "spark" in stack[0].f_globals:
            return stack[0].f_globals["spark"]  # type: ignore [no-any-return]

    msg = "spark not found in the stack."
    raise RuntimeError(msg)


def _get_cluster_name() -> str:
    from pyspark.errors.exceptions.connect import AnalysisException

    # Unfortunately the spark.conf.get api on serverless throws if key is not present,
    # and does not respect the default value.
    # We have to catch the exception and return the default value ourselves.
    with suppress(AnalysisException):
        spark = _get_spark()
        cluster_name = spark.conf.get(  # type: ignore [no-any-return]
            "spark.databricks.clusterUsageTags.clusterName", default="serverless"
        )
        if cluster_name:
            return cluster_name

    return "serverless"


def setup_dd_logging() -> None:
    """Set up logging to datadog."""
    dbutils = _get_dbutils()

    LogWrapper.DD_API_TOKEN = dbutils.secrets.get("keyvault", "DD-API-KEY")
    LogWrapper.cluster_name = _get_cluster_name()
    LogWrapper.user_name = (
        dbutils.notebook.entry_point.getDbutils()  # type: ignore [attr-defined]
        .notebook()
        .getContext()
        .userName()
        .get()
    )
    LogWrapper.logger_id = logger.add(__dd_sink, level="INFO")


def setup_logging(logger_name: str, **kwargs: Any) -> loguru.Logger:  # noqa: ANN401
    """Set up logging to datadog via stdout/stderr.

    :param logger_name: Descriptive name of this particular logger,
    e.g. dp-workflows, dp-notebooks etc.
    :type logger_name: str
    """
    # Since we are often called from notebooks,
    # we need to deal with log_setup happening multiple times.

    if LogWrapper.logger_id:
        logger.info("Log setup has already been done.")
    else:
        update_root_handler()
        setup_dd_logging()

    intercept_handler = InterceptHandler()
    logging.getLogger(logger_name).handlers = [intercept_handler]
    logging.getLogger(logger_name).setLevel(logging.INFO)

    return logger.bind(**kwargs)
