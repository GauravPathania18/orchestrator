"""
Comprehensive error handling for RAPTOR services
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional
from enum import Enum

class ErrorType(Enum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    VALIDATION = "validation"
    PROCESSING = "processing"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"

class RaptorError(Exception):
    """Base exception for RAPTOR-related errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN, details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self.message)

class TimeoutError(RaptorError):
    def __init__(self, message: str, timeout_duration: float, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.TIMEOUT, details)
        self.timeout_duration = timeout_duration

class ConnectionError(RaptorError):
    def __init__(self, message: str, service: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.CONNECTION, details)
        self.service = service

class ValidationError(RaptorError):
    def __init__(self, message: str, field: str, value: Any, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.VALIDATION, details)
        self.field = field
        self.value = value

class ProcessingError(RaptorError):
    def __init__(self, message: str, stage: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.PROCESSING, details)
        self.stage = stage

class ResourceError(RaptorError):
    def __init__(self, message: str, resource: str, details: Optional[Dict] = None):
        super().__init__(message, ErrorType.RESOURCE, details)
        self.resource = resource

def handle_exception(func):
    """Decorator for comprehensive error handling - supports both sync and async functions."""
    import asyncio
    import functools

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RaptorError as e:
            logging.error(f"RAPTOR Error [{e.error_type.value}]: {e.message}")
            if e.details:
                logging.error(f"Details: {e.details}")
            return create_error_response(e.error_type.value, e.message, e.details)
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return create_error_response("unknown", str(e), {"function": func.__name__})

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RaptorError as e:
            logging.error(f"RAPTOR Error [{e.error_type.value}]: {e.message}")
            if e.details:
                logging.error(f"Details: {e.details}")
            return create_error_response(e.error_type.value, e.message, e.details)
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return create_error_response("unknown", str(e), {"function": func.__name__})

    # Return async wrapper if function is async, sync wrapper otherwise
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def create_error_response(error_type: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "details": details or {},
        "timestamp": time.time()
    }

def log_service_health(service_name: str, status: str, details: Optional[Dict] = None):
    """Log service health status"""
    if status == "healthy":
        logging.info(f"✅ {service_name} service is healthy")
    else:
        logging.warning(f"⚠️  {service_name} service status: {status}")
        if details:
            logging.warning(f"Details: {details}")

def validate_input(data: Dict[str, Any], required_fields: list) -> None:
    """Validate input data against required fields"""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            "validation",
            missing_fields
        )

def validate_text_length(text: str, min_length: int = 1, max_length: int = 100000) -> None:
    """Validate text input length"""
    if not isinstance(text, str):
        raise ValidationError("Text must be a string", "text_type", type(text))
    
    if len(text.strip()) < min_length:
        raise ValidationError(f"Text must be at least {min_length} characters", "text_length", len(text))
    
    if len(text) > max_length:
        raise ValidationError(f"Text cannot exceed {max_length} characters", "text_length", len(text))

def validate_positive_integer(value: Any, field_name: str) -> int:
    """Validate positive integer input"""
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValidationError(f"{field_name} must be positive", field_name, value)
        return int_value
    except (ValueError, TypeError):
        raise ValidationError(f"{field_name} must be a valid integer", field_name, value)

def safe_execute_with_fallback(func, fallback_func, *args, **kwargs):
    """Execute function with fallback on failure"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.warning(f"Primary function failed: {e}, using fallback")
        try:
            return fallback_func(*args, **kwargs)
        except Exception as fallback_error:
            logging.error(f"Fallback function also failed: {fallback_error}")
            raise ProcessingError(
                f"Both primary and fallback functions failed: {e}, {fallback_error}",
                "execution",
                {"primary_error": str(e), "fallback_error": str(fallback_error)}
            )

class CircuitBreaker:
    """Circuit breaker pattern for service protection"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ConnectionError("Circuit breaker is open", func.__name__)

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution"""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
