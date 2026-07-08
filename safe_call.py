from functools import wraps


def safe_call(tool_name: str):
    """
    Decorator for gracefully handling tool failures.

    Instead of crashing the agent,
    the tool returns a readable error message.
    """

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            try:

                return function(*args, **kwargs)

            except Exception as error:

                return (
                    f"❌ {tool_name} failed.\n\n"
                    f"Reason: {str(error)}"
                )

        return wrapper

    return decorator