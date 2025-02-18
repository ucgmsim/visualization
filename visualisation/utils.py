"""Utility functions common to many plotting scripts."""

import inspect
from functools import wraps
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin

import docstring_parser
import typer
from docstring_parser.common import DocstringStyle


# Originally written by @Genfood: https://github.com/fastapi/typer/issues/336#issuecomment-2434726193
# Updated and modified for Python 3.13.
def from_docstring(command: Callable) -> Callable:
    """Apply help texts from the function's docstring to Typer arguments/options.

    It will only apply the help text if no help text has been explicitly set in
    the Typer argument/option.

    Parameters
    ----------
    command : Callable
        The function to decorate.

    Returns
    -------
    Callable
        The decorated function with help texts applied, without overwriting
        existing settings.
    """
    if command.__doc__ is None:
        return command

    # Parse the docstring and extract parameter descriptions
    docstring = docstring_parser.parse(command.__doc__, style=DocstringStyle.NUMPYDOC)
    param_help = {param.arg_name: param.description for param in docstring.params}

    # The command's full help text (summary + long description)
    command_help = (
        f"{docstring.short_description or ''}\n\n{docstring.long_description or ''}"
    )

    # Get the signature of the original function
    sig = inspect.signature(command)
    parameters = sig.parameters

    @wraps(command)
    def wrapper(**kwargs: Any) -> Any:  # numpydoc ignore=GL08
        return command(**kwargs)

    # Prepare a new mapping for parameters
    new_parameters = []

    for name, param in parameters.items():
        help_text = param_help.get(
            name, ""
        )  # Get help text from docstring if available
        param_type = (
            param.annotation if param.annotation is not inspect.Parameter.empty else str
        )

        # Handle Annotated types
        if get_origin(param_type) is Annotated:
            param_type, *metadata = get_args(param_type)
            new_metadata = []
            for m in metadata:
                if isinstance(m, typer.models.ArgumentInfo | typer.models.OptionInfo):
                    if not m.help:
                        m.help = help_text
                new_metadata.append(m)
            new_param = param.replace(annotation=Annotated[param_type, *new_metadata])

        # If it's an Option or Argument directly
        elif isinstance(
            param.default, typer.models.ArgumentInfo | typer.models.OptionInfo
        ):
            if not param.default.help:
                param.default.help = help_text
            new_param = param

        else:
            # If the parameter has no default, treat it as an Argument
            if param.default is inspect.Parameter.empty:
                new_param = param.replace(
                    default=typer.Argument(..., help=help_text), annotation=param_type
                )
            else:
                # If the parameter has a default, treat it as an Option
                new_param = param.replace(
                    default=typer.Option(param.default, help=help_text),
                    annotation=param_type,
                )

        new_parameters.append(new_param)

    # Create a new signature with updated parameters
    new_sig = sig.replace(parameters=new_parameters)

    # Apply the new signature to the wrapper function
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = command_help.strip()

    return wrapper


def format_description(arr: np.ndarray, dp: float = 0) -> str:
    """Format a statistical description of an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    dp : float, optional
        Decimal places to round to, by default 0.

    Returns
    -------
    str
        Formatted string containing min, mean, max, and standard deviation.
    """
    min = arr.min()
    mean = np.mean(arr)
    max = arr.max()
    std = np.std(arr)
    return f"min = {min:.{dp}f} / μ = {mean:.{dp}f} / σ = {std:.{dp}f} / max = {max:.{dp}f}"
