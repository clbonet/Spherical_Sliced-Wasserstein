"""
A small list of utilities to transform dataclasses from argparsers

>>> @dataparser
... class MyArguments:
...     x: int = 0
...     y: str = "optional"

>>> MyArguments()
MyArguments(x=0, y='optional')

>>> MyArguments(x=20)
MyArguments(x=20, y='optional')
"""

from typing import Any, Dict, Union, List, Optional# , Literal

import argparse
import dataclasses


def _post_init(cls, *args, **kwargs):
    for k in dir(cls):
        v = getattr(cls, k)
        if isinstance(v, _Field):
            if not isinstance(v.default, _MISSING):
                setattr(cls, k, v.default)
            elif v.action == "store_true":
                setattr(cls, k, False)
            elif v.choices is not None:
                setattr(cls, k, v.choices[0])
            else:
                raise Exception(f"No value provided for cls {cls.__class__}.{k}")


def dataparser(cls=None):
    """
    A class decorator to use in place of @dataclasses.dataclass

    >>> @dataparser
    ... class A:
    ...     x: int = 0

    >>> A()
    A(x=0)

    """

    def wrapper(cls):
        setattr(cls, dataclasses._POST_INIT_NAME, _post_init)
        cls = dataclasses.dataclass(cls)
        return cls

    if cls is None:
        return wrapper

    return wrapper(cls)


class _MISSING:
    def __repr__(self):
        return "<missing>"


ActionType = str # Literal["store_true"]


@dataclasses.dataclass()
class _Field:
    "Used to add metadata for positional args for."

    help: Optional[str] = None
    choices: Optional[List[Any]] = None
    positional: bool = False
    default: Any = _MISSING()
    required: Optional[bool] = None
    metavar: Optional[str] = None
    action: Optional[ActionType] = None
    nargs: Optional[str] = None


def Field(*args, **kwargs) -> Any:
    return _Field(*args, **kwargs)


def from_args(cls):
    """
    Returns an instance of the dataclass with fields populated using the program arguments.

    Parameters
    ==========
        cls: A dataclass type
    Returns
    =======
        args: cls - An instance of cls
    """
    parser = to_argparser(cls)
    namespace = parser.parse_args()
    name_dict = namespace.__dict__

    # should be in cls.__init__
    for k, v in name_dict.items():
        if isinstance(v, _Field):
            if not isinstance(v.default, _MISSING):
                name_dict[k] = v.default
            else:
                name_dict.pop(k)

    return cls(**name_dict)


def _is_optional(t):
    if hasattr(t, "__origin__") and t.__origin__ == Union and t.__slots__ is None:
        return True
    return False


def _get_optional_type(t):
    return t.__args__[0]


def to_argparser(cls) -> argparse.ArgumentParser:
    """
    Converts an annotated dataclass to a corresponding ArgumentParser.

    Parameters
    ==========
        cls: A dataclass type
    Returns
    =======
        parser: argparse.ArgumentParser - an argument parser
    """

    annotations = cls.__dict__["__dataclass_fields__"]
    help_str = "" if "__doc__" not in cls.__dict__ else cls.__doc__

    parser = argparse.ArgumentParser(help_str)
    for k, v in annotations.items():
        if _is_optional(v.type):
            argtype = _get_optional_type(v.type)
        else:
            argtype = v.type

        if isinstance(v.default, _Field):
            field = v.default
            kwargs: Dict[str,Any] = {}
            has_default = not isinstance(field.default, _MISSING)

            if field.choices is not None:
                kwargs["choices"] = field.choices
                if not has_default:
                    kwargs["default"] = field.default = field.choices[0]
                    has_default = True

            if has_default:
                kwargs["default"] = field.default

            if field.nargs is not None:
                kwargs["nargs"] = field.nargs
                argtype = argtype.__args__[0]

            if field.action is not None:
                parser.add_argument(f"--{k}", action=field.action)
                if field.action == "store_true":
                    field.default = False
                continue

            if not field.positional:
                required = (
                    field.required if field.required is not None else not has_default
                )
                kwargs["required"] = required

            if field.help is not None:
                kwargs["help"] = field.help
            elif has_default:
                kwargs["help"] = f"{k.upper()} [default={field.default}]"

            if field.metavar is not None:
                kwargs["metavar"] = field.metavar

            prefix = "--" if not field.positional else ""

            parser.add_argument(f"{prefix}{k}", type=argtype, **kwargs)
        else:
            required = isinstance(v.default, dataclasses._MISSING_TYPE)
            kwargs = {
                "required": required,
            }

            if not required:
                kwargs["default"] = v.default
                kwargs["help"] = f"[default={v.default}]"

            parser.add_argument(f"--{k}", type=argtype, **kwargs)

    return parser


def test_argparse():
    @dataparser
    class MyArguments:
        x: int = 0
        y: str = "optional"

    parser = to_argparser(MyArguments)
    args = parser.parse_args(["--x", "10", "--y", "y_value"])
    assert args.x == 10
    assert args.y == "y_value"


def test_optional():
    @dataparser
    class MyClass:
        x: Optional[int] = None

    parser = to_argparser(MyClass)
    args = parser.parse_args([])
    assert args.x is None

    args = parser.parse_args(["--x", "1"])
    assert args.x == 1
