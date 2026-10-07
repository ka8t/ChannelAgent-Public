"""Exact calculation, date arithmetic and unit conversion for the built-in MCP server "calc".
A parser of our own: nothing is evaluated as code (no `eval`, no `ast` compile), the
input is bounded and so is every intermediate result.

- Arithmetic: numbers (integers, decimals, `1e3`), `+ - * / // %`, `^` or `**` for a power,
  parentheses, unary signs, `sqrt(x)` and `abs(x)`. Computed with fractions, so `0.1 + 0.2` is
  exactly `0.3` and `1/3` is shown as `1/3 (about 0.333333333333)`.
- Dates: `2026-09-29 + 3 days` (days, weeks, months, years; a month added to the 31st ends on
  the last day of the shorter month), `2026-12-25 - 2026-10-04` (days between), `weekday
  2026-10-04`. The weekday is given with every date result.
- Units: `convert(value, from_unit, to_unit)` within one family (length, mass, volume, area,
  time, speed, data, temperature), with exact factors.

Every refusal raises CalcError with a message the model can act on.
"""

import calendar
import math
import re
from datetime import date, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction

MAX_INPUT_CHARS = 200
MAX_DEPTH = 30
MAX_DIGITS = 1000  # of a numerator or a denominator, at any step
MAX_DATE_OFFSET_DAYS = 3_660_000  # about 10,000 years
SHOWN_DIGITS = 12


class CalcError(ValueError):
    """The input is refused or cannot be computed; the message says why."""


# --- arithmetic ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<op>\*\*|//|[-+*/%^()])"
    r"|(?P<name>[A-Za-z]+)|(?P<comma>,))"
)
_FUNCTIONS = ("sqrt", "abs")


def _digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _bounded(value: Fraction) -> Fraction:
    if _digits(value) > MAX_DIGITS:
        raise CalcError(f"the result has more than {MAX_DIGITS} digits")
    return value


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise CalcError(f"unexpected character {text[pos:].strip()[:1]!r}")
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
        pos = match.end()
    return out


class _Parser:
    """expr := term (('+'|'-') term)*; term := unary (('*'|'/'|'//'|'%') unary)*;
    unary := ('+'|'-') unary | power; power := atom (('^'|'**') unary)?;
    atom := number | '(' expr ')' | function '(' expr ')'."""

    def __init__(self, tokens):
        self.tokens, self.pos, self.depth = tokens, 0, 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def take(self, value=None):
        kind, text = self.peek()
        if kind is None or (value is not None and text != value):
            raise CalcError(f"expected {value!r}" if value else "the expression ends too early")
        self.pos += 1
        return kind, text

    def nest(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError(f"more than {MAX_DEPTH} nested levels")

    def expr(self) -> Fraction:
        value = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            right = self.term()
            value = _bounded(value + right if op == "+" else value - right)
        return value

    def term(self) -> Fraction:
        value = self.unary()
        while self.peek()[1] in ("*", "/", "//", "%"):
            op = self.take()[1]
            right = self.unary()
            if op == "*":
                value = value * right
            elif right == 0:
                raise CalcError("division by zero")
            elif op == "/":
                value = value / right
            elif op == "//":
                value = Fraction(value // right)
            else:
                value = value % right
            value = _bounded(value)
        return value

    def unary(self) -> Fraction:
        if self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            self.nest()
            value = self.unary()
            self.depth -= 1
            return -value if op == "-" else value
        return self.power()

    def power(self) -> Fraction:
        base = self.atom()
        if self.peek()[1] in ("^", "**"):
            self.take()
            self.nest()
            exponent = self.unary()
            self.depth -= 1
            return _power(base, exponent)
        return base

    def atom(self) -> Fraction:
        kind, text = self.take()
        if kind == "num":
            return _number(text)
        if text == "(":
            self.nest()
            value = self.expr()
            self.take(")")
            self.depth -= 1
            return value
        if kind == "name" and text.lower() in _FUNCTIONS:
            self.take("(")
            self.nest()
            value = self.expr()
            self.take(")")
            self.depth -= 1
            return abs(value) if text.lower() == "abs" else _sqrt(value)
        if kind == "name":
            raise CalcError(f"unknown name {text!r}; functions: {', '.join(_FUNCTIONS)}")
        raise CalcError(f"unexpected {text!r}")


def _number(text: str) -> Fraction:
    """A number of the input, its size checked before it is built: `1e999999999` would
    otherwise be a billion-digit integer."""
    text = text.strip()
    if len(text) > 60:
        raise CalcError("a number longer than 60 characters")
    try:
        value = Decimal(text)
    except ArithmeticError:
        raise CalcError(f"{text!r} is not a number") from None
    if not value.is_finite():
        raise CalcError(f"{text!r} is not a number")
    if abs(value.adjusted()) > MAX_DIGITS:
        raise CalcError(f"a number with more than {MAX_DIGITS} digits")
    return Fraction(value)


def _power(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        if exponent == Fraction(1, 2):
            return _sqrt(base)
        raise CalcError("only a whole exponent (or 0.5, a square root) is supported")
    n = exponent.numerator
    if base == 0 and n < 0:
        raise CalcError("division by zero")
    if base not in (0, 1, -1):
        # Estimated before computing: 2^99999999 would take minutes and gigabytes.
        size = max(math.log10(abs(base.numerator) or 1), math.log10(base.denominator))
        if abs(n) * size > MAX_DIGITS:
            raise CalcError(f"the result has more than {MAX_DIGITS} digits")
    return _bounded(base**n)


def _sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise CalcError("the square root of a negative number")
    with localcontext() as ctx:
        ctx.prec = 40
        root = (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()
    exact = Fraction(root)
    # A perfect square gives an exact root: keep it exact.
    simple = exact.limit_denominator(10**6)
    return simple if simple * simple == value else exact


def _show(value: Fraction, fraction: bool = True) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    with localcontext() as ctx:
        ctx.prec = SHOWN_DIGITS
        approx = Decimal(value.numerator) / Decimal(value.denominator)
    small = abs(approx) < Decimal("1e-6")
    decimal_text = str(approx) if small else format(approx.normalize(), "f")
    exact_decimal = Fraction(Decimal(decimal_text)) == value
    if exact_decimal:
        return decimal_text
    if fraction and _digits(value) <= 12:
        return f"{value.numerator}/{value.denominator} (about {decimal_text})"
    return f"about {decimal_text}"


def arithmetic(text: str) -> str:
    parser = _Parser(_tokens(text))
    value = parser.expr()
    if parser.pos != len(parser.tokens):
        raise CalcError(f"unexpected {parser.peek()[1]!r}")
    return _show(value)


# --- dates ----------------------------------------------------------------------------------

_DATE = r"(\d{4}-\d{2}-\d{2})"
_UNITS = {
    "day": "days", "days": "days", "d": "days",
    "week": "weeks", "weeks": "weeks", "w": "weeks",
    "month": "months", "months": "months",
    "year": "years", "years": "years", "y": "years",
}  # fmt: skip
_OFFSET = re.compile(r"\s*([+-])\s*(\d{1,7})\s*([a-z]+)\s*", re.I)
_DATE_EXPR = re.compile(rf"^\s*{_DATE}((?:\s*[+-]\s*\d{{1,7}}\s*[a-z]+)+)\s*$", re.I)
_DATE_DIFF = re.compile(rf"^\s*{_DATE}\s*-\s*{_DATE}\s*$")
_WEEKDAY = re.compile(rf"^\s*(?:weekday|day of)\s*(?:of\s*)?{_DATE}\s*$", re.I)


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise CalcError(f"{text} is not a valid date (YYYY-MM-DD)") from None


def _with_weekday(day: date) -> str:
    return f"{day.isoformat()} ({calendar.day_name[day.weekday()]})"


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    year, month = divmod(index, 12)
    if not 1 <= year <= 9999:
        raise CalcError("the date leaves the years 1 to 9999")
    last = calendar.monthrange(year, month + 1)[1]
    return date(year, month + 1, min(day.day, last))


def dates(text: str) -> str | None:
    """The result of a date expression, or None when `text` is not one."""
    if match := _WEEKDAY.match(text):
        return _with_weekday(_parse_date(match.group(1)))
    if match := _DATE_DIFF.match(text):
        days = (_parse_date(match.group(1)) - _parse_date(match.group(2))).days
        weeks, rest = divmod(abs(days), 7)
        sign = "-" if days < 0 else ""
        return f"{days} days ({sign}{weeks} weeks and {rest} days)"
    if match := _DATE_EXPR.match(text):
        day = _parse_date(match.group(1))
        for sign, amount, unit in _OFFSET.findall(match.group(2)):
            kind = _UNITS.get(unit.lower())
            if kind is None:
                raise CalcError(f"unknown date unit {unit!r}; units: days, weeks, months, years")
            n = int(amount) * (-1 if sign == "-" else 1)
            try:
                if kind in ("days", "weeks"):
                    offset = n * (7 if kind == "weeks" else 1)
                    if abs(offset) > MAX_DATE_OFFSET_DAYS:
                        raise CalcError("the offset is too large")
                    day = day + timedelta(days=offset)
                else:
                    day = _add_months(day, n * (12 if kind == "years" else 1))
            except OverflowError:
                raise CalcError("the date leaves the years 1 to 9999") from None
        return _with_weekday(day)
    return None


def calculate(expression: str) -> str:
    text = (expression or "").strip()
    if not text:
        raise CalcError("the expression is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise CalcError(f"the expression is longer than {MAX_INPUT_CHARS} characters")
    result = dates(text)
    return result if result is not None else arithmetic(text)


# --- units ----------------------------------------------------------------------------------

F = Fraction
# family -> unit -> factor to the family's base unit; aliases point at a unit's factor.
_FAMILIES: dict[str, dict[str, Fraction]] = {
    "length": {
        "mm": F(1, 1000), "cm": F(1, 100), "m": F(1), "km": F(1000),
        "in": F("0.0254"), "ft": F("0.3048"), "yd": F("0.9144"), "mi": F("1609.344"),
        "nmi": F(1852),
    },
    "mass": {
        "mg": F(1, 10**6), "g": F(1, 1000), "kg": F(1), "t": F(1000),
        "oz": F("0.028349523125"), "lb": F("0.45359237"), "st": F("6.35029318"),
    },
    "volume": {
        "ml": F(1, 1000), "cl": F(1, 100), "dl": F(1, 10), "l": F(1), "m3": F(1000),
        "tsp": F("0.00492892159375"), "tbsp": F("0.01478676478125"),
        "floz": F("0.0295735295625"), "cup": F("0.2365882365"), "pt": F("0.473176473"),
        "qt": F("0.946352946"), "gal": F("3.785411784"), "ukgal": F("4.54609"),
    },
    "area": {
        "cm2": F(1, 10**4), "m2": F(1), "ha": F(10**4), "km2": F(10**6),
        "ft2": F("0.09290304"), "in2": F("0.00064516"), "acre": F("4046.8564224"),
        "mi2": F("2589988.110336"),
    },
    "time": {
        "ms": F(1, 1000), "s": F(1), "min": F(60), "h": F(3600), "day": F(86400),
        "week": F(604800), "year": F(31557600),
    },
    "speed": {
        "m/s": F(1), "km/h": F(1000, 3600), "mph": F("1609.344") / 3600, "kn": F(1852, 3600),
        "ft/s": F("0.3048"),
    },
    "data": {
        "bit": F(1, 8), "b": F(1), "kb": F(1000), "mb": F(10**6), "gb": F(10**9),
        "tb": F(10**12), "kib": F(1024), "mib": F(1024**2), "gib": F(1024**3),
        "tib": F(1024**4),
    },
}  # fmt: skip
_ALIASES = {
    "millimeter": "mm", "millimetre": "mm", "centimeter": "cm", "centimetre": "cm",
    "meter": "m", "metre": "m", "kilometer": "km", "kilometre": "km", "inch": "in",
    "inches": "in", "foot": "ft", "feet": "ft", "yard": "yd", "mile": "mi", "miles": "mi",
    "nautical mile": "nmi", "gram": "g", "gramme": "g", "kilogram": "kg", "kilo": "kg",
    "milligram": "mg", "tonne": "t", "ton": "t", "ounce": "oz", "pound": "lb", "lbs": "lb",
    "stone": "st", "liter": "l", "litre": "l", "milliliter": "ml", "millilitre": "ml",
    "centiliter": "cl", "deciliter": "dl", "cubic meter": "m3", "m³": "m3", "teaspoon": "tsp",
    "tablespoon": "tbsp", "fl oz": "floz", "fluid ounce": "floz", "pint": "pt", "quart": "qt",
    "gallon": "gal", "us gallon": "gal", "uk gallon": "ukgal", "imperial gallon": "ukgal",
    "m²": "m2", "square meter": "m2", "km²": "km2", "hectare": "ha", "ft²": "ft2",
    "square foot": "ft2", "square feet": "ft2", "acres": "acre", "second": "s", "sec": "s",
    "minute": "min", "hour": "h", "hr": "h", "days": "day", "d": "day", "weeks": "week",
    "years": "year", "millisecond": "ms", "kph": "km/h", "kmh": "km/h", "knot": "kn",
    "knots": "kn", "mps": "m/s", "byte": "b", "bytes": "b", "kilobyte": "kb",
    "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb", "bits": "bit",
}  # fmt: skip
_LABELS = {"b": "B", "kb": "kB", "mb": "MB", "gb": "GB", "tb": "TB", "kib": "KiB",
           "mib": "MiB", "gib": "GiB", "tib": "TiB", "l": "L"}  # fmt: skip
_TEMPERATURE = {"c": "C", "°c": "C", "celsius": "C", "f": "F", "°f": "F", "fahrenheit": "F",
                "k": "K", "kelvin": "K"}  # fmt: skip


def _unit(name: str) -> tuple[str, str]:
    key = " ".join((name or "").strip().lower().split())
    if key in _TEMPERATURE:
        return "temperature", _TEMPERATURE[key]
    for candidate in (key, key.rstrip("s") if len(key) > 3 else key):
        candidate = _ALIASES.get(candidate, candidate)
        for family, units in _FAMILIES.items():
            if candidate in units:
                return family, candidate
    raise CalcError(f"unknown unit {name!r}")


def _to_kelvin(value: Fraction, unit: str) -> Fraction:
    if unit == "C":
        return value + F("273.15")
    if unit == "F":
        return (value - 32) * F(5, 9) + F("273.15")
    return value


def _from_kelvin(value: Fraction, unit: str) -> Fraction:
    if unit == "C":
        return value - F("273.15")
    if unit == "F":
        return (value - F("273.15")) * F(9, 5) + 32
    return value


def convert(value, from_unit: str, to_unit: str) -> str:
    amount = _number(str(value))
    family, source = _unit(from_unit)
    other, target = _unit(to_unit)
    if family != other:
        raise CalcError(f"{from_unit} ({family}) cannot be converted to {to_unit} ({other})")
    if family == "temperature":
        result = _from_kelvin(_to_kelvin(amount, source), target)
    else:
        units = _FAMILIES[family]
        result = amount * units[source] / units[target]
    before, after = _LABELS.get(source, source), _LABELS.get(target, target)
    shown = _show(_bounded(result), fraction=False)
    return f"{_show(amount, fraction=False)} {before} = {shown} {after}"
