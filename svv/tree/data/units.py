"""Utility classes for configuring unit systems used during tree generation.

The growth pipeline traditionally assumes centimetre–gram–second (CGS) units,
but several parts of the SimVascular tooling operate in SI.  This module
provides a light-weight registry that lets callers describe the base units they
want to work with and automatically derives consistent conversions for common
hemodynamic quantities (pressure, flow, viscosity, etc.).

Typical usage
-------------

>>> from svv.tree.data.units import UnitSystem
>>> cgs = UnitSystem()  # defaults to cm, g, s
>>> si = UnitSystem(length='m', mass='kg', time='s', pressure='Pa')
>>> cgs.convert_value(120.0, 'pressure', to_system=si)  # 120 dyn/cm^2 to pascal
12.0

The API deliberately mirrors the small set of classes that previously existed
so that external notebooks or scripts that import ``BaseUnit`` or
``UnitSystem`` continue to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


class UnitError(ValueError):
    """Raised when an unknown unit symbol or quantity is requested."""


# Conversion factors from each symbol to the SI base for the given dimension.
_LENGTH_UNITS: Mapping[str, float] = {
    'm': 1.0,
    'cm': 1.0e-2,
    'mm': 1.0e-3,
    'um': 1.0e-6,
    'nm': 1.0e-9,
    'km': 1.0e3,
    'in': 0.0254,
    'ft': 0.3048,
    'yd': 0.9144,
    'mi': 1609.344,
}

_MASS_UNITS: Mapping[str, float] = {
    'kg': 1.0,
    'g': 1.0e-3,
    'mg': 1.0e-6,
    'ug': 1.0e-9,
    'lb': 0.45359237,
    'oz': 0.028349523125,
    'ton': 907.18474,
    'slug': 14.59390294,
}

_TIME_UNITS: Mapping[str, float] = {
    's': 1.0,
    'ms': 1.0e-3,
    'us': 1.0e-6,
    'ns': 1.0e-9,
    'min': 60.0,
    'hr': 3600.0,
    'day': 86400.0,
    'week': 604800.0,
}

_BASE_UNIT_TABLE: Mapping[str, Mapping[str, float]] = {
    'length': _LENGTH_UNITS,
    'mass': _MASS_UNITS,
    'time': _TIME_UNITS,
}


# Derived quantity dimensions expressed as exponents of the base dimensions.
_DERIVED_DIMENSIONS: Mapping[str, Mapping[str, int]] = {
    'area': {'length': 2},
    'volume': {'length': 3},
    'velocity': {'length': 1, 'time': -1},
    'acceleration': {'length': 1, 'time': -2},
    'density': {'mass': 1, 'length': -3},
    'pressure': {'mass': 1, 'length': -1, 'time': -2},
    'shear_stress': {'mass': 1, 'length': -1, 'time': -2},
    'volumetric_flow': {'length': 3, 'time': -1},
    'mass_flow': {'mass': 1, 'time': -1},
    'kinematic_viscosity': {'length': 2, 'time': -1},
    'dynamic_viscosity': {'mass': 1, 'length': -1, 'time': -1},
    'shear_rate': {'time': -1},
}


# Frequently used derived-unit aliases (value is factor that converts *to* SI).
_DERIVED_UNIT_LIBRARY: Mapping[str, Mapping[str, float]] = {
    'area': {'m^2': 1.0, 'cm^2': 1.0e-4, 'mm^2': 1.0e-6},
    'volume': {'m^3': 1.0, 'cm^3': 1.0e-6, 'mm^3': 1.0e-9, 'L': 1.0e-3},
    'velocity': {'m/s': 1.0, 'cm/s': 1.0e-2, 'mm/s': 1.0e-3},
    'acceleration': {'m/s^2': 1.0, 'cm/s^2': 1.0e-2},
    'density': {'kg/m^3': 1.0, 'g/cm^3': 1000.0},
    'pressure': {
        'Pa': 1.0,
        'kPa': 1.0e3,
        'MPa': 1.0e6,
        'bar': 1.0e5,
        'atm': 101325.0,
        'mmHg': 133.32236842105263,
        'dyn/cm^2': 0.1,
    },
    'shear_stress': {
        'Pa': 1.0,
        'dyn/cm^2': 0.1,
    },
    'volumetric_flow': {
        'm^3/s': 1.0,
        'cm^3/s': 1.0e-6,
        'L/min': 1.0e-3 / 60.0,
        'mL/min': 1.0e-6 / 60.0,
    },
    'mass_flow': {
        'kg/s': 1.0,
        'g/s': 1.0e-3,
        'g/min': 1.0e-3 / 60.0,
    },
    'kinematic_viscosity': {
        'm^2/s': 1.0,
        'cm^2/s': 1.0e-4,
    },
    'dynamic_viscosity': {
        'Pa*s': 1.0,
        'cP': 0.001,
    },
    'shear_rate': {'1/s': 1.0},
}


def _format_power(symbol: str, power: int) -> str:
    if power == 1:
        return symbol
    return f"{symbol}^{power}"


@dataclass(frozen=True)
class BaseUnit:
    """Represents a single base unit (length, mass, or time)."""

    dimension: str
    symbol: str
    factor_si: float

    @classmethod
    def from_symbol(cls, dimension: str, symbol: str) -> 'BaseUnit':
        try:
            factor = _BASE_UNIT_TABLE[dimension][symbol]
        except KeyError as exc:
            raise UnitError(f"Unsupported {dimension} unit '{symbol}'.") from exc
        return cls(dimension=dimension, symbol=symbol, factor_si=factor)

    def convert_to(self, other: 'BaseUnit', value: float) -> float:
        """Convert a value expressed in this base unit to another base unit."""

        if self.dimension != other.dimension:
            raise UnitError(
                f"Cannot convert between {self.dimension} and {other.dimension}."
            )
        si_value = value * self.factor_si
        return si_value / other.factor_si


class BaseUnitSystem:
    """Container for the three base dimensions used by the tree pipeline."""

    def __init__(self, *, length: str = 'cm', mass: str = 'g', time: str = 's'):
        self.length = BaseUnit.from_symbol('length', length)
        self.mass = BaseUnit.from_symbol('mass', mass)
        self.time = BaseUnit.from_symbol('time', time)

    def factor_to_si(self, dimension: str) -> float:
        unit = getattr(self, dimension)
        return unit.factor_si

    def describe(self) -> str:
        return (
            f"BaseUnitSystem(length='{self.length.symbol}', "
            f"mass='{self.mass.symbol}', time='{self.time.symbol}')"
        )

    __repr__ = describe


class DerivedUnit:
    """Derived quantity built from a :class:`BaseUnitSystem`."""

    def __init__(
        self,
        name: str,
        exponents: Mapping[str, int],
        base_units: BaseUnitSystem,
        *,
        symbol: Optional[str] = None,
        factor_override: Optional[float] = None,
    ) -> None:
        self.name = name
        self._exponents = dict(exponents)
        self._base = base_units
        self._factor_si = factor_override if factor_override is not None else self._compute_factor()
        self.symbol = symbol or self._default_symbol()

    def _compute_factor(self) -> float:
        factor = 1.0
        for dimension, power in self._exponents.items():
            base_factor = self._base.factor_to_si(dimension)
            factor *= base_factor ** power
        return factor

    def _default_symbol(self) -> str:
        numerator: list[str] = []
        denominator: list[str] = []
        for dimension, power in self._exponents.items():
            symbol = getattr(self._base, dimension).symbol
            if power > 0:
                numerator.append(_format_power(symbol, power))
            elif power < 0:
                denominator.append(_format_power(symbol, -power))
        num = ' * '.join(numerator) if numerator else '1'
        if not denominator:
            return num
        den = ' * '.join(denominator)
        return f"{num}/({den})"

    @property
    def factor_si(self) -> float:
        return self._factor_si

    def convert_to(self, other: 'DerivedUnit', value: float) -> float:
        if self._exponents != other._exponents:
            raise UnitError(
                f"Cannot convert between incompatible derived units "
                f"('{self.name}' and '{other.name}')."
            )
        si_value = value * self.factor_si
        return si_value / other.factor_si

    def override(self, *, symbol: Optional[str] = None, factor: Optional[float] = None) -> None:
        if symbol is not None:
            self.symbol = symbol
        if factor is not None:
            self._factor_si = factor

    def __repr__(self) -> str:
        return f"DerivedUnit(name='{self.name}', symbol='{self.symbol}')"


class NamedUnit(DerivedUnit):
    """Backward-compatible wrapper that mirrors the original NamedUnit API."""

    def __init__(self, derived: str, base_unit_system: BaseUnitSystem, **kwargs) -> None:
        if derived not in _DERIVED_DIMENSIONS:
            raise UnitError(f"Unsupported derived quantity '{derived}'.")
        super().__init__(derived, _DERIVED_DIMENSIONS[derived], base_unit_system, **kwargs)


class UnitSystem:
    """Represents a complete set of base and derived units.

    Parameters
    ----------
    length, mass, time : str, optional
        Symbols for the base units.  Defaults to centimetre–gram–second.
    **derived_units : str or tuple, optional
        Optional overrides for derived quantities.  A string is looked up in the
        internal unit library; a tuple ``(symbol, factor_to_si)`` can be
        provided to register ad-hoc units.
    """

    def __init__(self, *, length: str = 'cm', mass: str = 'g', time: str = 's', **derived_units) -> None:
        self.base = BaseUnitSystem(length=length, mass=mass, time=time)
        self._derived: Dict[str, DerivedUnit] = {}
        unknown_keys = set(derived_units) - set(_DERIVED_DIMENSIONS)
        if unknown_keys:
            raise UnitError(
                f"Unknown derived quantity overrides: {', '.join(sorted(unknown_keys))}."
            )
        for name, exponents in _DERIVED_DIMENSIONS.items():
            override = derived_units.get(name)
            symbol: Optional[str] = None
            factor_override: Optional[float] = None
            if override is not None:
                if isinstance(override, str):
                    try:
                        factor_override = _DERIVED_UNIT_LIBRARY[name][override]
                    except KeyError as exc:
                        raise UnitError(
                            f"Unknown unit '{override}' for derived quantity '{name}'."
                        ) from exc
                    symbol = override
                else:
                    try:
                        symbol, factor_override = override
                    except Exception as exc:  # pragma: no cover - defensive
                        raise UnitError(
                            f"Override for '{name}' must be a unit symbol or (symbol, factor_to_si)."
                        ) from exc
            derived_unit = DerivedUnit(name, exponents, self.base, symbol=symbol, factor_override=factor_override)
            self._derived[name] = derived_unit
            setattr(self, name, derived_unit)

    # ---------------------------------------------------------------------
    # Introspection helpers
    # ---------------------------------------------------------------------
    def list_quantities(self) -> Iterable[str]:
        """Return an iterable of recognised derived quantity names."""

        return self._derived.keys()

    def factor_to_si(self, quantity: str) -> float:
        if quantity in ('length', 'mass', 'time'):
            return self.base.factor_to_si(quantity)
        try:
            return self._derived[quantity].factor_si
        except KeyError as exc:
            raise UnitError(f"Unknown quantity '{quantity}'.") from exc

    def factor_from_si(self, quantity: str) -> float:
        return 1.0 / self.factor_to_si(quantity)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def convert_value(
        self,
        value: float,
        quantity: str,
        *,
        to_system: Optional['UnitSystem'] = None,
        from_system: Optional['UnitSystem'] = None,
    ) -> float:
        """Convert ``value`` expressed in ``from_system`` (defaults to ``self``)
        into the units of ``to_system`` (defaults to ``self``)."""

        src = from_system or self
        dst = to_system or self
        src_factor = src.factor_to_si(quantity)
        dst_factor = dst.factor_to_si(quantity)
        return value * src_factor / dst_factor

    def override(self, quantity: str, *, symbol: Optional[str] = None, factor_to_si: Optional[float] = None) -> None:
        """Override the symbol and/or SI scaling for a derived quantity."""

        if quantity not in self._derived:
            raise UnitError(f"Cannot override unknown quantity '{quantity}'.")
        self._derived[quantity].override(symbol=symbol, factor=factor_to_si)

    def summary(self) -> str:
        lines = [self.base.describe()]
        for name in sorted(self._derived):
            unit = self._derived[name]
            lines.append(f"  {name:20s} -> {unit.symbol}")
        return '\n'.join(lines)

    __repr__ = summary


class ConvertBaseUnit:
    """Helper that mirrors the legacy conversion object."""

    def __init__(self, base_unit_0: BaseUnit, base_unit_1: BaseUnit):
        self.factor = base_unit_0.convert_to(base_unit_1, 1.0)
