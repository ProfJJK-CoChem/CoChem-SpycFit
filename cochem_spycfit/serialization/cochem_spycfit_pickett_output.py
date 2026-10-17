"""Parsers for files written by Pickett's SPFIT.

The layouts follow the format statements in the JPL calpgm source ``calfit.c``
(https://spec.jpl.nasa.gov/ftp/pub/calpgm/):

* residual lines: ``"%5d: %s%14.5f%14.5f %10.5f %10.5f %9.5f"``, optionally followed for
  blended lines by ``"%14.5f %10.5f %6.4f"`` (average calculated frequency, its difference,
  weight). Infrared lines use 7 decimals and a negative experimental error (cm-1).
* parameter lines: ``"%4d %s %10.10s %s"`` (index, parameter code, label, value(error) and
  change). The value(error) text keeps at most three error digits, so full-precision values
  and errors are read from the ``.var`` file that SPFIT writes.
* summary lines: ``MICROWAVE AVG``/``MICROWAVE RMS`` (MHz) and ``END OF ITERATION n OLD, NEW
  RMS ERROR`` (dimensionless, weighted by the experimental errors).

Quantities absent from a file are returned as None and listed in ``missing``. Nothing is
replaced by zero or by a default (SF-IO-002, SF-CMP-012).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple


class PickettOutputError(ValueError):
    """Raised when text is not a recognisable SPFIT output or parameter file."""


@dataclass(frozen=True)
class SpfitResidual:
    """One residual line of an SPFIT iteration, as printed."""

    line: int
    quantum_numbers: Tuple[int, ...]
    obs: float
    calc: float
    obs_minus_calc: float
    exp_err: float
    est_err: float
    blend_avg_calc: Optional[float] = None
    blend_avg_diff: Optional[float] = None
    blend_weight: Optional[float] = None
    used_in_fit: bool = True

    @property
    def is_infrared(self) -> bool:
        """SPFIT marks wavenumber (cm-1) lines with a negative experimental error."""
        return self.exp_err < 0.0

    @property
    def is_blend_width_record(self) -> bool:
        """Blend-width records carry no quantum numbers and an observed value of zero."""
        return len(self.quantum_numbers) == 0


@dataclass(frozen=True)
class SpfitParameterEstimate:
    """One row of a ``NEW PARAMETER (EST. ERROR)`` block, at printed precision."""

    index: int
    code: int
    label: str
    value: float
    est_error: float
    change: Optional[float]
    text: str


@dataclass(frozen=True)
class SpfitIteration:
    """Everything SPFIT prints for one iteration."""

    number: int
    residuals: Tuple[SpfitResidual, ...]
    parameters: Tuple[SpfitParameterEstimate, ...]
    microwave_avg_mhz: Optional[float]
    microwave_rms_mhz: Optional[float]
    ir_avg: Optional[float]
    ir_rms: Optional[float]
    rms_error_old: Optional[float]
    rms_error_new: Optional[float]
    lines_rejected: int
    diverging: bool


@dataclass(frozen=True)
class SpfitFitReport:
    """Parsed content of an SPFIT ``.fit`` file."""

    title: str
    lines_requested: Optional[int]
    parameters_requested: Optional[int]
    iterations_requested: Optional[int]
    iterations: Tuple[SpfitIteration, ...]
    bad_lines: Tuple[str, ...]
    error_scaling_notes: Tuple[str, ...]
    missing: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def final(self) -> SpfitIteration:
        """The last iteration SPFIT completed."""
        return self.iterations[-1]


@dataclass(frozen=True)
class PickettParameter:
    """One parameter line of a ``.par`` or ``.var`` file."""

    code: int
    value: float
    error: float
    label: str


_QN_FIELD_WIDTH = 36
_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
_RESIDUAL_RE = re.compile(r"^\s*(\d+): (.*)$")
_FIELD_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")
_PARAM_LINE_RE = re.compile(r"^\s*(\d+)\s+(-?\d+) (.{10}) (.+)$")
_PARE_RE = re.compile(
    r"^\s*([-+]?\d+\.?\d*)\(\s*(\d+)\)(E[-+]\d+)?\s+([-+]?\d+\.?\d*(?:E[-+]\d+)?)\s*$"
)
_HEADER_RE = re.compile(
    r"LINES REQUESTED=\s*(\d+)\s+NUMBER OF PARAMETERS=\s*(\d+)\s+NUMBER OF ITERATIONS=\s*(\d+)"
)
_MW_AVG_RE = re.compile(r"MICROWAVE AVG =\s*(" + _FLOAT + r") MHz, IR AVG =\s*(" + _FLOAT + r")")
_MW_RMS_RE = re.compile(r"MICROWAVE RMS =\s*(" + _FLOAT + r") MHz, IR RMS =\s*(" + _FLOAT + r")")
_END_ITER_RE = re.compile(
    r"END OF ITERATION\s+(\d+) OLD, NEW RMS ERROR=\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")"
)
_REJECTED_RE = re.compile(r"^\s*(\d+) Lines rejected from fit")
_PICKETT_PARAM_RE = re.compile(
    r"^\s*(-?\d+)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*(?:/(.*))?$"
)


def _to_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def parse_value_error(text: str) -> Tuple[float, float, Optional[float]]:
    """Parse SPFIT's ``value(error)[E±nn] change`` text into (value, error, change).

    The error is printed as an integer in units of the last printed digit of the value,
    and an exponent suffix applies to value and error alike (``parer`` in calfit.c).
    """
    match = _PARE_RE.match(text)
    if match is None:
        raise PickettOutputError(f"Unrecognised SPFIT value(error) text: {text!r}")
    mantissa, err_digits, exponent, change = match.groups()
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    scale = 10.0 ** int(exponent[1:]) if exponent else 1.0
    value = float(mantissa) * scale
    error = int(err_digits) * 10.0 ** (-decimals) * scale
    change_value = _to_float(change) if change else None
    return value, error, change_value


def _parse_residual(rest: str, used_in_fit: bool, line_number: int) -> SpfitResidual:
    # qnfmt2 in calfit.c always writes 12 three-character slots: the quanta in use
    # (upper state, then lower state) followed by blank slots.
    qn_text, numeric_text = rest[:_QN_FIELD_WIDTH], rest[_QN_FIELD_WIDTH:]
    slots = [qn_text[i:i + 3] for i in range(0, len(qn_text), 3)]
    used = [s for s in slots if s.strip()]
    if any(not s.strip() for s in slots[:len(used)]):
        raise PickettOutputError(f"Residual line {line_number}: quantum-number slots are not contiguous")
    try:
        quanta = tuple(int(s) for s in used)
    except ValueError as exc:
        raise PickettOutputError(f"Residual line {line_number}: unreadable quantum numbers {qn_text!r}") from exc
    floats = list(_FIELD_FLOAT_RE.finditer(numeric_text))
    if len(floats) not in (5, 8):
        raise PickettOutputError(
            f"Residual line {line_number} has {len(floats)} numeric fields; expected 5 or 8"
        )
    values = [float(m.group(0)) for m in floats]
    blend = values[5:] if len(values) == 8 else [None, None, None]
    return SpfitResidual(
        line=line_number,
        quantum_numbers=quanta,
        obs=values[0],
        calc=values[1],
        obs_minus_calc=values[2],
        exp_err=values[3],
        est_err=values[4],
        blend_avg_calc=blend[0],
        blend_avg_diff=blend[1],
        blend_weight=blend[2],
        used_in_fit=used_in_fit,
    )


def parse_spfit_fit(text: str) -> SpfitFitReport:
    """Parse the text of an SPFIT ``.fit`` file.

    Raises:
        PickettOutputError: if the text contains no completed iteration.
    """
    lines = text.splitlines()
    if not lines:
        raise PickettOutputError("Empty SPFIT .fit text")

    title = lines[0].strip()
    lines_requested = parameters_requested = iterations_requested = None
    bad_lines: List[str] = []
    scaling_notes: List[str] = []
    iterations: List[SpfitIteration] = []

    residuals: List[SpfitResidual] = []
    parameters: List[SpfitParameterEstimate] = []
    mw_avg = ir_avg = mw_rms = ir_rms = None
    rejected = 0
    diverging = False
    in_parameters = False
    next_unused = False

    for raw in lines[1:]:
        line = raw.rstrip("\r")
        stripped = line.strip()

        header = _HEADER_RE.search(line)
        if header:
            lines_requested, parameters_requested, iterations_requested = (int(g) for g in header.groups())
            continue
        if stripped.startswith("Bad Line("):
            bad_lines.append(stripped)
            continue
        if stripped.startswith("PARAMETER ERRORS SCALED BY") or stripped.startswith(
            "WARNING: parameter errors multiplied by"
        ):
            scaling_notes.append(stripped)
            continue
        if "EXP.FREQ." in line and "CALC.FREQ." in line:
            residuals, parameters = [], []
            mw_avg = ir_avg = mw_rms = ir_rms = None
            rejected, diverging, in_parameters, next_unused = 0, False, False, False
            continue
        if "NEXT LINE NOT USED IN FIT" in line:
            next_unused = True
            continue
        rej = _REJECTED_RE.match(line)
        if rej:
            rejected = int(rej.group(1))
            continue
        if stripped.startswith("Fit Diverging"):
            diverging = True
            continue
        if "NEW PARAMETER (EST. ERROR)" in line:
            in_parameters = True
            continue

        avg = _MW_AVG_RE.search(line)
        if avg:
            in_parameters = False
            mw_avg, ir_avg = _to_float(avg.group(1)), _to_float(avg.group(2))
            continue
        rms = _MW_RMS_RE.search(line)
        if rms:
            mw_rms, ir_rms = _to_float(rms.group(1)), _to_float(rms.group(2))
            continue
        end = _END_ITER_RE.search(line)
        if end:
            iterations.append(
                SpfitIteration(
                    number=int(end.group(1)),
                    residuals=tuple(residuals),
                    parameters=tuple(parameters),
                    microwave_avg_mhz=mw_avg,
                    microwave_rms_mhz=mw_rms,
                    ir_avg=ir_avg,
                    ir_rms=ir_rms,
                    rms_error_old=_to_float(end.group(2)),
                    rms_error_new=_to_float(end.group(3)),
                    lines_rejected=rejected,
                    diverging=diverging,
                )
            )
            in_parameters = False
            continue

        if in_parameters:
            pmatch = _PARAM_LINE_RE.match(line)
            if pmatch:
                value, error, change = parse_value_error(pmatch.group(4))
                parameters.append(
                    SpfitParameterEstimate(
                        index=int(pmatch.group(1)),
                        code=int(pmatch.group(2)),
                        label=pmatch.group(3).strip(),
                        value=value,
                        est_error=error,
                        change=change,
                        text=pmatch.group(4).strip(),
                    )
                )
            continue

        rmatch = _RESIDUAL_RE.match(line)
        if rmatch:
            residuals.append(_parse_residual(rmatch.group(2), not next_unused, int(rmatch.group(1))))
            next_unused = False

    if not iterations:
        raise PickettOutputError("No completed SPFIT iteration (END OF ITERATION) found")

    missing: List[str] = []
    if lines_requested is None:
        missing.append("header (LINES REQUESTED / NUMBER OF PARAMETERS / NUMBER OF ITERATIONS)")
    final = iterations[-1]
    if final.microwave_rms_mhz is None:
        missing.append("MICROWAVE RMS of the final iteration")
    if not final.residuals:
        missing.append("residual table of the final iteration")
    if not final.parameters:
        missing.append("parameter table of the final iteration")

    return SpfitFitReport(
        title=title,
        lines_requested=lines_requested,
        parameters_requested=parameters_requested,
        iterations_requested=iterations_requested,
        iterations=tuple(iterations),
        bad_lines=tuple(bad_lines),
        error_scaling_notes=tuple(scaling_notes),
        missing=tuple(missing),
    )


def parse_pickett_parameters(text: str) -> Tuple[PickettParameter, ...]:
    """Parse the parameter lines of a ``.par`` or ``.var`` file.

    The second line holds NPAR first; the parameter lines follow the option line(s) as
    ``code value error /label``. Anything after them (the correlation block of a ``.var``)
    is ignored.

    Raises:
        PickettOutputError: if NPAR cannot be read or fewer than NPAR parameter lines exist.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        raise PickettOutputError("A .par/.var file needs a title, a control line and parameters")
    fields = lines[1].split()
    try:
        npar = int(fields[0])
    except (IndexError, ValueError) as exc:
        raise PickettOutputError(f"Cannot read NPAR from control line {lines[1]!r}") from exc

    params: List[PickettParameter] = []
    for raw in lines[2:]:
        if len(params) == npar:
            break
        match = _PICKETT_PARAM_RE.match(raw.rstrip("\r"))
        if match is None or "." not in match.group(2):
            continue
        params.append(
            PickettParameter(
                code=int(match.group(1)),
                value=_to_float(match.group(2)),
                error=_to_float(match.group(3)),
                label=(match.group(4) or "").strip(),
            )
        )
    if len(params) != npar:
        raise PickettOutputError(f"Expected {npar} parameter lines, found {len(params)}")
    return tuple(params)


__all__ = [
    "PickettOutputError",
    "PickettParameter",
    "SpfitFitReport",
    "SpfitIteration",
    "SpfitParameterEstimate",
    "SpfitResidual",
    "parse_pickett_parameters",
    "parse_spfit_fit",
    "parse_value_error",
]
