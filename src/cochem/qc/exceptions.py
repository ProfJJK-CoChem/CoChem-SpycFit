"""
CoChem Quantum Chemistry Domain Exception Hierarchy.

Provenance:
- Method Matrix [M]: Quantum spin-parity invariants, Poincaré-Hopf topological index,
  Sayvetz-Eckart orthogonal projection, and Marcus electron transfer conditions.
- Derived Relations [D]: IUPAC CIAAW isotopic data structures, CODATA 2022 constants,
  and Level-of-Theory canonical hashing.
- Empirical [E]: Gradient convergence norms (1.0e-4 a.u.) and Kabsch RMSD alignment thresholds (0.25 Å).
"""

from typing import Any, Optional, Sequence, Union


class CoChemQCException(Exception):
    """
    [M] Base exception for all CoChem Quantum Chemistry, thermochemistry, and wavefunction errors.

    All specialized domain exceptions in CoChem inherit from this base class to enable
    coarse-grained error catching or fine-grained diagnostic handling.
    """

    def __init__(self, message: str = "", *args: Any, **kwargs: Any) -> None:
        self.message = message or self._default_message()
        self.diagnostic_info = kwargs
        super().__init__(self.message, *args)

    def _default_message(self) -> str:
        return "A CoChem Quantum Chemistry error occurred."

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class SpinParityViolationError(CoChemQCException):
    """
    [M] Raised when spin-parity constraint (N_e + M) mod 2 != 1 or multiplicity ceiling M - 1 > N_e is violated.

    Mathematical Invariants:
        1. (N_e + M) mod 2 == 1:
           - Even N_e requires odd multiplicity M in {1, 3, 5, ...}.
           - Odd N_e requires even multiplicity M in {2, 4, 6, ...}.
        2. Unpaired electrons constraint: M - 1 <= N_e.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        n_electrons: Optional[int] = None,
        multiplicity: Optional[int] = None,
        charge: Optional[int] = None,
        total_nuclear_charge: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.n_electrons = n_electrons
        self.multiplicity = multiplicity
        self.charge = charge
        self.total_nuclear_charge = total_nuclear_charge

        if message is None:
            if n_electrons is not None and multiplicity is not None:
                if multiplicity - 1 > n_electrons:
                    msg = (
                        f"Spin-parity violation: multiplicity M={multiplicity} requires "
                        f"{multiplicity - 1} unpaired electrons, exceeding total electron count N_e={n_electrons}."
                    )
                else:
                    rem = (n_electrons + multiplicity) % 2
                    msg = (
                        f"Spin-parity violation: (N_e={n_electrons} + M={multiplicity}) mod 2 = {rem} != 1. "
                        f"Total electron count and spin multiplicity are physically incompatible."
                    )
                if charge is not None or total_nuclear_charge is not None:
                    ctx = []
                    if total_nuclear_charge is not None:
                        ctx.append(f"Z_total={total_nuclear_charge}")
                    if charge is not None:
                        ctx.append(f"charge={charge:+d}" if isinstance(charge, int) else f"charge={charge}")
                    msg += f" ({', '.join(ctx)})"
            elif multiplicity is not None:
                msg = f"Spin-parity violation: invalid spin multiplicity M={multiplicity}."
            elif n_electrons is not None:
                msg = f"Spin-parity violation: invalid electron configuration for N_e={n_electrons}."
            else:
                msg = (
                    "Spin-parity violation: Total electron count and spin multiplicity are physically "
                    "incompatible ((N_e + M) mod 2 != 1)."
                )
        else:
            msg = message

        super().__init__(msg, **kwargs)


class UnphysicalValenceError(CoChemQCException):
    """
    [M][D] Raised when an assigned oxidation state exceeds physical electronegativity/valence limits.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        symbol: Optional[str] = None,
        oxidation_state: Optional[Union[int, float]] = None,
        valence_limit: Optional[Union[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        self.symbol = symbol
        self.oxidation_state = oxidation_state
        self.valence_limit = valence_limit

        if message is None:
            if symbol is not None and oxidation_state is not None and valence_limit is not None:
                ox_str = f"{oxidation_state:+d}" if isinstance(oxidation_state, int) else f"{oxidation_state}"
                msg = (
                    f"Unphysical valence state for element '{symbol}': assigned oxidation state {ox_str} "
                    f"exceeds physical valence/electronegativity limit ({valence_limit})."
                )
            elif symbol is not None and oxidation_state is not None:
                ox_str = f"{oxidation_state:+d}" if isinstance(oxidation_state, int) else f"{oxidation_state}"
                msg = f"Unphysical valence state for element '{symbol}': oxidation state {ox_str} is unphysical."
            elif symbol is not None:
                msg = f"Unphysical valence state detected for element '{symbol}'."
            else:
                msg = "Unphysical valence state: oxidation state exceeds physical valence/electronegativity limit."
        else:
            msg = message

        super().__init__(msg, **kwargs)


class LevelOfTheoryMismatchError(CoChemQCException):
    """
    [M][D] Raised when frequency, property, or thermodynamic calculations diverge from the optimization Level of Theory (REQ-QC-085).
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        opt_lot: Optional[Any] = None,
        freq_lot: Optional[Any] = None,
        opt_hash: Optional[str] = None,
        freq_hash: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.opt_lot = opt_lot
        self.freq_lot = freq_lot
        self.opt_hash = opt_hash
        self.freq_hash = freq_hash

        if message is None:
            if opt_lot is not None and freq_lot is not None:
                msg = (
                    f"Level of theory mismatch: frequency/property level of theory '{freq_lot}' "
                    f"diverges from optimization geometry level of theory '{opt_lot}'."
                )
                if opt_hash is not None and freq_hash is not None:
                    msg += f" (Hash mismatch: opt={opt_hash[:8]}... vs freq={freq_hash[:8]}...)"
            elif opt_hash is not None and freq_hash is not None:
                msg = (
                    f"Level of theory hash mismatch: opt SHA-256={opt_hash} != freq SHA-256={freq_hash}."
                )
            else:
                msg = (
                    "Level of theory mismatch: frequency/property evaluation must use the exact identical "
                    "level of theory as preceding geometry optimization."
                )
        else:
            msg = message

        super().__init__(msg, **kwargs)


class NonStationaryFrequencyError(CoChemQCException):
    """
    [M][E] Raised when frequency calculations are attempted on non-stationary points (gradient norm > 1.0e-4 a.u.).
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        gradient_norm: Optional[float] = None,
        threshold: float = 1.0e-4,
        **kwargs: Any,
    ) -> None:
        self.gradient_norm = gradient_norm
        self.threshold = threshold

        if message is None:
            if gradient_norm is not None:
                msg = (
                    f"Non-stationary frequency calculation: gradient norm {gradient_norm:.3e} a.u. "
                    f"exceeds stationary point convergence threshold {threshold:.3e} a.u."
                )
            else:
                msg = (
                    f"Non-stationary frequency calculation: gradient norm exceeds stationary point "
                    f"convergence threshold ({threshold:.1e} a.u.)."
                )
        else:
            msg = message

        super().__init__(msg, **kwargs)


class WavefunctionInstabilityError(CoChemQCException):
    """
    [M] Raised when post-SCF orbital Hessian stability analysis detects negative eigenvalues (lambda_orb < 0).
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        lowest_eigenvalue: Optional[float] = None,
        instability_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.lowest_eigenvalue = lowest_eigenvalue
        self.instability_type = instability_type

        if message is None:
            if lowest_eigenvalue is not None and instability_type is not None:
                msg = (
                    f"Wavefunction instability detected ({instability_type}): lowest orbital Hessian "
                    f"eigenvalue lambda_orb = {lowest_eigenvalue:.6e} < 0."
                )
            elif lowest_eigenvalue is not None:
                msg = (
                    f"Wavefunction instability detected: lowest orbital Hessian eigenvalue "
                    f"lambda_orb = {lowest_eigenvalue:.6e} < 0."
                )
            elif instability_type is not None:
                msg = f"Wavefunction instability detected: {instability_type} mode is unstable."
            else:
                msg = "Wavefunction instability detected: orbital Hessian exhibits negative eigenvalues (lambda_orb < 0)."
        else:
            msg = message

        super().__init__(msg, **kwargs)


class IRCDisconnectionError(CoChemQCException):
    """
    [M][E] Raised when bidirectional IRC endpoints fail to connect to intended reactant/product minima (RMSD > 0.25 Å).
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        endpoint_type: Optional[str] = None,
        rmsd: Optional[float] = None,
        threshold: float = 0.25,
        **kwargs: Any,
    ) -> None:
        self.endpoint_type = endpoint_type
        self.rmsd = rmsd
        self.threshold = threshold

        if message is None:
            if endpoint_type is not None and rmsd is not None:
                msg = (
                    f"IRC trajectory disconnection: {endpoint_type} endpoint RMSD = {rmsd:.4f} Å "
                    f"exceeds connectivity threshold of {threshold:.2f} Å."
                )
            elif rmsd is not None:
                msg = (
                    f"IRC trajectory disconnection: endpoint RMSD = {rmsd:.4f} Å "
                    f"exceeds connectivity threshold of {threshold:.2f} Å."
                )
            elif endpoint_type is not None:
                msg = f"IRC trajectory disconnection: {endpoint_type} endpoint failed to match expected target minimum."
            else:
                msg = (
                    f"IRC trajectory disconnection: bidirectional IRC endpoints failed to connect "
                    f"to intended minima (RMSD > {threshold:.2f} Å)."
                )
        else:
            msg = message

        super().__init__(msg, **kwargs)


class QTAIMTopologyError(CoChemQCException):
    """
    [M] Raised when critical point topology violates the Poincaré-Hopf invariant (n_NCP - n_BCP + n_RCP - n_CCP != 1).
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        n_ncp: Optional[int] = None,
        n_bcp: Optional[int] = None,
        n_rcp: Optional[int] = None,
        n_ccp: Optional[int] = None,
        euler_char: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.n_ncp = n_ncp
        self.n_bcp = n_bcp
        self.n_rcp = n_rcp
        self.n_ccp = n_ccp

        computed_euler: Optional[int] = None
        if n_ncp is not None and n_bcp is not None and n_rcp is not None and n_ccp is not None:
            computed_euler = n_ncp - n_bcp + n_rcp - n_ccp
        self.euler_char = euler_char if euler_char is not None else computed_euler

        if message is None:
            if n_ncp is not None and n_bcp is not None and n_rcp is not None and n_ccp is not None:
                msg = (
                    f"QTAIM topology violation: Poincaré-Hopf invariant violated. "
                    f"(n_NCP={n_ncp} - n_BCP={n_bcp} + n_RCP={n_rcp} - n_CCP={n_ccp} = {self.euler_char}) != 1."
                )
            elif self.euler_char is not None:
                msg = (
                    f"QTAIM topology violation: Poincaré-Hopf Euler characteristic "
                    f"chi = {self.euler_char} != 1."
                )
            else:
                msg = (
                    "QTAIM topology violation: Poincaré-Hopf invariant violated "
                    "(n_NCP - n_BCP + n_RCP - n_CCP != 1)."
                )
        else:
            msg = message

        super().__init__(msg, **kwargs)


class MarcusConvergenceError(CoChemQCException):
    """
    [M] Raised when 4-point Nelson state machine calculations fail to converge during Marcus reorganization pipeline.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        step_name: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.step_name = step_name
        self.reason = reason

        if message is None:
            if step_name is not None and reason is not None:
                msg = f"Marcus 4-point calculation failed at step '{step_name}': {reason}."
            elif step_name is not None:
                msg = f"Marcus 4-point calculation failed at step '{step_name}'."
            elif reason is not None:
                msg = f"Marcus 4-point calculation failed to converge: {reason}."
            else:
                msg = "Marcus 4-point state machine calculation failed to converge."
        else:
            msg = message

        super().__init__(msg, **kwargs)


class IsotopeNotFoundError(CoChemQCException):
    """
    [D] Raised when an isotope mass number is absent in IUPAC CIAAW dataset.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        symbol: Optional[str] = None,
        mass_number: Optional[int] = None,
        available_mass_numbers: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        self.symbol = symbol
        self.mass_number = mass_number
        self.available_mass_numbers = list(available_mass_numbers) if available_mass_numbers is not None else None

        if message is None:
            if symbol is not None and mass_number is not None and self.available_mass_numbers:
                msg = (
                    f"Isotope '^{mass_number}{symbol}' not found in IUPAC CIAAW isotopic database. "
                    f"Available mass numbers for {symbol}: {sorted(self.available_mass_numbers)}."
                )
            elif symbol is not None and mass_number is not None:
                msg = f"Isotope '^{mass_number}{symbol}' not found in IUPAC CIAAW isotopic database."
            elif symbol is not None:
                msg = f"Isotope for element '{symbol}' not found in IUPAC CIAAW isotopic database."
            else:
                msg = "Isotope not found in IUPAC CIAAW isotopic database."
        else:
            msg = message

        super().__init__(msg, **kwargs)
