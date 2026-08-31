"""
CoChem Interactive Dynamic Mendeleev Isotope Labeler & KIE Predictor Widget (REQ-QC-086).

Provenance:
- Method Matrix [M]: Interactive isotopic substitution pattern selector, live Bigeleisen-Mayer KIE evaluation.
- Derived Relations [D]: Dynamic Mendeleev atomic and isotopic mass lookups.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import get_ciaaw_atomic_mass, get_ciaaw_isotope_mass
from cochem.qc.cochem_isotopes import compute_isotopic_substitution_kie
from cochem.qc.models import IsotopeSubstitutionPayload


class IsotopeLabelerState(BaseModel):
    """Data contract representing the active isotopic substitution selection and preview."""
    model_config = ConfigDict(frozen=True)

    symbols: List[str] = Field(default_factory=list)
    substituted_atoms: Dict[int, int] = Field(default_factory=dict)
    reference_zpe_kcal_mol: float = 0.0
    substituted_zpe_kcal_mol: float = 0.0
    predicted_kie: float = 1.0
    temperature_k: float = 298.15
    status_message: str = "Ready"


class IsotopeLabelerWidget:
    """
    [REQ-QC-086] Interactive Isotopic Substitution Selector and Real-Time KIE Predictor.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        cartesian_hessian: Optional[Sequence[Sequence[float]]] = None,
        temperature_k: float = 298.15,
    ) -> None:
        self.symbols = list(symbols)
        self.cartesian_hessian = cartesian_hessian
        self.temperature_k = temperature_k
        self.substitutions: Dict[int, int] = {}
        self._state = IsotopeLabelerState(
            symbols=self.symbols,
            temperature_k=temperature_k,
        )

    def set_isotope(self, atom_index: int, mass_number: int) -> IsotopeLabelerState:
        """
        Assign an isotopic mass number to a specific atom index and recompute KIE.
        """
        if atom_index < 0 or atom_index >= len(self.symbols):
            raise ValueError(f"Atom index {atom_index} out of bounds [0, {len(self.symbols)-1}].")

        sym = self.symbols[atom_index]
        # Validate that isotope exists in mendeleev
        _ = get_ciaaw_isotope_mass(sym, mass_number)

        self.substitutions[atom_index] = mass_number
        return self._recompute()

    def clear_isotope(self, atom_index: int) -> IsotopeLabelerState:
        """Remove custom isotopic substitution on an atom."""
        if atom_index in self.substitutions:
            del self.substitutions[atom_index]
        return self._recompute()

    def clear_all(self) -> IsotopeLabelerState:
        """Reset all atoms to standard natural abundance atomic weights."""
        self.substitutions.clear()
        return self._recompute()

    def _recompute(self) -> IsotopeLabelerState:
        if self.cartesian_hessian is None or not self.substitutions:
            self._state = IsotopeLabelerState(
                symbols=self.symbols,
                substituted_atoms=dict(self.substitutions),
                reference_zpe_kcal_mol=0.0,
                substituted_zpe_kcal_mol=0.0,
                predicted_kie=1.0,
                temperature_k=self.temperature_k,
                status_message="No substitutions or Hessian not loaded.",
            )
            return self._state

        payload = compute_isotopic_substitution_kie(
            symbols=self.symbols,
            cartesian_hessian_hartree_bohr2=self.cartesian_hessian,
            substituted_atoms=self.substitutions,
            temperature_k=self.temperature_k,
        )

        self._state = IsotopeLabelerState(
            symbols=self.symbols,
            substituted_atoms=dict(self.substitutions),
            reference_zpe_kcal_mol=payload.reference_zpe_kcal_mol,
            substituted_zpe_kcal_mol=payload.substituted_zpe_kcal_mol,
            predicted_kie=payload.kie_bigeleisen_mayer or 1.0,
            temperature_k=self.temperature_k,
            status_message=f"KIE computed: {payload.kie_bigeleisen_mayer:.4f}",
        )
        return self._state

    def get_state(self) -> IsotopeLabelerState:
        return self._state
