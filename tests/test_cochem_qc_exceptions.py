"""
Unit tests for CoChem Quantum Chemistry Domain Exception Hierarchy.

Strictly adhering to:
- Zero-Mock Mandate (zero mocks, zero stubs, zero synthetic arrays, real exception objects).
- Provenance tracking [M], [D], [E].
"""

import pytest

import cochem.qc as cochem_qc
from cochem.qc.exceptions import (
    CoChemQCException,
    IRCDisconnectionError,
    IsotopeNotFoundError,
    LevelOfTheoryMismatchError,
    MarcusConvergenceError,
    NonStationaryFrequencyError,
    QTAIMTopologyError,
    SpinParityViolationError,
    UnphysicalValenceError,
    WavefunctionInstabilityError,
)


class TestCoChemQCExceptions:
    """Test suite for CoChem QC domain exceptions hierarchy."""

    SPECIALIZED_CLASSES = [
        SpinParityViolationError,
        UnphysicalValenceError,
        LevelOfTheoryMismatchError,
        NonStationaryFrequencyError,
        WavefunctionInstabilityError,
        IRCDisconnectionError,
        QTAIMTopologyError,
        MarcusConvergenceError,
        IsotopeNotFoundError,
    ]

    def test_all_exceptions_inherit_from_base(self) -> None:
        """[M] Verify all 9 specialized domain exceptions inherit from CoChemQCException and Exception."""
        for exc_cls in self.SPECIALIZED_CLASSES:
            assert issubclass(exc_cls, CoChemQCException)
            assert issubclass(exc_cls, Exception)

    def test_polymorphic_catch_all_exceptions(self) -> None:
        """[M] Verify that raising any specialized exception can be caught by CoChemQCException."""
        for exc_cls in self.SPECIALIZED_CLASSES:
            caught = False
            try:
                raise exc_cls()
            except CoChemQCException as e:
                caught = True
                assert isinstance(e, exc_cls)
                assert isinstance(e, CoChemQCException)
                assert isinstance(e, Exception)
            assert caught, f"Failed to catch {exc_cls.__name__} via CoChemQCException base."

    def test_module_reexport_integrity(self) -> None:
        """[D] Verify cochem.qc re-exports all 10 exception classes in __all__."""
        expected_symbols = [
            "CoChemQCException",
            "SpinParityViolationError",
            "UnphysicalValenceError",
            "LevelOfTheoryMismatchError",
            "NonStationaryFrequencyError",
            "WavefunctionInstabilityError",
            "IRCDisconnectionError",
            "QTAIMTopologyError",
            "MarcusConvergenceError",
            "IsotopeNotFoundError",
        ]
        for sym in expected_symbols:
            assert hasattr(cochem_qc, sym), f"cochem.qc missing export {sym}"
            assert sym in cochem_qc.__all__, f"{sym} not in cochem.qc.__all__"

    def test_base_exception_instantiation_and_details(self) -> None:
        """Verify base exception message, repr, and kwargs details."""
        exc = CoChemQCException("Base QC calculation failed.", code=500, module="qc")
        assert str(exc) == "Base QC calculation failed."
        assert exc.message == "Base QC calculation failed."
        assert exc.diagnostic_info["code"] == 500
        assert exc.diagnostic_info["module"] == "qc"
        assert repr(exc) == "CoChemQCException('Base QC calculation failed.')"

        # Default message fallback
        default_exc = CoChemQCException()
        assert "A CoChem Quantum Chemistry error occurred." in str(default_exc)
        assert repr(default_exc) == "CoChemQCException('A CoChem Quantum Chemistry error occurred.')"

    def test_spin_parity_violation_error(self) -> None:
        """[M] Verify SpinParityViolationError diagnostic formatting and attribute storage."""
        # Case 1: Incompatible parity (even electrons, even multiplicity)
        exc = SpinParityViolationError(n_electrons=10, multiplicity=2, charge=0, total_nuclear_charge=10)
        assert exc.n_electrons == 10
        assert exc.multiplicity == 2
        assert exc.charge == 0
        assert exc.total_nuclear_charge == 10
        assert "Spin-parity violation" in str(exc)
        assert "(N_e=10 + M=2) mod 2 = 0 != 1" in str(exc)
        assert "Z_total=10" in str(exc)
        assert "charge=+0" in str(exc)

        # Case 2: Incompatible parity (odd electrons, odd multiplicity)
        exc_odd = SpinParityViolationError(n_electrons=9, multiplicity=1)
        assert "(N_e=9 + M=1) mod 2 = 0 != 1" in str(exc_odd)

        # Case 3: Multiplicity exceeds electron count ceiling
        exc_ceiling = SpinParityViolationError(n_electrons=2, multiplicity=5)
        assert "multiplicity M=5 requires 4 unpaired electrons, exceeding total electron count N_e=2" in str(exc_ceiling)

        # Case 4: Partial parameters & Custom message & Default
        assert "invalid spin multiplicity M=3" in str(SpinParityViolationError(multiplicity=3))
        assert "invalid electron configuration for N_e=5" in str(SpinParityViolationError(n_electrons=5))
        assert "Custom spin failure" in str(SpinParityViolationError("Custom spin failure"))
        assert "physically incompatible" in str(SpinParityViolationError())

        # Base catching
        with pytest.raises(CoChemQCException):
            raise SpinParityViolationError(n_electrons=9, multiplicity=1)

    def test_unphysical_valence_error(self) -> None:
        """[M][D] Verify UnphysicalValenceError diagnostic message and limits."""
        exc = UnphysicalValenceError(symbol="C", oxidation_state=6, valence_limit=4)
        assert exc.symbol == "C"
        assert exc.oxidation_state == 6
        assert exc.valence_limit == 4
        assert "Unphysical valence state for element 'C'" in str(exc)
        assert "+6" in str(exc)
        assert "limit (4)" in str(exc)

        # Fallbacks
        assert "oxidation state +5 is unphysical" in str(UnphysicalValenceError(symbol="N", oxidation_state=5))
        assert "detected for element 'Fe'" in str(UnphysicalValenceError(symbol="Fe"))
        assert "Unphysical valence state" in str(UnphysicalValenceError())
        assert "Manual valence error" in str(UnphysicalValenceError("Manual valence error"))

        with pytest.raises(CoChemQCException):
            raise UnphysicalValenceError(symbol="O", oxidation_state=-3)

    def test_level_of_theory_mismatch_error(self) -> None:
        """[M][D] Verify LevelOfTheoryMismatchError with methods or SHA hashes."""
        exc_methods = LevelOfTheoryMismatchError(
            opt_lot="B3LYP/def2-SVP",
            freq_lot="wB97X-D/def2-TZVP",
            opt_hash="a1b2c3d4e5f600112233",
            freq_hash="f6e5d4c3b2a199887766",
        )
        assert exc_methods.opt_lot == "B3LYP/def2-SVP"
        assert exc_methods.freq_lot == "wB97X-D/def2-TZVP"
        assert exc_methods.opt_hash == "a1b2c3d4e5f600112233"
        assert exc_methods.freq_hash == "f6e5d4c3b2a199887766"
        assert "Level of theory mismatch" in str(exc_methods)
        assert "B3LYP/def2-SVP" in str(exc_methods)
        assert "wB97X-D/def2-TZVP" in str(exc_methods)
        assert "Hash mismatch: opt=a1b2c3d4... vs freq=f6e5d4c3..." in str(exc_methods)

        exc_hash_only = LevelOfTheoryMismatchError(opt_hash="1111", freq_hash="2222")
        assert "Level of theory hash mismatch" in str(exc_hash_only)
        assert "Level of theory mismatch" in str(LevelOfTheoryMismatchError())
        assert "Custom LoT mismatch" in str(LevelOfTheoryMismatchError("Custom LoT mismatch"))

        with pytest.raises(CoChemQCException):
            raise LevelOfTheoryMismatchError("LoT divergent")

    def test_non_stationary_frequency_error(self) -> None:
        """[M][E] Verify NonStationaryFrequencyError gradient reporting."""
        exc = NonStationaryFrequencyError(gradient_norm=2.5e-4, threshold=1.0e-4)
        assert exc.gradient_norm == 2.5e-4
        assert exc.threshold == 1.0e-4
        assert "Non-stationary frequency calculation" in str(exc)
        assert "2.500e-04 a.u." in str(exc)
        assert "1.000e-04 a.u." in str(exc)

        assert "Non-stationary frequency calculation" in str(NonStationaryFrequencyError())
        assert "Unconverged gradient" in str(NonStationaryFrequencyError("Unconverged gradient"))

        with pytest.raises(CoChemQCException):
            raise NonStationaryFrequencyError(gradient_norm=5.0e-3)

    def test_wavefunction_instability_error(self) -> None:
        """[M] Verify WavefunctionInstabilityError with orbital Hessian eigenvalue."""
        exc = WavefunctionInstabilityError(
            lowest_eigenvalue=-0.042,
            instability_type="UNSTABLE_INTERNAL_RHF_UHF",
        )
        assert exc.lowest_eigenvalue == -0.042
        assert exc.instability_type == "UNSTABLE_INTERNAL_RHF_UHF"
        assert "Wavefunction instability detected (UNSTABLE_INTERNAL_RHF_UHF)" in str(exc)
        assert "-4.200000e-02" in str(exc)

        assert "-1.000000e-03" in str(WavefunctionInstabilityError(lowest_eigenvalue=-1e-3))
        assert "RHF_UHF mode is unstable" in str(WavefunctionInstabilityError(instability_type="RHF_UHF"))
        assert "Wavefunction instability detected" in str(WavefunctionInstabilityError())
        assert "Custom stability issue" in str(WavefunctionInstabilityError("Custom stability issue"))

        with pytest.raises(CoChemQCException):
            raise WavefunctionInstabilityError(lowest_eigenvalue=-1e-3)

    def test_irc_disconnection_error(self) -> None:
        """[M][E] Verify IRCDisconnectionError with endpoint RMSD metrics."""
        exc = IRCDisconnectionError(
            endpoint_type="Forward",
            rmsd=0.485,
            threshold=0.25,
        )
        assert exc.endpoint_type == "Forward"
        assert exc.rmsd == 0.485
        assert exc.threshold == 0.25
        assert "IRC trajectory disconnection: Forward endpoint RMSD = 0.4850 Å" in str(exc)
        assert "0.25 Å" in str(exc)

        assert "endpoint RMSD = 0.3500 Å" in str(IRCDisconnectionError(rmsd=0.35))
        assert "Reverse endpoint failed to match" in str(IRCDisconnectionError(endpoint_type="Reverse"))
        assert "IRC trajectory disconnection" in str(IRCDisconnectionError())
        assert "Custom path disconnection" in str(IRCDisconnectionError("Custom path disconnection"))

        with pytest.raises(CoChemQCException):
            raise IRCDisconnectionError(endpoint_type="Reverse", rmsd=0.35)

    def test_qtaim_topology_error(self) -> None:
        """[M] Verify QTAIMTopologyError Poincaré-Hopf Euler characteristic."""
        exc = QTAIMTopologyError(n_ncp=3, n_bcp=1, n_rcp=0, n_ccp=0)
        assert exc.n_ncp == 3
        assert exc.n_bcp == 1
        assert exc.n_rcp == 0
        assert exc.n_ccp == 0
        assert exc.euler_char == 2
        assert "Poincaré-Hopf invariant violated" in str(exc)
        assert "n_NCP=3 - n_BCP=1 + n_RCP=0 - n_CCP=0 = 2" in str(exc)

        assert "chi = 3 != 1" in str(QTAIMTopologyError(euler_char=3))
        assert "Poincaré-Hopf invariant violated" in str(QTAIMTopologyError())
        assert "Custom topological violation" in str(QTAIMTopologyError("Custom topological violation"))

        with pytest.raises(CoChemQCException):
            raise QTAIMTopologyError(n_ncp=4, n_bcp=4, n_rcp=0, n_ccp=0)

    def test_marcus_convergence_error(self) -> None:
        """[M] Verify MarcusConvergenceError with step name and diagnostic reason."""
        exc = MarcusConvergenceError(
            step_name="charged_geometry_opt",
            reason="SCF cycles exceeded maximum limit (150 cycles)",
        )
        assert exc.step_name == "charged_geometry_opt"
        assert exc.reason == "SCF cycles exceeded maximum limit (150 cycles)"
        assert "Marcus 4-point calculation failed at step 'charged_geometry_opt': SCF cycles exceeded maximum limit (150 cycles)." in str(exc)

        assert "failed at step 'opt_neutral'" in str(MarcusConvergenceError(step_name="opt_neutral"))
        assert "failed to converge: Divergent energy" in str(MarcusConvergenceError(reason="Divergent energy"))
        assert "failed to converge" in str(MarcusConvergenceError())
        assert "Custom Marcus error" in str(MarcusConvergenceError("Custom Marcus error"))

        with pytest.raises(CoChemQCException):
            raise MarcusConvergenceError(step_name="neutral_sp", reason="Divergent energy")

    def test_isotope_not_found_error(self) -> None:
        """[D] Verify IsotopeNotFoundError with element symbol, mass number, and available isotopes."""
        exc = IsotopeNotFoundError(symbol="C", mass_number=99, available_mass_numbers=[12, 13, 14])
        assert exc.symbol == "C"
        assert exc.mass_number == 99
        assert exc.available_mass_numbers == [12, 13, 14]
        assert "Isotope '^99C' not found in IUPAC CIAAW isotopic database." in str(exc)
        assert "Available mass numbers for C: [12, 13, 14]" in str(exc)

        assert "Isotope '^99C' not found" in str(IsotopeNotFoundError(symbol="C", mass_number=99))
        assert "Isotope for element 'H' not found" in str(IsotopeNotFoundError(symbol="H"))
        assert "Isotope not found in IUPAC CIAAW" in str(IsotopeNotFoundError())
        assert "Custom isotope failure" in str(IsotopeNotFoundError("Custom isotope failure"))

        with pytest.raises(CoChemQCException):
            raise IsotopeNotFoundError(symbol="H", mass_number=10)
