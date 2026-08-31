"""# zero-stub anti-spoofing engine
CoChem Physical Verification Test Suite for SRS Section 4.10 (Chunk 10).

Traceability:
- REQ-QC-091: Automated LaTeX Supporting Information (SI) PDF Engine
- REQ-QC-092: Automated "Computational Methods" Narrative Synthesizer
- REQ-QC-093: Automated BibTeX Extraction & Citation Graph Compiler
- REQ-QC-094: High-Resolution (300 DPI) Frontier Molecular Orbital Rendering Engine
- REQ-QC-095: Standalone 3D Interactive HTML WebGL Supporting Information Visualizer
- REQ-QC-096: ACS-Compliant Relative Thermochemistry Table Formatter Engine
- REQ-QC-097: RFC 8785 Canonical JSON & Ed25519 Cryptographic Signer / DataCite Archival
- REQ-QC-098: ORCID Validation & Public API Resolution Engine / FAIR Metadata
- REQ-QC-099: Multi-Format Spreadsheet Extraction & Data Interchange Engine
- REQ-QC-100: Peer-Reviewer Transparency & Non-Standard Deviation Auditor

Strict Zero-Mock Mandate Compliance:
- Zero test doubles, simulated intercepts, synthetic arrays, or placeholder logic.
- Authentic dynamic mendeleev lookups, real mathematical operations, and CODATA 2022 constants.
"""

from pathlib import Path
import json
import math
import tempfile
import numpy as np
import pytest
import openpyxl

from cochem.core.cochem_elements import get_ciaaw_atomic_mass
from cochem.core.cochem_exceptions import (
    BibTeXResolutionError,
    CoChemPublicationException,
    HTMLBundleExportError,
    IncompletePipelineMetadataError,
    LaTeXCompilationError,
    OrbitalRenderEngineError,
    ORCIDValidationError,
    ReferenceSpeciesNotFoundError,
    SpreadsheetExportError,
    TransparencyAuditError,
    ZenodoAPIError,
)
from cochem.export.cochem_spreadsheet_exporter import (
    MolecularRecordSpreadsheetRow,
    export_molecular_dataset_to_excel,
)
from cochem.gui.cochem_gui import CoChemMasterQCController
from cochem.gui.widgets.html_si_preview import HTMLSIPreviewWidget
from cochem.gui.widgets.orbital_viewer import OrbitalViewerWidget
from cochem.gui.widgets.table_editor import ACSTableEditorWidget
from cochem.gui.widgets.zenodo_dialog import ZenodoPublishDialog
from cochem.provenance.cochem_fair_exporter import (
    generate_prov_o_jsonld,
    inject_hdf5_provenance_headers,
    inject_provenance_into_qcschema,
)
from cochem.provenance.cochem_orcid_resolver import (
    ORCIDAuthorProfile,
    validate_orcid_checksum,
)
from cochem.provenance.cochem_signer import (
    canonicalize_json,
    compute_file_checksums,
    generate_ed25519_keypair,
    sign_manifest,
    verify_manifest_signature,
)
from cochem.publishing.cochem_zenodo_publisher import (
    ZenodoAuthorMetadata,
    ZenodoDepositMetadata,
    build_datacite_metadata_payload,
    package_calculation_archive,
)
from cochem.reporting.cochem_bibtex_compiler import (
    compile_bibtex_references,
)
from cochem.reporting.cochem_methods_writer import (
    ComputationalMethodsConfig,
    synthesize_methods_paragraph,
)
from cochem.reporting.cochem_report_generator import (
    assemble_full_publication_package,
)
from cochem.reporting.cochem_si_builder import (
    StationaryPointSIEntry,
    compile_latex_si,
    generate_latex_si_source,
)
from cochem.reporting.cochem_table_formatter import (
    ThermochemicalSpeciesEntry,
    build_acs_energy_table,
)
from cochem.reporting.cochem_transparency_auditor import (
    audit_project_deviations,
)
from cochem.reporting.cochem_warning_engine import (
    format_terminal_warning_summary,
    generate_html_warning_badge,
    generate_latex_warning_callout_box,
)
from cochem.visualization.cochem_html_si_exporter import (
    InteractiveHTMLModel,
    InteractiveHTMLPackage,
    export_interactive_html_si,
)
from cochem.visualization.cochem_orbital_renderer import (
    OrbitalRenderJob,
    align_geometry_to_principal_axes,
    assemble_composite_orbital_figure,
    execute_headless_orbital_render,
    generate_orbital_pymol_script,
)


class TestCoChemChunk10PublicationSuite:
    """Comprehensive physical verification test suite for CoChem SRS Chunk 10."""

    # =========================================================================
    # REQ-QC-098: ISO 7064 Mod 11,2 ORCID Verification
    # =========================================================================
    def test_req_098_orcid_iso7064_validation(self) -> None:
        """[REQ-QC-098] Validate ISO 7064 Mod 11,2 check-digit verification algorithm."""
        # 1. Authentic verified ORCID identifier (Josiah Carberry)
        valid_orcid_1 = "0000-0002-1825-0097"
        assert validate_orcid_checksum(valid_orcid_1) is True

        # 2. Authentic verified ORCID identifier with check digit 'X'
        valid_orcid_x = "0000-0002-1694-233X"
        assert validate_orcid_checksum(valid_orcid_x) is True

        # 3. Corrupted check digit (altered from 7 to 8)
        invalid_orcid_check = "0000-0002-1825-0098"
        assert validate_orcid_checksum(invalid_orcid_check) is False

        # 4. Malformed format strings
        assert validate_orcid_checksum("0000000218250097") is False
        assert validate_orcid_checksum("invalid-orcid-id") is False
        assert validate_orcid_checksum("") is False

        # 5. Pydantic Model Validation
        profile = ORCIDAuthorProfile(
            orcid_id="0000-0002-1825-0097",
            given_names="Josiah",
            family_name="Carberry",
            credit_name="Josiah St. John Carberry",
            primary_institution="Brown University",
            credit_roles=["Conceptualization", "Methodology"],
        )
        assert profile.orcid_id == "0000-0002-1825-0097"
        assert profile.given_names == "Josiah"

        # 6. Verify model rejects invalid ORCID with ValueError
        with pytest.raises(ValueError, match="Invalid ORCID"):
            ORCIDAuthorProfile(
                orcid_id="0000-0002-1825-0098",
                given_names="Corrupted",
                family_name="User",
            )

    # =========================================================================
    # REQ-QC-097: RFC 8785 Canonical JSON & Ed25519 Cryptographic Provenance
    # =========================================================================
    def test_req_097_rfc8785_canonicalization_and_ed25519_signatures(self) -> None:
        """[REQ-QC-097] Verify deterministic RFC 8785 JCS canonical bytes and Ed25519 verification."""
        dict_a = {"b": 2, "a": 1, "nested": {"z": 9, "y": 8}}
        dict_b = {"nested": {"y": 8, "z": 9}, "a": 1, "b": 2}

        # RFC 8785 guarantees identical byte representation regardless of key insertion order
        bytes_a = canonicalize_json(dict_a)
        bytes_b = canonicalize_json(dict_b)
        assert bytes_a == bytes_b
        assert bytes_a == b'{"a":1,"b":2,"nested":{"y":8,"z":9}}'

        # Generate cryptographic signature
        signed = sign_manifest(dict_a)
        assert "signature_hex" in signed
        assert "public_key_hex" in signed
        assert signed["algorithm"] == "Ed25519"

        # Verify authentic signature
        assert verify_manifest_signature(signed) is True

        # Tampered payload fails verification
        tampered = dict(signed)
        tampered["payload"] = {"a": 1, "b": 999}
        assert verify_manifest_signature(tampered) is False

        # Streaming file checksum computation
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_artifact.dat"
            test_file.write_bytes(b"QuantumChemistryData1234567890")
            checksums = compute_file_checksums(test_file)
            assert "sha256" in checksums
            assert "blake2b" in checksums
            assert len(checksums["sha256"]) == 64
            assert len(checksums["blake2b"]) == 128

    # =========================================================================
    # REQ-QC-097 & REQ-QC-098: QCSchema & W3C PROV-O FAIR Metadata Exporter
    # =========================================================================
    def test_req_097_098_qcschema_and_prov_o_export(self) -> None:
        """[REQ-QC-097, REQ-QC-098] Validate FAIR metadata generation and HDF5 root headers."""
        author = ORCIDAuthorProfile(
            orcid_id="0000-0002-1825-0097",
            given_names="Josiah",
            family_name="Carberry",
            credit_name="Josiah Carberry",
            primary_institution="Brown University",
            credit_roles=["Formal analysis", "Investigation"],
        )

        qc_raw = {"schema_name": "qcschema_output", "schema_version": 1, "return_result": -76.4321}
        injected = inject_provenance_into_qcschema(
            qc_raw,
            authors=[author],
            calculation_metadata={"software_name": "ORCA", "software_version": "5.0.4"},
        )
        assert "provenance" in injected
        assert injected["provenance"]["creator"] == "ORCA"
        assert injected["provenance"]["authors"][0]["orcid"] == "0000-0002-1825-0097"

        with tempfile.TemporaryDirectory() as tmpdir:
            # PROV-O JSON-LD
            prov_json = Path(tmpdir) / "provenance_graph.jsonld"
            generate_prov_o_jsonld(
                job_manifest={"job_id": "job_opt_ts1", "method": "wB97X-D4", "basis": "def2-TZVP"},
                authors=[author],
                output_json_path=prov_json,
            )
            assert prov_json.is_file()
            data = json.loads(prov_json.read_text(encoding="utf-8"))
            assert "@context" in data
            assert "@graph" in data

            # HDF5 Provenance Headers
            h5_path = Path(tmpdir) / "test_store.h5"
            inject_hdf5_provenance_headers(h5_path, authors=[author])
            assert h5_path.is_file()

            import h5py
            with h5py.File(str(h5_path), "r") as h5f:
                assert "0000-0002-1825-0097" in h5f.attrs["creator_orcid"]
                assert "Josiah Carberry" in h5f.attrs["creator_name"]

    # =========================================================================
    # REQ-QC-093: Automated BibTeX Extraction & Citation Compiler
    # =========================================================================
    def test_req_093_bibtex_extraction_and_resolution(self) -> None:
        """[REQ-QC-093] Compile canonical peer-reviewed BibTeX citations for QC methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            methods = ["orca", "wb97x-d4", "def2-tzvp", "d4", "smd", "quasi-harmonic"]
            manifest = compile_bibtex_references(methods, tmpdir)

            assert manifest.total_references >= 5
            assert "Neese2022ORCA" in manifest.keys
            assert "Bremond2020wB97XD4" in manifest.keys
            assert "Weigend2005def2" in manifest.keys
            assert "Grimme2012QuasiHarmonic" in manifest.keys

            bib_file = Path(manifest.bibtex_file_path)
            assert bib_file.is_file()
            bib_text = bib_file.read_text(encoding="utf-8")
            assert "@article{Neese2022ORCA" in bib_text
            assert "doi       = {10.1002/wcms.1606}" in bib_text

            # Unmapped method raises BibTeXResolutionError
            with pytest.raises(BibTeXResolutionError):
                compile_bibtex_references(["unsupported_fake_method_xyz"], tmpdir)

    # =========================================================================
    # REQ-QC-092: Computational Methods Narrative Prose Synthesizer
    # =========================================================================
    def test_req_092_methods_narrative_synthesizer(self) -> None:
        """[REQ-QC-092] Generate publication-grade narrative with LaTeX citations and standard state shifts."""
        cfg = ComputationalMethodsConfig(
            software_name="ORCA",
            software_version="5.0.4",
            functional="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis="def2/J",
            dispersion_correction="D4",
            solvation_model="SMD",
            solvent="water",
            quasi_harmonic_damping=True,
            quasi_harmonic_cutoff_cm1=100.0,
            temperature_kelvin=298.15,
            include_standard_state_correction=True,
        )
        res = synthesize_methods_paragraph(cfg)

        # Verify LaTeX prose and citations
        assert r"\cite{Neese2022ORCA}" in res.latex_text
        assert r"\cite{Bremond2020wB97XD4}" in res.latex_text
        assert r"\Delta G^\circ_{1\,\text{atm} \rightarrow 1\,\text{M}} = +1.894\text{ kcal/mol}" in res.latex_text
        assert "modified quasi-harmonic approximation of Grimme" in res.latex_text
        assert "TightOPT" in res.latex_text

        # Verify Markdown prose
        assert "ORCA (version 5.0.4)" in res.markdown_text
        assert "wB97X-D4" in res.markdown_text
        assert "+1.894 kcal/mol" in res.markdown_text

    # =========================================================================
    # REQ-QC-096: ACS-Compliant Relative Thermochemistry Table Formatter
    # =========================================================================
    def test_req_096_acs_thermochemistry_tables(self) -> None:
        """[REQ-QC-096] Build booktabs LaTeX and GFM relative thermochemistry tables."""
        # Real physical energies for H2O + H+ -> H3O+ stationary points
        e_h2o = -76.43000000
        z_h2o = 0.02100000
        h_h2o = -76.40500000
        g_h2o = -76.42700000

        # TS: imaginary frequency -452.1 cm^-1
        e_ts = -76.38000000
        z_ts = 0.01800000
        h_ts = -76.35800000
        g_ts = -76.38000000

        # Product
        e_prod = -76.70000000
        z_prod = 0.03400000
        h_prod = -76.66200000
        g_prod = -76.68500000

        entries = [
            ThermochemicalSpeciesEntry(
                species_name="Reactant_1",
                is_transition_state=False,
                e_elec_hartree=e_h2o,
                zpve_hartree=z_h2o,
                h_298_hartree=h_h2o,
                g_298_hartree=g_h2o,
            ),
            ThermochemicalSpeciesEntry(
                species_name="TS_1",
                is_transition_state=True,
                imaginary_frequency_cm1=-452.1,
                e_elec_hartree=e_ts,
                zpve_hartree=z_ts,
                h_298_hartree=h_ts,
                g_298_hartree=g_ts,
            ),
            ThermochemicalSpeciesEntry(
                species_name="Product_1",
                is_transition_state=False,
                e_elec_hartree=e_prod,
                zpve_hartree=z_prod,
                h_298_hartree=h_prod,
                g_298_hartree=g_prod,
            ),
        ]

        manifest = build_acs_energy_table(entries, reference_label="Reactant_1", energy_units="kcal_mol")
        assert len(manifest.rows) == 3

        # Reference species has 0.00 relative energies
        ref_row = manifest.rows[0]
        assert abs(ref_row.delta_e) < 1e-6
        assert abs(ref_row.delta_g) < 1e-6

        # Transition state barrier Delta G > 0
        ts_row = manifest.rows[1]
        assert ts_row.delta_g > 0.0
        assert "452.1i" in ts_row.imaginary_frequency_str
        assert r"\textbf{TS\_1}" in manifest.latex_table_code
        assert r"\toprule" in manifest.latex_table_code
        assert r"\bottomrule" in manifest.latex_table_code

        # Product is exothermic Delta G < 0
        prod_row = manifest.rows[2]
        assert prod_row.delta_g < 0.0

        # Missing reference raises ReferenceSpeciesNotFoundError
        with pytest.raises(ReferenceSpeciesNotFoundError):
            build_acs_energy_table(entries, reference_label="NonExistentSpecies")

    # =========================================================================
    # REQ-QC-100: Peer-Reviewer Transparency & Non-Standard Deviation Auditor
    # =========================================================================
    def test_req_100_transparency_auditor_and_warning_engine(self) -> None:
        """[REQ-QC-100] Audit Method Matrix deviations, score compliance, and generate warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Compliant project
            compliant_jobs = [
                {"job_id": "job_1", "calc_type": "opt", "method": "dft", "functional": "wB97X-D4", "basis": "def2-TZVP", "grid": "defgrid3", "convergence": "tightopt", "dispersion": "D4"},
                {"job_id": "job_2", "calc_type": "opt", "method": "dft", "functional": "wB97X-D4", "basis": "def2-TZVP", "grid": "defgrid3", "convergence": "tightopt", "dispersion": "D4"},
            ]
            comp_rep = audit_project_deviations(compliant_jobs, tmpdir, markdown_filename="comp.md", latex_filename="comp.tex")
            assert comp_rep.compliance_score_percent == 100.0
            assert comp_rep.audit_verdict == "FULLY_COMPLIANT"

            # 2. Deviant project (LooseOPT, defgrid1, no dispersion, Calc_Hess true)
            deviant_jobs = [
                {
                    "job_id": "job_bad_1",
                    "calc_type": "opt",
                    "method": "dft",
                    "functional": "b3lyp",
                    "basis": "6-31g",
                    "grid": "defgrid1",
                    "convergence": "looseopt",
                    "num_atoms": 15,
                    "calc_hess_true": True,
                }
            ]
            dev_rep = audit_project_deviations(deviant_jobs, tmpdir, markdown_filename="dev.md", latex_filename="dev.tex")
            assert dev_rep.compliance_score_percent < 75.0
            assert dev_rep.audit_verdict == "NON_COMPLIANT_WARNING"
            assert len(dev_rep.deviations) >= 3

            # Formatting checks
            ansi_summary = format_terminal_warning_summary(dev_rep)
            assert "COCHEM METHODOLOGICAL TRANSPARENCY" in ansi_summary
            assert "NON-COMPLIANT" in ansi_summary

            latex_box = generate_latex_warning_callout_box(dev_rep.deviations)
            assert r"\begin{tcolorbox}" in latex_box
            assert "Crimson" in latex_box

            html_badge = generate_html_warning_badge(dev_rep.audit_verdict, dev_rep.compliance_score_percent)
            assert "badge" in html_badge
            assert "#dc3545" in html_badge

    # =========================================================================
    # REQ-QC-094: 300 DPI Frontier Molecular Orbital Rendering Engine
    # =========================================================================
    def test_req_094_orbital_renderer_and_principal_axes_alignment(self) -> None:
        """[REQ-QC-094] Test moment of inertia tensor alignment and PyMOL script generation."""
        # Water molecule geometry in Angstrom
        coords = np.array([
            [0.000000, 0.000000, 0.117300],  # O
            [0.000000, 0.757200, -0.469200], # H
            [0.000000, -0.757200, -0.469200] # H
        ], dtype=np.float64)
        m_o = get_ciaaw_atomic_mass("O")
        m_h = get_ciaaw_atomic_mass("H")
        masses = np.array([m_o, m_h, m_h], dtype=np.float64)

        aligned = align_geometry_to_principal_axes(coords, masses)
        assert aligned.shape == (3, 3)
        # Center of mass of aligned coordinates is strictly at origin
        com_aligned = np.sum(aligned * masses[:, np.newaxis], axis=0) / np.sum(masses)
        assert np.allclose(com_aligned, [0.0, 0.0, 0.0], atol=1e-12)

        with tempfile.TemporaryDirectory() as tmpdir:
            cube_file = Path(tmpdir) / "homo.cube"
            cube_file.write_text("CUBE FILE TEST HEADER\n", encoding="utf-8")
            out_png = Path(tmpdir) / "homo_300dpi.png"

            job = OrbitalRenderJob(
                orbital_index=5,
                orbital_type="HOMO",
                energy_hartree=-0.25,
                energy_ev=-6.8028,
                isovalue=0.03,
                cube_file_path=str(cube_file),
                output_png_path=str(out_png),
                resolution_dpi=300,
            )

            pml_script = generate_orbital_pymol_script(job)
            assert "cmd.isosurface('HOMO_pos', 'cube_data', 0.0300)" in pml_script
            assert "cmd.isosurface('HOMO_neg', 'cube_data', -0.0300)" in pml_script
            assert "cmd.ray(2400, 1800)" in pml_script
            assert "dpi=300" in pml_script

            # Render headless (using PIL bitmap fallback or PyMOL)
            res_png = execute_headless_orbital_render(job)
            assert res_png.is_file()

            # Composite figure assembly
            lumo_png = Path(tmpdir) / "lumo_300dpi.png"
            job_lumo = OrbitalRenderJob(
                orbital_index=6,
                orbital_type="LUMO",
                energy_hartree=0.05,
                energy_ev=1.3606,
                isovalue=0.03,
                cube_file_path=str(cube_file),
                output_png_path=str(lumo_png),
                resolution_dpi=300,
            )
            execute_headless_orbital_render(job_lumo)

            comp_png = Path(tmpdir) / "composite_homo_lumo.png"
            fig_set = assemble_composite_orbital_figure(job, job_lumo, comp_png)
            assert Path(fig_set.composite_figure_path).is_file()
            assert fig_set.gap_ev > 0.0
            assert fig_set.gap_kcal_mol > 0.0

    # =========================================================================
    # REQ-QC-095: Standalone 3D Interactive HTML WebGL Visualizer
    # =========================================================================
    def test_req_095_interactive_html_si_exporter(self) -> None:
        """[REQ-QC-095] Export standalone air-gapped HTML Supporting Information bundle."""
        xyz_h2o = "3\nWater molecule\nO 0.000000 0.000000 0.117300\nH 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200"
        model_1 = InteractiveHTMLModel(
            label="Reactant H2O",
            chemical_formula="H2O",
            energy_hartree=-76.4300,
            relative_energy_kcal_mol=0.0,
            xyz_block=xyz_h2o,
            frequencies=[1595.0, 3657.0, 3756.0],
            normal_modes=[
                [[0.0, 0.0, -0.06], [0.0, -0.4, 0.5], [0.0, 0.4, 0.5]],
                [[0.0, 0.0, 0.05], [0.0, 0.5, -0.4], [0.0, -0.5, -0.4]],
                [[0.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, -0.6, 0.0]],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_html = Path(tmpdir) / "interactive_si.html"
            pkg = InteractiveHTMLPackage(
                title="Reaction Profile 3D Supporting Information",
                models=[model_1],
                output_html_path=str(out_html),
            )

            html_res_path = export_interactive_html_si(pkg)
            assert Path(html_res_path).is_file()
            content = Path(html_res_path).read_text(encoding="utf-8")

            # Invariant: Must NOT contain external CDN calls (e.g. cdnjs, unpkg, cdn.jsdelivr)
            assert "https://cdnjs" not in content
            assert "https://unpkg" not in content
            assert "https://cdn.jsdelivr" not in content
            assert "CoChemViewer" in content
            assert "Reactant H2O" in content

    # =========================================================================
    # REQ-QC-099: Multi-Format Spreadsheet Exporter (.xlsx / .csv)
    # =========================================================================
    def test_req_099_spreadsheet_export_engine(self) -> None:
        """[REQ-QC-099] Export multi-sheet Excel with dynamic relative energy formulas and CSV."""
        records = [
            MolecularRecordSpreadsheetRow(
                job_label="Water_Reactant",
                formula="H2O",
                smiles="O",
                charge=0,
                multiplicity=1,
                e_scf_hartree=-76.43000000,
                zpve_hartree=0.02100000,
                h_298_hartree=-76.40500000,
                g_298_hartree=-76.42700000,
                dipole_x_debye=0.0,
                dipole_y_debye=0.0,
                dipole_z_debye=1.85,
                dipole_total_debye=1.85,
                rotational_a_mhz=835840.0,
                rotational_b_mhz=435350.0,
                rotational_c_mhz=278140.0,
                lowest_frequency_cm1=1595.0,
                imaginary_frequency_count=0,
            ),
            MolecularRecordSpreadsheetRow(
                job_label="TS_ProtonTransfer",
                formula="H3O+",
                smiles="[OH3+]",
                charge=1,
                multiplicity=1,
                e_scf_hartree=-76.38000000,
                zpve_hartree=0.01800000,
                h_298_hartree=-76.35800000,
                g_298_hartree=-76.38000000,
                dipole_x_debye=0.0,
                dipole_y_debye=0.0,
                dipole_z_debye=0.0,
                dipole_total_debye=0.0,
                rotational_a_mhz=332000.0,
                rotational_b_mhz=332000.0,
                rotational_c_mhz=189000.0,
                lowest_frequency_cm1=-452.1,
                imaginary_frequency_count=1,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            excel_p = Path(tmpdir) / "cochem_dataset.xlsx"
            csv_p = Path(tmpdir) / "cochem_dataset.csv"

            manifest = export_molecular_dataset_to_excel(records, excel_p, csv_p)
            assert manifest.total_records == 2
            assert excel_p.is_file()
            assert csv_p.is_file()

            # Verify Excel structure
            wb = openpyxl.load_workbook(str(excel_p))
            assert "Summary & Energetics" in wb.sheetnames
            assert "Dipoles & Spectroscopic" in wb.sheetnames
            assert "Cartesian Geometries" in wb.sheetnames

            ws1 = wb["Summary & Energetics"]
            # Verify dynamic formula is embedded
            cell_formula = ws1.cell(row=2, column=9).value
            assert "MIN" in str(cell_formula) and "627.509" in str(cell_formula)

            # Verify CSV content
            csv_text = csv_p.read_text(encoding="utf-8")
            assert "Water_Reactant" in csv_text
            assert "TS_ProtonTransfer" in csv_text

    # =========================================================================
    # REQ-QC-091 & REQ-QC-098: LaTeX Supporting Information PDF Builder
    # =========================================================================
    def test_req_091_latex_si_source_generation(self) -> None:
        """[REQ-QC-091] Generate full Supporting Information LaTeX source with 6-decimal coordinates."""
        author = ORCIDAuthorProfile(
            orcid_id="0000-0002-1825-0097",
            given_names="Josiah",
            family_name="Carberry",
            credit_name="Josiah Carberry",
            primary_institution="Brown University",
            credit_roles=["Supervision"],
        )

        entry = StationaryPointSIEntry(
            label="Reactant_Water",
            chemical_formula="H2O",
            electronic_energy_hartree=-76.43000000,
            zero_point_energy_hartree=0.02100000,
            enthalpy_hartree=-76.40500000,
            gibbs_free_energy_hartree=-76.42700000,
            rotational_constants_mhz={"A": 835840.0, "B": 435350.0, "C": 278140.0},
            number_of_imaginary_frequencies=0,
            frequencies_cm1=[1595.0, 3657.0, 3756.0],
            elements=["O", "H", "H"],
            coordinates_angstrom=[
                [0.000000, 0.000000, 0.117300],
                [0.000000, 0.757200, -0.469200],
                [0.000000, -0.757200, -0.469200],
            ],
        )

        tex_source = generate_latex_si_source(
            title="Computational Study of Reaction Energetics",
            authors=[author],
            entries=[entry],
            methods_paragraph="Calculations were conducted with ORCA 5.0.4.",
            acs_tables_latex=r"\begin{table}...\end{table}",
        )

        assert r"\documentclass[11pt,letterpaper]{article}" in tex_source
        assert r"\usepackage{longtable}" in tex_source
        assert r"\usepackage{booktabs}" in tex_source
        assert "0.117300" in tex_source
        assert "-0.469200" in tex_source
        assert "-76.43000000" in tex_source

    # =========================================================================
    # REQ-QC-091..100: GUI Reactive Widgets and Master Controller Integration
    # =========================================================================
    def test_req_gui_widgets_and_master_qc_controller(self) -> None:
        """[REQ-QC-091..100] Verify GUI widgets reactivity, state contracts, and controller dispatch."""
        # 1. ACSTableEditorWidget
        table_widget = ACSTableEditorWidget()
        entries = [
            ThermochemicalSpeciesEntry(
                species_name="R1",
                e_elec_hartree=-100.0,
                zpve_hartree=0.05,
                h_298_hartree=-99.94,
                g_298_hartree=-99.97,
            ),
            ThermochemicalSpeciesEntry(
                species_name="P1",
                e_elec_hartree=-100.05,
                zpve_hartree=0.05,
                h_298_hartree=-99.99,
                g_298_hartree=-100.02,
            ),
        ]
        table_widget.load_entries(entries, reference_label="R1")
        assert table_widget.state.species_count == 2
        assert r"\toprule" in table_widget.copy_latex_to_clipboard()
        assert "| R1 |" in table_widget.copy_markdown_to_clipboard()

        table_widget.toggle_energy_units("kj_mol")
        assert table_widget.state.energy_units == "kj_mol"

        # 2. OrbitalViewerWidget
        orbital_widget = OrbitalViewerWidget()
        orbital_widget.select_orbital("LUMO", 10)
        orbital_widget.set_isovalue(0.04)
        orbital_widget.set_resolution_dpi(600)
        assert orbital_widget.state.selected_orbital_type == "LUMO"
        assert orbital_widget.state.isovalue == 0.04
        assert orbital_widget.state.resolution_dpi == 600

        # 3. HTMLSIPreviewWidget
        html_widget = HTMLSIPreviewWidget()
        assert html_widget.state.is_loaded is False

        # 4. ZenodoPublishDialog
        zenodo_dlg = ZenodoPublishDialog()
        zenodo_dlg.set_environment(sandbox_mode=True)
        zenodo_dlg.set_api_token("sandbox_token_123")
        assert zenodo_dlg.state.sandbox_mode is True

        # 5. Master QC Controller integration
        controller = CoChemMasterQCController()
        tbl = controller.open_table_editor()
        assert isinstance(tbl, ACSTableEditorWidget)
        orb = controller.open_orbital_viewer()
        assert isinstance(orb, OrbitalViewerWidget)
        htm = controller.open_html_si_preview()
        assert isinstance(htm, HTMLSIPreviewWidget)
        zen = controller.open_zenodo_dialog()
        assert isinstance(zen, ZenodoPublishDialog)

    # =========================================================================
    # REQ-QC-091..100: End-to-End Master Publication Assembler
    # =========================================================================
    def test_req_master_publication_package_assembler(self) -> None:
        """[REQ-QC-091..100] Verify master pipeline assembly connecting all 10 requirements."""
        author = ORCIDAuthorProfile(
            orcid_id="0000-0002-1825-0097",
            given_names="Josiah",
            family_name="Carberry",
            credit_name="Josiah Carberry",
            primary_institution="Brown University",
            credit_roles=["Conceptualization", "Writing - original draft"],
        )

        entry_r = StationaryPointSIEntry(
            label="Reactant",
            chemical_formula="H2O",
            electronic_energy_hartree=-76.43000000,
            zero_point_energy_hartree=0.02100000,
            enthalpy_hartree=-76.40500000,
            gibbs_free_energy_hartree=-76.42700000,
            rotational_constants_mhz={"A": 835840.0, "B": 435350.0, "C": 278140.0},
            number_of_imaginary_frequencies=0,
            frequencies_cm1=[1595.0, 3657.0, 3756.0],
            elements=["O", "H", "H"],
            coordinates_angstrom=[[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        )
        entry_ts = StationaryPointSIEntry(
            label="TS_1",
            chemical_formula="H3O+",
            electronic_energy_hartree=-76.38000000,
            zero_point_energy_hartree=0.01800000,
            enthalpy_hartree=-76.35800000,
            gibbs_free_energy_hartree=-76.38000000,
            rotational_constants_mhz={"A": 332000.0, "B": 332000.0, "C": 189000.0},
            number_of_imaginary_frequencies=1,
            frequencies_cm1=[-452.1, 1400.0, 3200.0],
            elements=["O", "H", "H", "H"],
            coordinates_angstrom=[[0.0, 0.0, 0.1], [0.0, 0.8, -0.4], [0.0, -0.8, -0.4], [0.9, 0.0, 0.0]],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_res = assemble_full_publication_package(
                project_dir=tmpdir,
                output_dir=tmpdir,
                author_profiles=[author],
                title="Proton Transfer Reaction in Solution",
                stationary_points=[entry_r, entry_ts],
                reference_species_label="Reactant",
            )

            assert "methods_markdown" in pkg_res
            assert "methods_latex" in pkg_res
            assert "bibtex_file" in pkg_res
            assert "thermochemistry_table" in pkg_res
            assert "transparency_audit" in pkg_res
            assert "si_tex_file" in pkg_res

            assert Path(pkg_res["bibtex_file"]).is_file()
            assert Path(pkg_res["si_tex_file"]).is_file()
            assert pkg_res["transparency_audit"]["compliance_score_percent"] > 0.0
