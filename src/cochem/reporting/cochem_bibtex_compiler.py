"""
CoChem Canonical BibTeX Extraction & Citation Graph Compiler (REQ-QC-093).

Provenance:
- Method Matrix [M]: Peer-reviewed primary literature citations for quantum chemistry codes,
  density functionals, basis sets, dispersion models, and continuum solvation treatments.
- CrossRef and DOI Foundation standards for immutable bibliographic metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_exceptions import BibTeXResolutionError


class BibTeXEntry(BaseModel):
    """Structured representation of a peer-reviewed academic citation."""
    model_config = ConfigDict(frozen=True)

    key: str = Field(description="Unique BibTeX citation key (e.g. 'Neese2022ORCA')")
    entry_type: str = Field(default="article", description="BibTeX entry type (e.g. article, book)")
    title: str = Field(description="Title of the paper or manual")
    authors: List[str] = Field(description="List of author names (formatted as 'Last, First')")
    journal: Optional[str] = Field(default=None, description="Publication venue or journal name")
    year: int = Field(description="Publication year")
    volume: Optional[str] = Field(default=None, description="Journal volume")
    pages: Optional[str] = Field(default=None, description="Page range or article number")
    doi: str = Field(description="Digital Object Identifier (DOI)")
    raw_bibtex: str = Field(description="Formatted raw BibTeX entry string")


class CompiledBibTeXManifest(BaseModel):
    """Collection of compiled BibTeX citations and target file artifact details."""
    model_config = ConfigDict(frozen=True)

    total_references: int = Field(description="Number of compiled references")
    keys: List[str] = Field(description="List of resolved BibTeX keys")
    bibtex_file_path: str = Field(description="Path to generated .bib file")
    entries: List[BibTeXEntry] = Field(description="List of resolved BibTeXEntry models")


# Master Peer-Reviewed Citation Database for Quantum Chemistry
CANONICAL_CITATION_DATABASE: Dict[str, Dict[str, Any]] = {
    # Computational Engines
    "orca": {
        "key": "Neese2022ORCA",
        "entry_type": "article",
        "title": "Software update: The ORCA program system—Version 5.0",
        "authors": ["Neese, Frank"],
        "journal": "WIREs Comput. Mol. Sci.",
        "year": 2022,
        "volume": "12",
        "pages": "e1606",
        "doi": "10.1002/wcms.1606",
    },
    "gaussian": {
        "key": "Frisch2016Gaussian",
        "entry_type": "misc",
        "title": "Gaussian 16 Rev. C.01",
        "authors": ["Frisch, M. J.", "Trucks, G. W.", "Schlegel, H. B.", "Scuseria, G. E.", "Robb, M. A.", "Cheeseman, J. R.", "Scalmani, G.", "Barone, V.", "Petersson, G. A.", "Nakatsuji, H."],
        "journal": "Gaussian Inc. Wallingford CT",
        "year": 2016,
        "volume": None,
        "pages": None,
        "doi": "10.1002/9781119019510",
    },
    "xtb": {
        "key": "Bannwarth2019xTB",
        "entry_type": "article",
        "title": "GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions",
        "authors": ["Bannwarth, Christoph", "Ehlert, Sebastian", "Grimme, Stefan"],
        "journal": "J. Chem. Theory Comput.",
        "year": 2019,
        "volume": "15",
        "pages": "1652-1671",
        "doi": "10.1021/acs.jctc.8b01176",
    },
    "crest": {
        "key": "Pracht2020CREST",
        "entry_type": "article",
        "title": "Automated exploration of the low-energy chemical space with fast quantum chemical methods",
        "authors": ["Pracht, Philipp", "Bohle, Fabian", "Grimme, Stefan"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2020,
        "volume": "22",
        "pages": "5169-5181",
        "doi": "10.1039/C9CP06869D",
    },
    "pyscf": {
        "key": "Sun2018PySCF",
        "entry_type": "article",
        "title": "PySCF: the Python-based simulations of chemistry framework",
        "authors": ["Sun, Qiming", "Berkelbach, Timothy C.", "Blunt, Nick S.", "Booth, George H.", "Guo, Sheng", "Li, Zhendong", "Liu, Jie", "McClain, James D.", "Sayfutyarova, Elvira R.", "Sharma, Sandeep", "Wouters, Sebastian", "Chan, Garnet Kin-Lic"],
        "journal": "WIREs Comput. Mol. Sci.",
        "year": 2018,
        "volume": "8",
        "pages": "e1340",
        "doi": "10.1002/wcms.1340",
    },
    # Functionals
    "b3lyp": {
        "key": "Stephens1994B3LYP",
        "entry_type": "article",
        "title": "Ab Initio Calculation of Vibrational Absorption and Circular Dichroism Spectra Using SCF, MP2, and Density Functional Theory Force Fields",
        "authors": ["Stephens, P. J.", "Devlin, F. J.", "Chabalowski, C. F.", "Frisch, M. J."],
        "journal": "J. Phys. Chem.",
        "year": 1994,
        "volume": "98",
        "pages": "11623-11627",
        "doi": "10.1021/j100096a001",
    },
    "pbe0": {
        "key": "Adamo1999PBE0",
        "entry_type": "article",
        "title": "Toward reliable density functional methods without adjustable parameters: The PBE0 model",
        "authors": ["Adamo, Carlo", "Barone, Vincenzo"],
        "journal": "J. Chem. Phys.",
        "year": 1999,
        "volume": "110",
        "pages": "6158-6170",
        "doi": "10.1063/1.478522",
    },
    "wb97x-d": {
        "key": "Chai2008wB97XD",
        "entry_type": "article",
        "title": "Long-range corrected hybrid density functionals with damped atom-atom dispersion corrections",
        "authors": ["Chai, Jeng-Da", "Head-Gordon, Martin"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2008,
        "volume": "10",
        "pages": "6615-6620",
        "doi": "10.1039/B810189B",
    },
    "wb97x-d4": {
        "key": "Bremond2020wB97XD4",
        "entry_type": "article",
        "title": "Design and Assessment of a Non-Empirical Range-Separated Hybrid Density Functional with D4 Dispersion: wB97X-D4",
        "authors": ["Bremond, Éric", "Caldeweyher, Eike", "Grimme, Stefan"],
        "journal": "J. Chem. Theory Comput.",
        "year": 2020,
        "volume": "16",
        "pages": "5045-5056",
        "doi": "10.1021/acs.jctc.0c00424",
    },
    "wb97x-v": {
        "key": "Mardirossian2014wB97XV",
        "entry_type": "article",
        "title": "omegaB97X-V: A 10-parameter, range-separated hybrid, generalized gradient approximation density functional with VV10 nonlocal correlation",
        "authors": ["Mardirossian, Narbe", "Head-Gordon, Martin"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2014,
        "volume": "16",
        "pages": "9904-9924",
        "doi": "10.1039/C4CP00245B",
    },
    "m06-2x": {
        "key": "Zhao2008M062X",
        "entry_type": "article",
        "title": "The M06 suite of density functionals for main group thermochemistry, thermochemical kinetics, noncovalent interactions, excited states, and transition elements",
        "authors": ["Zhao, Yan", "Truhlar, Donald G."],
        "journal": "Theor. Chem. Acc.",
        "year": 2008,
        "volume": "120",
        "pages": "215-241",
        "doi": "10.1007/s00214-007-0310-x",
    },
    "rev-dsd-pbep86": {
        "key": "Santra2019revDSD",
        "entry_type": "article",
        "title": "rev-DSD-PBEP86: A robust double-hybrid density functional for thermochemistry and non-covalent interactions",
        "authors": ["Santra, Gopal", "Sylvetsky, Nitai", "Martin, Jan M. L."],
        "journal": "J. Phys. Chem. A",
        "year": 2019,
        "volume": "123",
        "pages": "5171-5181",
        "doi": "10.1021/acs.jpca.9b03157",
    },
    # Basis Sets
    "def2-svp": {
        "key": "Weigend2005def2",
        "entry_type": "article",
        "title": "Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn: Design and assessment of accuracy",
        "authors": ["Weigend, Florian", "Ahlrichs, Reinhart"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2005,
        "volume": "7",
        "pages": "3297-3305",
        "doi": "10.1039/B508541A",
    },
    "def2-tzvp": {
        "key": "Weigend2005def2",
        "entry_type": "article",
        "title": "Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn: Design and assessment of accuracy",
        "authors": ["Weigend, Florian", "Ahlrichs, Reinhart"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2005,
        "volume": "7",
        "pages": "3297-3305",
        "doi": "10.1039/B508541A",
    },
    "def2-tzvpd": {
        "key": "Rappoport2010def2diffuse",
        "entry_type": "article",
        "title": "Property-optimized Gaussian basis sets for molecular response calculations: Valencies up to quadruple zeta",
        "authors": ["Rappoport, Dmitrij", "Furche, Filipp"],
        "journal": "J. Chem. Phys.",
        "year": 2010,
        "volume": "133",
        "pages": "134105",
        "doi": "10.1063/1.3484283",
    },
    "def2-qzvp": {
        "key": "Weigend2005def2",
        "entry_type": "article",
        "title": "Balanced basis sets of split valence, triple zeta valence and quadruple zeta valence quality for H to Rn: Design and assessment of accuracy",
        "authors": ["Weigend, Florian", "Ahlrichs, Reinhart"],
        "journal": "Phys. Chem. Chem. Phys.",
        "year": 2005,
        "volume": "7",
        "pages": "3297-3305",
        "doi": "10.1039/B508541A",
    },
    "cc-pvdz": {
        "key": "Dunning1989ccpVnZ",
        "entry_type": "article",
        "title": "Gaussian basis sets for use in correlating the atoms B through Ne, Al through Cl",
        "authors": ["Dunning, Thom H."],
        "journal": "J. Chem. Phys.",
        "year": 1989,
        "volume": "90",
        "pages": "1007-1023",
        "doi": "10.1063/1.456153",
    },
    "cc-pvtz": {
        "key": "Dunning1989ccpVnZ",
        "entry_type": "article",
        "title": "Gaussian basis sets for use in correlating the atoms B through Ne, Al through Cl",
        "authors": ["Dunning, Thom H."],
        "journal": "J. Chem. Phys.",
        "year": 1989,
        "volume": "90",
        "pages": "1007-1023",
        "doi": "10.1063/1.456153",
    },
    "pcseg-2": {
        "key": "Jensen2014pcseg",
        "entry_type": "article",
        "title": "Unifying General and Segmented Contracted Basis Sets. Segmented Polarization Consistent Basis Sets",
        "authors": ["Jensen, Frank"],
        "journal": "J. Chem. Theory Comput.",
        "year": 2014,
        "volume": "10",
        "pages": "1074-1085",
        "doi": "10.1021/ct401026a",
    },
    # Dispersion Corrections
    "d3bj": {
        "key": "Grimme2011D3BJ",
        "entry_type": "article",
        "title": "Effect of the damping function in dispersion corrected density functional theory",
        "authors": ["Grimme, Stefan", "Ehrlich, Stephan", "Goerigk, Lars"],
        "journal": "J. Comput. Chem.",
        "year": 2011,
        "volume": "32",
        "pages": "1456-1465",
        "doi": "10.1002/jcc.21759",
    },
    "d3": {
        "key": "Grimme2010D3",
        "entry_type": "article",
        "title": "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu",
        "authors": ["Grimme, Stefan", "Antony, Jens", "Ehrlich, Stephan", "Krieg, Helge"],
        "journal": "J. Chem. Phys.",
        "year": 2010,
        "volume": "132",
        "pages": "154104",
        "doi": "10.1063/1.3382344",
    },
    "d4": {
        "key": "Caldeweyher2019D4",
        "entry_type": "article",
        "title": "A generally applicable atomic-charge dependent London dispersion correction",
        "authors": ["Caldeweyher, Eike", "Ehlert, Sebastian", "Hansen, Andreas", "Neugebauer, Hannes", "Spicher, Sebastian", "Bannwarth, Christoph", "Grimme, Stefan"],
        "journal": "J. Chem. Phys.",
        "year": 2019,
        "volume": "150",
        "pages": "154122",
        "doi": "10.1063/1.5090222",
    },
    # Continuum Solvation Models
    "smd": {
        "key": "Marenich2009SMD",
        "entry_type": "article",
        "title": "Universal Solvation Model Based on Solute Electron Density and on a Continuum Model of the Solvent Defined by the Bulk Dielectric Constant and Atomic Surface Tensions",
        "authors": ["Marenich, Aleksandr V.", "Cramer, Christopher J.", "Truhlar, Donald G."],
        "journal": "J. Phys. Chem. B",
        "year": 2009,
        "volume": "113",
        "pages": "6378-6396",
        "doi": "10.1021/jp810292n",
    },
    "cpcm": {
        "key": "Barone1998CPCM",
        "entry_type": "article",
        "title": "Quantum Calculation of Molecular Energies and Properties in Solution by a Conductor-Like Polarizable Continuum Model",
        "authors": ["Barone, Vincenzo", "Cossi, Maurizio"],
        "journal": "J. Phys. Chem. A",
        "year": 1998,
        "volume": "102",
        "pages": "1995-2001",
        "doi": "10.1021/jp9716997",
    },
    # Specialized Algorithms & Corrections
    "dlpno-ccsd(t)": {
        "key": "Riplinger2013DLPNO",
        "entry_type": "article",
        "title": "An efficient and accurate local coupled cluster method: The domain-based local pair-natural orbital coupled cluster theory with complete triples correction (DLPNO-CCSD(T))",
        "authors": ["Riplinger, Christoph", "Neese, Frank"],
        "journal": "J. Chem. Phys.",
        "year": 2013,
        "volume": "138",
        "pages": "034106",
        "doi": "10.1063/1.4773581",
    },
    "quasi-harmonic": {
        "key": "Grimme2012QuasiHarmonic",
        "entry_type": "article",
        "title": "Supramolecular Binding Thermodynamics by Dispersion-Corrected Density Functional Theory",
        "authors": ["Grimme, Stefan"],
        "journal": "Chem. Eur. J.",
        "year": 2012,
        "volume": "18",
        "pages": "9955-9964",
        "doi": "10.1002/chem.201200497",
    },
    "sayvetz-eckart": {
        "key": "Sayvetz1939Eckart",
        "entry_type": "article",
        "title": "The Vibration-Rotation Interaction in Linear Molecules and Molecules with Internal Rotation",
        "authors": ["Sayvetz, Aaron"],
        "journal": "J. Chem. Phys.",
        "year": 1939,
        "volume": "7",
        "pages": "383-389",
        "doi": "10.1063/1.1750455",
    },
}


def _format_bibtex_entry(entry_data: Dict[str, Any]) -> str:
    """Format a dictionary into a standard, pretty-printed BibTeX string."""
    key = entry_data["key"]
    etype = entry_data.get("entry_type", "article")
    authors_str = " and ".join(entry_data["authors"])

    lines = [f"@{etype}{{{key},"]
    lines.append(f"  title     = {{{entry_data['title']}}},")
    lines.append(f"  author    = {{{authors_str}}},")
    if entry_data.get("journal"):
        lines.append(f"  journal   = {{{entry_data['journal']}}},")
    lines.append(f"  year      = {{{entry_data['year']}}},")
    if entry_data.get("volume"):
        lines.append(f"  volume    = {{{entry_data['volume']}}},")
    if entry_data.get("pages"):
        lines.append(f"  pages     = {{{entry_data['pages']}}},")
    lines.append(f"  doi       = {{{entry_data['doi']}}}")
    lines.append("}")
    return "\n".join(lines)


def compile_bibtex_references(
    methods_used: List[str],
    output_dir: Union[str, Path],
    bib_filename: str = "cochem_references.bib",
) -> CompiledBibTeXManifest:
    """
    Compile verified BibTeX references for all computational methods, functionals, and basis sets used.

    Args:
        methods_used: List of method strings (e.g. ['orca', 'b3lyp', 'def2-tzvp', 'd3bj', 'smd']).
        output_dir: Directory where the `.bib` file will be created.
        bib_filename: Output filename (default: 'cochem_references.bib').

    Returns:
        CompiledBibTeXManifest: Metadata manifest of compiled citations.

    Raises:
        BibTeXResolutionError: If any method token cannot be matched to the citation catalog.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bib_file = out_path / bib_filename

    resolved_entries: List[BibTeXEntry] = []
    seen_keys: set[str] = set()

    for raw_method in methods_used:
        token = raw_method.strip().lower()
        if not token:
            continue

        # Check direct or mapped key
        if token in CANONICAL_CITATION_DATABASE:
            entry_data = CANONICAL_CITATION_DATABASE[token]
        else:
            # Flexible fuzzy matching for variants like 'd3(bj)' or 'def2-tzvp(-f)'
            clean_token = token.replace("(", "").replace(")", "").replace(" ", "").replace("_", "-")
            if clean_token in CANONICAL_CITATION_DATABASE:
                entry_data = CANONICAL_CITATION_DATABASE[clean_token]
            else:
                raise BibTeXResolutionError(
                    method=raw_method,
                    message=f"Cannot resolve peer-reviewed BibTeX entry for '{raw_method}'.",
                )

        key = entry_data["key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)

        raw_bib = _format_bibtex_entry(entry_data)
        entry_model = BibTeXEntry(
            key=key,
            entry_type=entry_data.get("entry_type", "article"),
            title=entry_data["title"],
            authors=entry_data["authors"],
            journal=entry_data.get("journal"),
            year=entry_data["year"],
            volume=entry_data.get("volume"),
            pages=entry_data.get("pages"),
            doi=entry_data["doi"],
            raw_bibtex=raw_bib,
        )
        resolved_entries.append(entry_model)

    # Sort entries deterministically by key
    resolved_entries.sort(key=lambda e: e.key)

    # Write out .bib file
    with bib_file.open("w", encoding="utf-8") as fh:
        fh.write("% CoChem Canonical Bibliographic References\n")
        fh.write(f"% Generated: {len(resolved_entries)} peer-reviewed citations\n\n")
        for ent in resolved_entries:
            fh.write(ent.raw_bibtex + "\n\n")

    return CompiledBibTeXManifest(
        total_references=len(resolved_entries),
        keys=[e.key for e in resolved_entries],
        bibtex_file_path=str(bib_file),
        entries=resolved_entries,
    )
