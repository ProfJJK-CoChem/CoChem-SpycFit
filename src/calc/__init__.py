"""# zero-stub anti-spoofing engine
Calc Package Forwarder.
"""
from cochem_base.calc.cochem_method_auditor import audit_calculation_parameters, generate_methodology_audit_report
from cochem_base.calc.cochem_engine import dispatch_calculation, validate_qc_results, MethodologyAuditError
