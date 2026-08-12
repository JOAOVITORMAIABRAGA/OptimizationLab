from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from domain.expressions import StructuredExpression
from domain.problem import ConstraintSpec, DomainSpec, OptimizationProblem, ObjectiveSpec, VariableSpec
from domain.objectives import ObjectiveKind
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType


SUPPORTED_UNARY_OPERATIONS = {"neg"}
SUPPORTED_BINARY_OPERATIONS = {"add", "sub", "mul", "div", "pow", "mod"}
SUPPORTED_FUNCTIONS = {"sum", "abs", "min", "max"}


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return not self.errors


class ValidationEngine:
    def validate(self, problem: OptimizationProblem) -> ValidationReport:
        report = ValidationReport()

        if not problem.name:
            report.errors.append("Problem name is required.")

        if not problem.objective:
            report.errors.append("Problem objective is required.")
        else:
            self._validate_objective(problem.objective, problem.variables, report)

        if not problem.variables:
            report.errors.append("At least one variable is required.")
        else:
            for variable in problem.variables:
                self._validate_variable(variable, report)

        for constraint in problem.constraints:
            self._validate_constraint(constraint, problem.variables, report)

        self._validate_problem_family(problem, report)
        self._validate_representation(problem, report)
        self._validate_consistency(problem, report)

        return report

    def _validate_objective(self, objective: ObjectiveSpec, variables: List[VariableSpec], report: ValidationReport) -> None:
        if not objective.kind:
            report.errors.append("Objective kind is required.")
        if objective.kind == ObjectiveKind.MULTI:
            if objective.sense is not None:
                report.warnings.append("Objective-level sense is ignored for multiobjective models; each objective declares its own sense.")
            if len(objective.objectives) < 2:
                report.errors.append("Multiobjective problems require at least two independent objectives.")
            for component in objective.objectives:
                self._validate_expression(component.expression, variables, report, context=f"objective:{component.id}")
            return
        if not objective.sense:
            report.errors.append("Objective sense is required.")
        if objective.expression is None:
            report.errors.append("Objective expression is required.")
        else:
            self._validate_expression(objective.expression, variables, report, context="objective")

    def _validate_variable(self, variable: VariableSpec, report: ValidationReport) -> None:
        if not variable.name:
            report.errors.append("Each variable must have a name.")

        if variable.variable_type is None:
            report.errors.append(f"Variable '{variable.name}' must declare a variable type.")

        if variable.domain is None:
            report.errors.append(f"Variable '{variable.name}' must declare a domain.")
        else:
            self._validate_domain(variable, report)

        if variable.variable_type == VariableType.CONTINUOUS and variable.domain is not None:
            if variable.domain.kind not in {"continuous"}:
                report.errors.append(f"Domain for continuous variable '{variable.name}' must be continuous.")
        if variable.variable_type == VariableType.BINARY and variable.domain is not None:
            if variable.domain.kind != "binary" or variable.domain.values != [0, 1]:
                report.errors.append(f"Binary variable '{variable.name}' must use a binary domain.")
        if variable.variable_type == VariableType.INTEGER and variable.domain is not None:
            if variable.domain.kind not in {"integer"}:
                report.errors.append(f"Integer variable '{variable.name}' must use an integer domain.")

    def _validate_domain(self, variable: VariableSpec, report: ValidationReport) -> None:
        domain = variable.domain
        if domain.kind == "continuous":
            if domain.lower is None or domain.upper is None:
                report.errors.append(f"Continuous domain for variable '{variable.name}' must define lower and upper bounds.")
            if domain.lower is not None and domain.upper is not None and domain.lower > domain.upper:
                report.errors.append(f"Bounds for variable '{variable.name}' are inconsistent.")
        elif domain.kind == "integer":
            if domain.lower is None or domain.upper is None:
                report.errors.append(f"Integer domain for variable '{variable.name}' must define bounds.")
            if domain.lower is not None and domain.upper is not None and domain.lower > domain.upper:
                report.errors.append(f"Bounds for variable '{variable.name}' are inconsistent.")
        elif domain.kind == "binary":
            if domain.values is None or domain.values != [0, 1]:
                report.errors.append(f"Binary domain for variable '{variable.name}' must be [0, 1].")
        elif domain.kind == "permutation":
            if not domain.values and not domain.elements:
                report.errors.append(f"Permutation domain for variable '{variable.name}' must define values or elements.")
        elif domain.kind == "categorical":
            if not domain.categories:
                report.errors.append(f"Categorical domain for variable '{variable.name}' must define categories.")
        elif domain.kind == "discrete":
            if not domain.values and not domain.elements:
                report.errors.append(f"Discrete domain for variable '{variable.name}' must define values or elements.")

    def _validate_constraint(self, constraint: ConstraintSpec, variables: List[VariableSpec], report: ValidationReport) -> None:
        if not constraint.id:
            report.errors.append("Each constraint must have an id.")
        if not constraint.name:
            report.errors.append("Each constraint must have a name.")
        if constraint.kind not in {"hard", "soft"}:
            report.errors.append(f"Constraint '{constraint.name}' has an invalid kind.")
        if constraint.relation not in {"eq", "le", "ge", "bound", "custom"}:
            report.errors.append(f"Constraint '{constraint.name}' has an invalid relation.")

        if constraint.relation in {"eq", "le", "ge", "custom"}:
            if constraint.expression is None:
                report.errors.append(f"Constraint '{constraint.name}' must define an expression.")
            else:
                self._validate_expression(constraint.expression, variables, report, context=f"constraint:{constraint.name}")
        elif constraint.relation == "bound":
            if constraint.lower_bound is None and constraint.upper_bound is None:
                report.errors.append(f"Constraint '{constraint.name}' must define bounds.")

    def _validate_problem_family(self, problem: OptimizationProblem, report: ValidationReport) -> None:
        if problem.problem_family is None:
            report.errors.append("Problem family is required.")

    def _validate_representation(self, problem: OptimizationProblem, report: ValidationReport) -> None:
        if problem.solution_representation is None:
            report.errors.append("Solution representation is required.")
            return

        if problem.solution_representation.kind == SolutionRepresentationKind.PERMUTATION:
            for variable in problem.variables:
                if variable.variable_type not in {VariableType.DISCRETE, VariableType.INTEGER}:
                    report.errors.append("Permutation representation requires discrete or integer variables.")
        elif problem.solution_representation.kind == SolutionRepresentationKind.VECTOR:
            for variable in problem.variables:
                if variable.variable_type not in {VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.BINARY}:
                    report.errors.append("Vector representation is only valid for continuous, integer or binary variables.")
        elif problem.solution_representation.kind == SolutionRepresentationKind.GRAPH:
            for variable in problem.variables:
                if variable.variable_type != VariableType.DISCRETE:
                    report.errors.append("Graph representation requires discrete variables.")

    def _validate_consistency(self, problem: OptimizationProblem, report: ValidationReport) -> None:
        variable_names = {variable.name for variable in problem.variables}
        if problem.objective is not None and problem.objective.expression is not None:
            self._validate_expression_references(problem.objective.expression, variable_names, report, context="objective")
        for constraint in problem.constraints:
            if constraint.expression is not None:
                self._validate_expression_references(constraint.expression, variable_names, report, context=f"constraint:{constraint.name}")

        if problem.problem_family == ProblemFamily.ROUTING:
            if problem.solution_representation is None or problem.solution_representation.kind != SolutionRepresentationKind.PERMUTATION:
                report.errors.append("Routing problems require a permutation-based representation.")

        if problem.problem_family == ProblemFamily.FEATURE_SELECTION:
            if any(variable.variable_type != VariableType.BINARY for variable in problem.variables):
                report.errors.append("Feature selection problems require binary variables.")

    def _validate_expression(self, expression: StructuredExpression, variables: List[VariableSpec], report: ValidationReport, context: str) -> None:
        variable_names = {variable.name for variable in variables}
        self._validate_expression_references(expression, variable_names, report, context=context)
        self._validate_expression_structure(expression, report, context=context)

    def _validate_expression_references(self, expression: StructuredExpression, variable_names: Set[str], report: ValidationReport, context: str) -> None:
        if expression.kind == "variable":
            if expression.name not in variable_names:
                report.errors.append(f"Unknown variable '{expression.name}' referenced in {context}.")
        elif expression.kind in {"binary", "unary", "function"}:
            for arg in expression.args:
                self._validate_expression_references(arg, variable_names, report, context=context)

    def _validate_expression_structure(self, expression: StructuredExpression, report: ValidationReport, context: str) -> None:
        if expression.kind == "literal":
            return
        if expression.kind == "variable":
            return
        if expression.kind == "unary":
            if expression.op not in SUPPORTED_UNARY_OPERATIONS:
                report.errors.append(f"Unsupported unary operation '{expression.op}' in {context}.")
                return
            if len(expression.args) != 1:
                report.errors.append(f"Unary expression in {context} must have exactly one argument.")
                return
            self._validate_expression_structure(expression.args[0], report, context=context)
            return
        if expression.kind == "binary":
            if expression.op not in SUPPORTED_BINARY_OPERATIONS:
                report.errors.append(f"Unsupported binary operation '{expression.op}' in {context}.")
                return
            if len(expression.args) != 2:
                report.errors.append(f"Binary expression in {context} must have exactly two arguments.")
                return
            self._validate_expression_structure(expression.args[0], report, context=context)
            self._validate_expression_structure(expression.args[1], report, context=context)
            self._validate_types(expression, report, context=context)
            return
        if expression.kind == "function":
            if expression.name not in SUPPORTED_FUNCTIONS:
                report.errors.append(f"Unsupported function '{expression.name}' in {context}.")
                return
            if not expression.args:
                report.errors.append(f"Function '{expression.name}' in {context} must have at least one argument.")
                return
            for arg in expression.args:
                self._validate_expression_structure(arg, report, context=context)
            return
        report.errors.append(f"Invalid expression structure in {context}.")

    def _validate_types(self, expression: StructuredExpression, report: ValidationReport, context: str) -> None:
        if not expression.args:
            return
        left = expression.args[0]
        right = expression.args[1]
        if left.kind == "literal" and right.kind == "literal":
            if not isinstance(left.value, (int, float)) or not isinstance(right.value, (int, float)):
                report.errors.append(f"Incompatible literal types in {context}.")
        elif left.kind == "literal" and right.kind != "literal":
            if not isinstance(left.value, (int, float)):
                report.errors.append(f"Incompatible operand types in {context}.")
        elif right.kind == "literal" and left.kind != "literal":
            if not isinstance(right.value, (int, float)):
                report.errors.append(f"Incompatible operand types in {context}.")
        elif left.kind == "variable" and right.kind == "variable":
            return
        elif left.kind == "function" or right.kind == "function":
            return
        else:
            return
