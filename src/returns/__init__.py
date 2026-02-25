"""
Return models: Student-t and covariance handling.

This module implements:
- Multivariate Student-t distributions for asset returns
- LKJ-prior covariance matrix handling
- Conditional distributions and transformations
"""

from src.returns.student_t import StudentTReturnModel, create_symmetric_return_model
from src.returns.covariance import (
    CovarianceModel,
    create_identity_covariance,
    create_compound_symmetric_covariance,
)

__all__ = [
    "StudentTReturnModel",
    "create_symmetric_return_model",
    "CovarianceModel",
    "create_identity_covariance",
    "create_compound_symmetric_covariance",
]
