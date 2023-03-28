"""
Scoring holds the classes for calculating DOPE, Ramachandran and OPENMM energy
"""

from .dope_score import Parallel_DOPE_Score, DOPE_Score

try:
    from .ramachandran_score import Parallel_Ramachandran_Score, Ramachandran_Score
except Exception as e:
    import warnings
    warnings.warn(f"{e}. Will not be able to calculate Ramachandran score.")