"""
`Scoring` holds classes for calculating DOPE and Ramachandran scores.
"""
class RaiseErrorOnInit:
    module = 'unknown module is creating an ImportError'
    def __init__(self,*args, **kwargs):
        raise ImportError(f'{self.module}. Therefore {self.__class__.__name__} can not be used')
try:
    from .dope_score import Parallel_DOPE_Score, DOPE_Score
except ImportError as e:
    import warnings
    warnings.warn(f"{e}. Modeller is probably not installed.")
    class DOPE_Score(RaiseErrorOnInit):
        module = e
    class Parallel_DOPE_Score(RaiseErrorOnInit):
        module = e

try:
    from .ramachandran_score import Parallel_Ramachandran_Score, Ramachandran_Score
except Exception as e:
    class Parallel_Ramachandran_Score(RaiseErrorOnInit):
        module = e
    class Ramachandran_Score(RaiseErrorOnInit):
        module = e
    import warnings
    warnings.warn(f"{e}. Will not be able to calculate Ramachandran score.")