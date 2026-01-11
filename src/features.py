# src/mikolajkuna/features.py

"""
Definicja cech wejściowych używanych w modelach ML
i cech interesujących do analizy kontrfaktycznej / SHAP.
"""

FEATURES = [
    "age",
    "gender",
    "education_level",
    "job_level",
    "experience_years",
    "distance_from_home",
    "absence",
    "child"
]

FEATURES_OF_INTEREST = [
    "gender",
    "child",
    "education_level",
    "job_level"
]
