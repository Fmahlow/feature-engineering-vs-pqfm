"""
Wrappers em torno do pqfmlib (https://github.com/rscarmo/pqfmlib) para gerar
as caracteristicas quanticas de referencia -- o "alvo a bater" de cada base.

Ver PLANO_ESTUDO_FEATURE_ENGINEERING_VS_PQFM.md, Secao 5.

Decisao de design (documentada no plano): usar sempre simulacao por vetor de
estado (AerSimulator / statevector), nunca hardware IBM real, para remover
efeitos de ruido/deriva de calibracao como fator de confusao.

Para bases de alta dimensao (I-SPY1, Molecular Toxicity, 100+ qubits
simulados), preferir a simulacao por blocos/MPS ja implementada no pqfmlib
para o Heisenberg map, ou a formula fechada (dequantizacao, ver
classical_features.exact_dequantization) em vez de AerSimulator generico --
ver Risco R2 no plano e a nota de GPU abaixo.

Nota de GPU: se optar por AerSimulator genuino (nao a rota de dequantizacao),
instalar qiskit-aer-gpu e passar device="GPU" ao AerSimulator. So compensa
para os regimes de 78-156 qubits do CD-Ising-glass/XYZ (blocos ainda rasos,
mas em volume maior de amostras x configuracoes x dobras de CV).
"""

from __future__ import annotations

import pandas as pd


def generate_cd_ising_features(X: pd.DataFrame, qpu_topology: str = "ibm_kingston", **kwargs):
    """CDIsingProjectiveQFM do pqfmlib, observaveis <Z> e <ZZ>."""
    raise NotImplementedError("TODO: instanciar pqfmlib.CDIsingProjectiveQFM(...).run()")


def generate_heisenberg_features(X: pd.DataFrame, **kwargs):
    """HeisenbergProjectiveQFM do pqfmlib, observaveis <X>, <Y>, <Z>."""
    raise NotImplementedError("TODO: instanciar pqfmlib.HeisenbergProjectiveQFM(...).run()")


def generate_xyz_features(X: pd.DataFrame, variant: str = "diagonal", **kwargs):
    """XYZProjectiveQFM do pqfmlib.

    variant in {"diagonal", "cross", "full", "shared_feature"} -- ver
    Secao 2.4 da dissertacao / Eqs. 31-39 do arXiv:2606.13641.
    """
    raise NotImplementedError("TODO: instanciar pqfmlib.XYZProjectiveQFM(...).run()")
