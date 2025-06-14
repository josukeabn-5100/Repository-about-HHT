# extrair_hht_lote.py - Versão atualizada com foco em sinais epilépticos e canais específicos

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyemd import EMD
from scipy.signal import hilbert
import traceback

# Canais-alvo definidos pelo plano: F3, F4, C3, C4, P3, P4
CANAIS_INTERESSE = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

# Função para verificar se o nome do arquivo é do grupo com epilepsia
def is_epilepsy(filename):
    return 'no_epilepsy' not in filename.lower()

# Função para processar um único arquivo .npy
def processar_arquivo(path, salvar_em):
    try:
        dados = np.load(path, allow_pickle=True).item()
        sinais = dados.get("dados")
        canais = dados.get("canais")
        sfreq = dados.get("sfreq")
        
        if sinais is None or canais is None or sfreq is None:
            return "Formato inválido"

        indices = [i for i, nome in enumerate(canais) if nome in CANAIS_INTERESSE]
        if not indices:
            return "Sem canais de interesse"

        resultados = {}

        for i in indices:
            nome_canal = canais[i]
            sinal = sinais[i]

            if len(sinal) < 10:
                continue

            try:
                emd = EMD()
                imfs = emd(sinal)

                hilbert_imfs = hilbert(imfs, axis=1)
                amp = np.abs(hilbert_imfs)
                inst_phase = np.unwrap(np.angle(hilbert_imfs), axis=1)
                inst_freq = np.diff(inst_phase, axis=1) * sfreq / (2 * np.pi)

                resultados[nome_canal] = {
                    "imfs": imfs,
                    "amplitude": amp,
                    "frequencia": inst_freq
                }

            except Exception as canal_erro:
                resultados[nome_canal] = {"erro": str(canal_erro)}

        if not resultados:
            return "Sem dados úteis"

        nome_base = os.path.splitext(os.path.basename(path))[0]
        salvar_path = os.path.join(salvar_em, nome_base + "_hht.npy")
        np.save(salvar_path, resultados)
        return "OK"

    except Exception as e:
        return f"Erro: {str(e)}"

# Caminhos de entrada e saída
PASTA_ENTRADA = "/home/josue/Documents/Dados do Estágio/preprocessado"
PASTA_SAIDA = "/home/josue/Documents/Dados do Estágio/hht"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# Log para salvar resultados
log = []

for root, _, files in os.walk(PASTA_ENTRADA):
    for file in tqdm(files, desc=f"Processando {os.path.basename(root)}"):
        if file.endswith(".npy"):
            full_path = os.path.join(root, file)
            if is_epilepsy(full_path):
                resultado = processar_arquivo(full_path, PASTA_SAIDA)
                log.append({"arquivo": file, "status": resultado})

# Salvar log em CSV
log_path = os.path.join(PASTA_SAIDA, "log_extracao_hht.csv")
pd.DataFrame(log).to_csv(log_path, index=False)
print(f"Processamento concluído. Log salvo em: {log_path}")
