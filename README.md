# Repository-about-HHT
# converter_lote_edf_duplo.py
import os
import numpy as np
import pyedflib
from tqdm import tqdm


def converter_edf_para_npy(caminho_entrada, caminho_saida):
    try:
        with pyedflib.EdfReader(caminho_entrada) as f:
            n_canais = f.signals_in_file
            sinais = np.zeros((n_canais, f.getNSamples()[0]))
            for i in range(n_canais):
                sinais[i, :] = f.readSignal(i)

        np.save(caminho_saida, sinais)
        return True, "Sucesso"
    except Exception as e:
        return False, str(e)


def processar_pasta_geral(pasta_base, pasta_destino):
    os.makedirs(pasta_destino, exist_ok=True)
    log_path = os.path.join("logs", "log_conversao.txt")
    os.makedirs("logs", exist_ok=True)

    with open(log_path, 'w') as log:
        for categoria in ["epilepsy", "no_epilepsy"]:
            entrada = os.path.join(pasta_base, categoria)
            saida = os.path.join(pasta_destino, categoria)
            os.makedirs(saida, exist_ok=True)

            for root, _, files in os.walk(entrada):
                for arquivo in tqdm(files, desc=f"Convertendo {categoria}"):
                    if arquivo.endswith(".edf"):
                        caminho_edf = os.path.join(root, arquivo)
                        nome_base = os.path.splitext(arquivo)[0]
                        caminho_npy = os.path.join(saida, f"{nome_base}.npy")
                        sucesso, msg = converter_edf_para_npy(caminho_edf, caminho_npy)
                        log.write(f"{arquivo}: {msg}\n")


if __name__ == "__main__":
    base = "/home/josue/Documents/Dados do Estágio/edf"
    destino = "/home/josue/Documents/Dados do Estágio/numpy"
    processar_pasta_geral(base, destino)


# preprocessar_lote_npy.py
import os
import numpy as np
import scipy.signal as signal
from tqdm import tqdm


def preprocessar_sinal(sinal, fs=256):
    sinal_filtrado = signal.medfilt(sinal, kernel_size=5)
    sinal_filtrado = signal.detrend(sinal_filtrado)
    sinal_filtrado = signal.detrend(sinal_filtrado, type='linear')
    sinal_filtrado = (sinal_filtrado - np.mean(sinal_filtrado)) / np.std(sinal_filtrado)
    return sinal_filtrado


def processar_pasta(caminho_entrada, caminho_saida):
    os.makedirs(caminho_saida, exist_ok=True)
    for classe in ["epilepsy", "no_epilepsy"]:
        entrada = os.path.join(caminho_entrada, classe)
        saida = os.path.join(caminho_saida, classe)
        os.makedirs(saida, exist_ok=True)

        for arquivo in tqdm(os.listdir(entrada), desc=f"Processando {classe}"):
            if not arquivo.endswith(".npy"):
                continue
            try:
                caminho_arquivo = os.path.join(entrada, arquivo)
                dados = np.load(caminho_arquivo)
                dados_processados = np.array([preprocessar_sinal(canal) for canal in dados if np.std(canal) > 0.1])

                if dados_processados.size == 0:
                    continue
                np.save(os.path.join(saida, arquivo), dados_processados)
            except Exception as e:
                print(f"Erro ao processar {arquivo}: {e}")


if __name__ == "__main__":
    entrada = "/home/josue/Documents/Dados do Estágio/numpy"
    saida = "/home/josue/Documents/Dados do Estágio/preprocessado"
    processar_pasta(entrada, saida)


# verificar_processamento.py
import os
import numpy as np
import pandas as pd


def contar_amostras(caminho):
    try:
        dados = np.load(caminho)
        return dados.shape[1]
    except:
        return 0


def verificar(pasta_original, pasta_preproc):
    resultados = []

    for classe in ["epilepsy", "no_epilepsy"]:
        pasta_o = os.path.join(pasta_original, classe)
        pasta_p = os.path.join(pasta_preproc, classe)

        for arquivo in os.listdir(pasta_o):
            if not arquivo.endswith(".npy"):
                continue

            path_o = os.path.join(pasta_o, arquivo)
            path_p = os.path.join(pasta_p, arquivo)
            original = contar_amostras(path_o)
            preproc = contar_amostras(path_p)
            descartadas = original - preproc
            aviso = "OK" if preproc > 0 else "Descartado ou falha"

            resultados.append([arquivo, original, preproc, descartadas, aviso])

    df = pd.DataFrame(resultados, columns=["Arquivo", "Original (amostras)", "Pré-processado", "Descartado", "Aviso"])
    os.makedirs("logs", exist_ok=True)
    df.to_csv("logs/verificacao_preprocessamento.csv", index=False)


if __name__ == "__main__":
    original = "/home/josue/Documents/Dados do Estágio/numpy"
    preproc = "/home/josue/Documents/Dados do Estágio/preprocessado"
    verificar(original, preproc)


# extrair_hht_lote.py
import os
import numpy as np
import pandas as pd
from PyEMD import EMD
from scipy.signal import hilbert
from tqdm import tqdm


def extrair_hht(sinal, fs=256):
    emd = EMD()
    imfs = emd(sinal)
    resultados = []
    for imf in imfs:
        if len(imf) < 2:
            continue
        h = hilbert(imf)
        amplitude = np.abs(h)
        fase = np.unwrap(np.angle(h))
        freq_inst = np.diff(fase) * fs / (2 * np.pi)
        resultados.append((amplitude[1:], freq_inst))
    return resultados


def processar_hht_lote(pasta_preproc, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    log_path = os.path.join("logs", "log_extracao_hht.txt")
    with open(log_path, 'w') as log:
        for classe in ["epilepsy", "no_epilepsy"]:
            entrada = os.path.join(pasta_preproc, classe)
            saida = os.path.join(pasta_saida, classe)
            os.makedirs(saida, exist_ok=True)

            for arquivo in tqdm(os.listdir(entrada), desc=f"Extraindo HHT - {classe}"):
                if not arquivo.endswith(".npy"):
                    continue
                try:
                    dados = np.load(os.path.join(entrada, arquivo))
                    resultado_arquivo = []
                    for canal in dados:
                        resultado_arquivo.append(extrair_hht(canal))
                    np.save(os.path.join(saida, arquivo), resultado_arquivo)
                    log.write(f"{arquivo}: OK\n")
                except Exception as e:
                    log.write(f"{arquivo}: Erro - {e}\n")


if __name__ == "__main__":
    pasta_preproc = "/home/josue/Documents/Dados do Estágio/preprocessado"
    pasta_saida = "/home/josue/Documents/Dados do Estágio/hht_resultados"
    processar_hht_lote(pasta_preproc, pasta_saida)
