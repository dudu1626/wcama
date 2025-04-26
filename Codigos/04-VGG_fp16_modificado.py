import io
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynvml
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from codecarbon import EmissionsTracker  # ## CODECARBON
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from thop import profile
from torch.amp import GradScaler, autocast  # Quantização de precisão mista
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from torchvision import datasets, transforms

## Configurações
plt.style.use("seaborn-v0_8")
sns.set_theme()

# Constante para inicialização do gerador de números aleatórios
SEED = 158763

## Definindo semente
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## Seed
set_seed()

# Verifica se a GPU está disponível
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dispositivo)

# Inicializa a biblioteca pynvml
def monitorar_gpu():
    # Inicializa o NVML para monitoramento da GPU
    pynvml.nvmlInit()

    # Obtém o handle da primeira GPU (índice 0)
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    memoria_usada = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)  # MB
    uso_gpu = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    consumo_energia = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # W
    
    pynvml.nvmlShutdown()

    return memoria_usada, uso_gpu, consumo_energia


# Função para criar um diretório com incremento, se necessário
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Cria o diretório pai 'alexNetMNIST_' com incremento, se necessário
nome_modelo = 'VGG11-FP16'
diretorio_pai = criar_diretorio_incrementado('resultadosVGG11', nome_modelo)
print(f'Diretório criado: {diretorio_pai}')

# Cria o diretório 'AlexNetCarbon'
diretorio_carbon = criar_diretorio_incrementado(diretorio_pai, 'VGG11_carbono')
print(f'Diretório Carbono criado: {diretorio_carbon}')

# Carrega e normaliza o CIFAR10
transformacao = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     transforms.RandomHorizontalFlip(),
                                     #transforms.RandomRotation(10), # testar se melhora
                                     ])


conjunto_treino_completo = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformacao)
conjunto_teste = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformacao)

classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Divide o conjunto de treino em treino e validação
tamanho_treino = int(0.8 * len(conjunto_treino_completo))
tamanho_validacao = len(conjunto_treino_completo) - tamanho_treino
conjunto_treino, conjunto_validacao = random_split(conjunto_treino_completo, [tamanho_treino, tamanho_validacao])

tamanho_batch = 32
num_nucleos = min(4, torch.get_num_threads())  # Ajuste baseado na máquina

carregador_treino = DataLoader(conjunto_treino, batch_size=tamanho_batch, shuffle=True, num_workers=num_nucleos)
carregador_validacao = DataLoader(conjunto_validacao, batch_size=tamanho_batch, shuffle=False, num_workers=num_nucleos)
carregador_teste = DataLoader(conjunto_teste, batch_size=tamanho_batch, shuffle=False, num_workers=num_nucleos)

## Definição da arquitetura da rede
class VGG11(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0):
        super(VGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloco 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloco 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def inicializar_pesos(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

modelo = VGG11().to(dispositivo)
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.0001)
scaler = GradScaler(dispositivo)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=5, factor=0.1)

print(modelo)

modelo.apply(inicializar_pesos)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um buffer de string
sys.stdout = buffer = io.StringIO()

# Chamar a função summary
summary(modelo, (3, 32, 32))

# Obter o valor da string do buffer
resumo_str = buffer.getvalue()

# Restaurar a saída padrão original
sys.stdout = saida_padrao_original

# Salvar a string de resumo em um arquivo
with open(f'{diretorio_pai}/resumo_modelo.txt', 'w') as f:
    f.write(resumo_str)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um arquivo
sys.stdout = open(f'{diretorio_pai}/saida.txt', 'w')

# Definições iniciais
maximo_epocas = 50
tempos_treino = []
potencias_treino = []

# Função para treinar e validar um modelo
def treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, epocas, i):
    #early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    modelo.train()
    tempo_inicio= datetime.now()
    for epoca in range(epocas):
        perda_acumulada = 0.0
        corretos = 0
        total = 0
        for i, dados in enumerate(carregador_treino, 0):
            entradas, rotulos = dados[0].to(dispositivo).half(), dados[1].to(dispositivo)
            otimizador.zero_grad()

            with autocast('cuda'):
                saidas = modelo(entradas)
                perda = criterio(saidas, rotulos)
        
            scaler.scale(perda).backward()

            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)  # Aplicar gradient clipping
            
            scaler.step(otimizador)
            scaler.update()          
            
            perda_acumulada += perda.item()
            _, previstos = torch.max(saidas.data, 1)
            total += rotulos.size(0)
            corretos += (previstos == rotulos).sum().item()


        perda_treino = perda_acumulada / len(carregador_treino)
        acuracia_treino = corretos / total
        print(f'Época {epoca + 1}, Perda Treino: {perda_treino:.4f}, Acurácia Treino: {acuracia_treino:.4f}')

        # Validação
        modelo.eval()
        perda_validacao = 0.0
        corretos = 0
        total = 0
        with torch.no_grad():
            for dados in carregador_validacao:
                imagens, rotulos = dados[0].to(dispositivo).half(), dados[1].to(dispositivo)
                
                with autocast('cuda'):
                    saidas = modelo(imagens)

                perda = criterio(saidas, rotulos)
                perda_validacao += perda.item()
                _, previstos = torch.max(saidas.data, 1)
                total += rotulos.size(0)
                corretos += (previstos == rotulos).sum().item()
        
        perda_validacao /= len(carregador_validacao)
        acuracia_validacao = corretos / total
        
        print(
            f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Acurácia Validação: {acuracia_validacao:.4f}')
        
        scheduler.step(perda_validacao)

        ## Verifique o Early Stopping
        #early_stopping(perda_validacao)
        #if early_stopping.early_stop:
        #    print(f"Early stopping na época {epoca + 1}")
        #    break

        memoria_usada_treino, uso_gpu_treino, consumo_energia_treino = monitorar_gpu() # monitoramento por época
        potencias_treino.append(consumo_energia_treino)
           
    tempo_fim = datetime.now()
    tempo_treino = (tempo_fim - tempo_inicio)
    tempos_treino.append(tempo_treino.total_seconds())

    consumo_energia_final = sum(potencias_treino)

    potencias_treino.clear()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino.total_seconds(), consumo_energia_final

# Treinamento e seleção do melhor modelo entre 10 candidatos
numero_modelos = 10
medias_acuracia_validacao = []
indice_melhor_modelo = -1
melhor_modelo = None
modelos = []
metricas = []
media_metricas = []
emissoes_treino = [] #Adicionado

# ## Inicializa o rastreador de emissões
print("Iniciando treinamento...")
tracker_train = EmissionsTracker(output_dir= diretorio_carbon,output_file="emissoes_treino.csv")
tracker_train.start()

def criar_modelo():
    modelo = VGG11().to(dispositivo)
    otimizador = optim.Adam(modelo.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, 'min', patience=5, factor=0.1)
    criterio = nn.CrossEntropyLoss()

    return modelo, otimizador, scheduler, criterio

for i in range(numero_modelos):
    print("______________________________________________________________________________________________________")
    print(f'Treinando modelo {i + 1}/{numero_modelos}')

    # Inicia o rastreador de emissões para este treinamento ---------------------
    tracker_train = EmissionsTracker(output_dir=diretorio_carbon, 
                                     output_file=f"emissoes_treino_modelo_{i+1}.csv",
                                     allow_multiple_runs=True)
    tracker_train.start()

    entrada = torch.randn(1, 3, 32, 32).to(dispositivo)
    
    modelo, otimizador, scheduler, criterio = criar_modelo()

    flops, parametros = profile(modelo, inputs=(entrada,), verbose=False)

    perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia = (
        treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, maximo_epocas, i))
    

    # Para o rastreador e armazena as emissões  ----------------------------
    emissoes = tracker_train.stop()
    emissoes_treino.append(emissoes)


    metricas.append((perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia, emissoes))

    
    # Calcular a média das métricas após o treino de cada modelo
    media_perda_treino = np.mean([m[0] for m in metricas])
    media_acuracia_treino = np.mean([m[1] for m in metricas])
    media_perda_validacao = np.mean([m[2] for m in metricas])
    media_acuracia_validacao = np.mean([m[3] for m in metricas])
    media_tempo_treino = np.mean([m[4] for m in metricas]) ## adicionado depois
    media_consumo_energia = np.mean([m[5] for m in metricas]) ## adicionado depois
    media_emissoes = np.mean([m[6] for m in metricas]) ## adicionado depois

    print(f'Modelo {i + 1}: Média Perda Treino: {media_perda_treino:.4f}, Média Acurácia Treino: {media_acuracia_treino:.4f}, '
            f'Média Perda Validação: {media_perda_validacao:.4f}, Média Acurácia Validação: {media_acuracia_validacao:.4f}')
    print(f'Tempo médio de treino: {media_tempo_treino}')
    print(f'FLOPs: {flops}')
    print(f'Parâmetros: {parametros}')
    print(f'Consumo médio de energia: {media_consumo_energia} W')
    print(f'Emissões médias: {media_emissoes} kg CO₂')

    medias_acuracia_validacao.append(media_acuracia_validacao)
    media_metricas.append(
        (media_perda_treino, media_acuracia_treino, media_perda_validacao, media_acuracia_validacao,
            media_tempo_treino, media_consumo_energia, media_emissoes))
    modelos.append(modelo)

# tracker_train.stop()
# print(f"Emissões durante o treinamento total: {tracker_train.final_emissions} kg CO₂")

# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas = pd.DataFrame(media_metricas, columns=['Média Perda Treino', 'Média Acurácia Treino', 'Média Perda Validação',
                                                    'Média Acurácia Validação', 'Tempo médio Treino', 'Consumo médio Energia Treino',
                                                    'Emissões Treino'])

# # Criar dataframe com as métricas codecarbon
# df_codecarbon = pd.DataFrame(resultado_emissoes)
# df.to_csv(f'{diretorio_pai}/emissoes_treino_carbon.csv', index=False) # Adicionado

# Adiciona uma coluna 'Modelo_x' ao DataFrame
nomes_modelos = ['Modelo_' + str(i + 1) for i in range(numero_modelos)]
df_metricas.insert(0, 'Modelo', nomes_modelos)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas.to_excel(f'{diretorio_pai}/metricas_modelos_treino.xlsx', index=False)

# Seleciona o melhor modelo com base na maior acurácia de validação
indice_melhor_modelo = medias_acuracia_validacao.index(max(medias_acuracia_validacao))
melhor_modelo = modelos[indice_melhor_modelo]

print('************************************************************************************************')
print(f'O melhor modelo é o {nomes_modelos[indice_melhor_modelo]} com a maior média de acurácia de validação: {medias_acuracia_validacao[indice_melhor_modelo]:.4f}')
print('************************************************************************************************')

# Calcular a média dos tempos de treino e consumo de energia
media_tempo_treino = np.mean(tempos_treino)
media_consumo_energia = np.mean(potencias_treino)
print(f'Tempo Médio de Treino: {media_tempo_treino} segundos')
print(f'Consumo Médio de Energia: {media_consumo_energia} W')

potencias_teste = []

# Coleta as métricas de desempenho na inferência
# @track_emissions(project_name="VGG11_original", output_dir=diretorio_carbon)
def inferencia_e_metricas(modelo, teste_loader):
    # ativar o modo de avaliação
    modelo.eval()

    rotulos_reais = []
    rotulos_preditos = []

    # Medição de tempo de inferência - Início
    inicio_tempo_teste = datetime.now()

    with torch.no_grad():
        for dados in teste_loader:
            imagens, rotulos = dados
            imagens, rotulos = imagens.to(dispositivo).half(), rotulos.to(dispositivo)

            with autocast('cuda'):
                saidas = modelo(imagens)

            _, preditos = torch.max(saidas, 1)

            rotulos_reais.extend(rotulos.cpu().numpy())
            rotulos_preditos.extend(preditos.cpu().numpy())

            memoria_usada_teste, uso_gpu_teste, consumo_energia_teste = monitorar_gpu() # monitoramento por época
            potencias_teste.append(consumo_energia_teste)

    # Medição de tempo de inferência - Fim
    fim_tempo_teste = datetime.now()

    acuracia = accuracy_score(rotulos_reais, rotulos_preditos)
    precisao = precision_score(rotulos_reais, rotulos_preditos, average='macro', zero_division=0)
    recall = recall_score(rotulos_reais, rotulos_preditos, average='macro')
    f1 = f1_score(rotulos_reais, rotulos_preditos, average='macro')
    potencias_teste_final = sum(potencias_teste)

    potencias_teste.clear()
    
    cm = confusion_matrix(rotulos_reais, rotulos_preditos)

    tempo_inferencia = (fim_tempo_teste - inicio_tempo_teste)

    return acuracia, precisao, recall, f1, tempo_inferencia.total_seconds(), cm, potencias_teste_final

# Coleta as métricas de desempenho na inferência
acuracia, precisao, recall, f1, tempo_inferencia, cm, energia = inferencia_e_metricas(melhor_modelo, carregador_teste)



# Chamar a função summary
summary(melhor_modelo, (3, 32, 32))

# Obter o valor da string do buffer
resumo_melhor_str = buffer.getvalue()

# Restaurar a saída padrão original
sys.stdout = saida_padrao_original

# Salvar a string de resumo em um arquivo
with open(f'{diretorio_pai}/resumo_melhor_modelo.txt', 'w') as f:
    f.write(resumo_melhor_str)



# Calcula e imprime as métricas de desempenho da primeira inferência
print(f'Acurácia: {acuracia:.4f}')
print(f'Precisão: {precisao:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Tempo de Inferência total: {tempo_inferencia} segundos')
print(f'Consumo de Energia total: {energia:.4f} W')


# Inicializa listas para armazenar tempos de todas as inferências subsequentes
tempos_teste = []
acuracia_acumulada = []
precisao_acumulada = []
recall_acumulada = []
f1_acumulada = []
consumo_energia_acumulada = []
metricas_teste = []
emissoes_teste = []


## Inicializa o rastreador de emissões
print("Iniciando inferência...")

## Realiza 10 inferências para medir o tempo
for i in range(10):
    
    # Inicia o rastreador de emissões para este teste
    tracker_infer = EmissionsTracker(output_dir=diretorio_carbon, 
                                     output_file=f"emissoes_teste_{i+1}.csv",
                                     allow_multiple_runs=True)
    tracker_infer.start()

    acuracia, precisao, recall, f1, tempo_inferencia, _, consumo_energia_teste = inferencia_e_metricas(melhor_modelo, carregador_teste)

    # Para o rastreador e armazena as emissões
    emissoes = tracker_infer.stop()
    emissoes_teste.append(emissoes)

    tempos_teste.append(tempo_inferencia)
    acuracia_acumulada.append(acuracia)
    precisao_acumulada.append(precisao)
    recall_acumulada.append(recall)
    f1_acumulada.append(f1)
    consumo_energia_acumulada.append(consumo_energia_teste)

    metricas_teste.append((acuracia, precisao, recall, f1, tempo_inferencia, consumo_energia_teste, emissoes))


    # Calcular a média das métricas após o teste de inferência
    media_acuracia_teste = sum(acuracia_acumulada) / len(acuracia_acumulada)
    media_precisao_teste = sum(precisao_acumulada) / len(precisao_acumulada)
    media_recall_teste = sum(recall_acumulada) / len(recall_acumulada)
    media_f1_teste = sum(f1_acumulada) / len(f1_acumulada)
    media_tempo_teste = sum(tempos_teste) / len(tempos_teste)
    media_consumo_energia_teste = sum(consumo_energia_acumulada) / len(consumo_energia_acumulada) # teho dúvida quanto a essa média

    # metricas_teste.append((media_acuracia_teste, media_precisao_teste, media_recall_teste,
    #                     media_f1_teste, media_tempo_teste, media_consumo_energia_teste))  # modificado


# tracker_infer.stop()
# print(f"Emissões durante a inferência: {tracker_infer.final_emissions} kg CO₂")


# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas_teste = pd.DataFrame(metricas_teste, columns=['Média Acurácia Teste', 'Média Precisão Teste', 'Média Recall Teste',
                                                          'Média F1 Score Teste', 'Tempo médio Teste', 'Consumo médio Energia Teste',
                                                          'Emissões Teste'])

# Adiciona uma coluna 'Modelo_x' ao DataFrame
df_metricas_teste.insert(0, 'Modelo', nome_modelo)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas_teste.to_excel(f'{diretorio_pai}/metricas_modelos_teste.xlsx', index=False)


## Dataframe de emissões --------------------
# Cria um DataFrame com as emissões de cada treinamento
df_emissoes_treino = pd.DataFrame({
    'Modelo': nomes_modelos,
    'Emissões de CO2 (kg)': emissoes_treino
})

# Salva as emissões em um arquivo CSV
df_emissoes_treino.to_csv(f'{diretorio_pai}/emissoes_treino.csv', index=False)


# Cria um DataFrame com as emissões de cada teste
df_emissoes_teste = pd.DataFrame({
    'Teste': range(1, 11),
    'Emissões de CO2 (kg)': emissoes_teste
})

# Salva as emissões em um arquivo CSV
df_emissoes_teste.to_csv(f'{diretorio_pai}/emissoes_teste.csv', index=False)

# ----------------------------------------------

# Imprime a média do tempo de teste
print(f'Média da Acurácia: {media_acuracia_teste:.4f}')
print(f'Média da Precisão: {media_precisao_teste:.4f}')
print(f'Média do Recall: {media_recall_teste:.4f}')
print(f'Média do F1 Score: {media_f1_teste:.4f}')
print(f'Média do Tempo de Inferência: {media_tempo_teste:.4f} segundos')
print(f'Média do Consumo de Energia: {media_consumo_energia_teste:.4f} W')

sys.stdout.close()
sys.stdout = saida_padrao_original


# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
plt.xlabel('Previstos')
plt.ylabel('Verdadeiros')

# Salvar a figura
plt.savefig(f'{diretorio_pai}/matriz_confusao.png')
plt.close()


# Salva as métricas da inferencia em um arquivo
with open(f'{diretorio_pai}/metricas_inferencia.txt', 'w') as f:
    f.write(f'Acurácia: {media_acuracia_teste:.4f}\n')
    f.write(f'Precisão: {media_precisao_teste:.4f}\n')
    f.write(f'Revocação: {media_precisao_teste:.4f}\n')
    f.write(f'Medida-F1: {media_f1_teste:.4f}\n')
    f.write(f'Média do Tempo de Teste: {media_tempo_teste} segundos\n')
    f.write(f'Média do Consumo de Energia: {media_consumo_energia_teste} W\n')

# pynvml.nvmlShutdown()
print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')