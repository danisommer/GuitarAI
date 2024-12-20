import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Função para processar o áudio e extrair o cromagrama
def extract_chromagram(audio_path):
    # Carregar o áudio
    y, sr = librosa.load(audio_path, sr=None)

    # Extrair o cromagrama
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

    return chromagram, sr

# Função para detectar simultaneidade ou sequência de notas
def classify_events(chromagram):
    # Definir um limiar para considerar uma frequência como ativa
    threshold = 0.3
    
    # Lista para armazenar os eventos classificados
    events = []

    # Iterar sobre cada quadro de tempo no cromagrama
    for i in range(chromagram.shape[1]):
        active_notes = np.where(chromagram[:, i] > threshold)[0]

        if len(active_notes) > 3:  # Muitas notas simultâneas -> acorde
            events.append("acorde")
        elif len(active_notes) > 0:  # Poucas notas sequenciais -> dedilhado
            events.append("dedilhado")
        else:
            events.append("silêncio")  # Nenhuma nota detectada

    return events

# Função para mapear notas a acordes simples
def map_chord(chromagram):
    chords = []
    threshold = 0.3

    for i in range(chromagram.shape[1]):
        active_notes = np.where(chromagram[:, i] > threshold)[0]

        # Simplesmente mapear as notas detectadas para um acorde básico
        if len(active_notes) > 0:
            chord_name = "+".join([librosa.midi_to_note(60 + n) for n in active_notes])
            chords.append(chord_name)
        else:
            chords.append("N")  # Nenhum acorde detectado

    return chords

# Função principal para processar o áudio e transcrever acordes e dedilhados
def transcribe_audio(audio_path):
    chromagram, sr = extract_chromagram(audio_path)

    # Classificar eventos como acordes ou dedilhados
    events = classify_events(chromagram)

    # Mapear os eventos a acordes
    chords = map_chord(chromagram)

    # Combinar eventos e acordes em uma transcrição
    transcription = [(event, chord) for event, chord in zip(events, chords)]

    return transcription

# Visualização dos resultados
def visualize_transcription(transcription, chromagram):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.title("Cromagrama")
    plt.colorbar()

    for i, (event, chord) in enumerate(transcription):
        if event != "silêncio":
            plt.text(i * 0.1, 5, f"{event}\n{chord}", color="white", fontsize=8)

    plt.show()

# Testar o protótipo com um arquivo de áudio de exemplo
if __name__ == "__main__":
    audio_path = "sua_musica.mp3"  # Substitua pelo caminho do seu arquivo de áudio

    # Transcrever o áudio
    transcription = transcribe_audio(audio_path)

    # Exibir os resultados
    chromagram, _ = extract_chromagram(audio_path)
    visualize_transcription(transcription, chromagram)

    # Imprimir a transcrição final
    for i, (event, chord) in enumerate(transcription):
        print(f"{i}: {event} - {chord}")