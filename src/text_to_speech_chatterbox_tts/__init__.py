import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device = "cpu"
)
texto = "Hola, bienvenidos a Inteligencia artificial generativa, vamos a crear audios, imagenes, texto, y si el pc agunta videos."

print("Generando audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, # nos ayuda con la expresión o emoción
    cfg_weight=0.4, # Nos ajusta el ritmo si la voz va demasiado rapida
    temperature=0.8, # Controla la variabilidad, más alta = más variabilidad sino sería menos predecible
    repetition_penalty=1.2, # Penaliza repeticiones extrañas
    min_p=0.05, # Filtra opciones improbables durante la generación
    top_p=1.0, # Controla el muestreo probabilistico, 1.0 lo deja abierto
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado.")
