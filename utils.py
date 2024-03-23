import io
import soundfile as sf
import preproccess_kpv as kpv

def audio_streamer(audio_data, sample_rate):
    """
    Generator function to stream audio data in chunks.

    :param audio_data: The audio data to be streamed.
    :param sample_rate: The sample rate of the audio data.
    :return: A generator yielding chunks of audio data.
    """
    audio_stream = io.BytesIO()
    sf.write(audio_stream, audio_data, sample_rate, format='wav')
    audio_stream.seek(0)

    chunk_size = 4096  # Size of each chunk in bytes
    while True:
        chunk = audio_stream.read(chunk_size)
        if not chunk:
            break
        yield chunk

def process_text_for_tts(lang, text):
    """
    Preprocesses the text to be synthesized.

    :param text: The text to be synthesized.
    :return: The preprocessed text.
    """
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")
    if lang == "kpv":
        text = kpv.proccess(text)
    return text

def replace_unsupported_chars(text):
    unsupported_chars_dict = {'«': '"', '»': '"', '„': '"', '“': '"', '”': '"', '…': '...', '—': '-', '–': '-', '№': '#', '’': "'", '‘': "'", '‚': "'", '‛': "'", '‹': "'", '›': "'", '‟': '"'}
    for key, value in unsupported_chars_dict.items():
        text = text.replace(key, value)
    return text