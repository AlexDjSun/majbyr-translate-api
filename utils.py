import io
import re
import soundfile as sf

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

UNITS = ['', 'ӧтик', 'кык', 'куим', 'нёль', 'вит', 'квайт', 'сизим', 'кӧкъямыс', 'ӧкмыс']
TENS = ['', 'дас', 'кызь', 'комын', 'нелямын', 'ветымын', 'квайтымын', 'сизимдас', 'кӧкъямысдас', 'ӧкмысдас']
HUNDREDS = "сё"
THOUSANDS = "сюрс"
MILLIONS = "миллён"

INVALID_VOCAB   =  [' i', 'je', 'jo', 'ju', 'ja', 'a', 'ä', 'á', 'å', 'b', 'c', 'č', 'ć', 'd', 'ď', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'ľ', 'm', 'n', 'ń',
					'o', 'ö', 'õ', 'p', 'q', 'r', 's', 'š', 'ś', 't', 'ť', 'u', 'ü', 'ú', 'v', 'w', 'x', 'y', 'z', 'ž', 'ź', "\'", "’", "+"]
VALID_VOCAB 	=  [' и', 'е', 'ё', 'ю', 'йа', 'а', 'а', 'а', 'о', 'б', 'ц', 'тш', 'ч', 'д', 'дь', 'э', 'ф', 'г', 'х', 'й', 'к', 'л', 'ль', 'м', 'н', 'нь',
                    'о', 'ö', 'ö', 'п', 'к', 'р', 'с', 'ш', 'сь', 'т', 'ть', 'у', 'ы', 'у', 'в', 'в', 'кс', 'ы', 'з', 'ж', 'зь', "ь", "ь", "плюс "]

def convert_group_to_words(group, group_index):
    if int(group[0]) == 0:
        group_index = 0
    num_list = []
    i = 0
    for digit in group:
        if i == 0:
            num_list.append(UNITS[int(digit)])
        elif i == 1:
            num_list.append(TENS[int(digit)])
        elif i == 2:
            if digit != '0':
                num_list.append(HUNDREDS)
            num_list.append(UNITS[int(digit)])
        i += 1
    # reverse the list
    num_list = num_list[::-1]
    num_str = ''.join(num_list)
    return num_str + {0: '', 1: THOUSANDS, 2: MILLIONS} [group_index]

def convert_number_to_words(num):
    if len(num) > 9:
        numstr = ""
        for digit in num:
            if digit == '0':
                numstr = numstr + " нуль"
            else:
                numstr = numstr + ' ' + UNITS[int(digit)]
        return numstr

    # split number into groups of 3 digits
    num = num[::-1]
    num_groups = [num[i:i+3] for i in range(0, len(num), 3)]
    # convert each group of 3 digits to words
    ordered_num = [convert_group_to_words(group, i) for i, group in enumerate(num_groups)]
    return ''.join(ordered_num[::-1])

def transliterate(text):
    text = text.lower()
    for i in range(len(INVALID_VOCAB)):
        text = text.replace(INVALID_VOCAB[i], VALID_VOCAB[i])
    return text

def replace_digit_nums(text):
	numbers = re.findall(r'\d+', text)
	if not numbers:
		return text
	for number in numbers:
		text = text.replace(number, convert_number_to_words(number))
	return text

def process_text(lang, text):
    """
    Preprocesses the text to be synthesized.

    :param text: The text to be synthesized.
    :return: The preprocessed text.
    """
    print('process_text', lang, text)
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")
    if lang == "kpv":
        print('kpv')
        text = transliterate(text)
        text = replace_digit_nums(text)
        text = " " + text.replace("ӧ", "ö").replace("і", "i").replace("\n", "\n  ").replace("ц", "тс").replace("щ", "шш")
        print(text)
    return text
