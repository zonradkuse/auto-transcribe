# This tool/script transcribes interviews using vosk
#
# Input is any 16kHz mono PCM wav file.
# Use ffmpeg to convert any audio input file:
# ffmpeg -i in.wav -acodec pcm_s16le -ac 1 -ar 16000 out.wav
#
# The output will be in SubRip Subtitle Format.
#
# The accuracy of the German model is suboptimal if the audio quality is not already great (close 
# to perfect). You probably want to post-process the result and compare it to the recording.
#
if __name__ != '__main__':
    print("This file is not intended to be imported")
    exit(1)

import sys
import wave
import json
import logging
import os
import argparse
import srt
import datetime
import multiprocessing

import recasepunc
from vosk import Model, KaldiRecognizer, SpkModel
import numpy as np
from sklearn.cluster import KMeans

# must be in __main__
from recasepunc import WordpieceTokenizer

# disable vosk logs
from vosk import SetLogLevel
SetLogLevel(-1)

parser = argparse.ArgumentParser(description='Transcribe interviews.')
parser.add_argument('--language-model', type=str,
                    help='The path to the vosk language model')
parser.add_argument('--speaker-model', type=str,
                    help='The path to the vosk speaker model')
parser.add_argument('--punctuation-model', type=str,
                    help='The path to the punctuation model')
parser.add_argument('--speakers', type=int, default=2,
                    help='Number of speakers during the interview')
parser.add_argument('--log-level', type=str, default="INFO",
                    help='Set the loglevel')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='Set the number of threads allowed to use for processing multiple audio files')
parser.add_argument('audio', nargs = '+', help = 'The audio file to transcribe')

args = parser.parse_args()

numeric_level = getattr(logging, args.log_level.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)

logging.basicConfig(level=numeric_level)

if args.speaker_model is None or args.language_model is None:
    print("Language and speaker model must both be specified!")
    parser.print_help()
    exit(1)

if not os.path.exists(args.speaker_model):
    print ("Please download the speaker model from https://alphacephei.com/vosk/models and unpack as {} in the current folder.".format(args.speaker_model))
    exit (1)

if not os.path.exists(args.language_model):
    print ("Please download the language model from https://alphacephei.com/vosk/models and unpack as {} in the current folder.".format(args.language_model))
    exit (1)

# Large vocabulary free form recognition
model = Model(model_path=args.language_model)
spk_model = SpkModel(args.speaker_model)

def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)


def transcribe_audio(audio_path):
    # TODO replace by ffmpeg preprocessing
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print ("Audio file must be WAV format mono PCM.")
        exit (1)

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetSpkModel(spk_model)

    result_data = []

    while True:
        data = wf.readframes(600)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            logging.debug(f"VOSK intermediate result {res}")
            result_data.append(res)

    # add remaining transcription at end of file
    res = json.loads(rec.FinalResult())
    result_data.append(res)


    # Use first voice signature as baseline
    spk_sig = None
    for res in result_data:
        if 'spk' in res:
            spk_sig = res['spk']
            break

    # If there is no speaker signature, use a default vector (1x128) instead. Use 0.5 as value.
    # 0-vector would only yield 0 distance and 1 vector would yield a meaningless distance.
    if spk_sig is None:
        spk_sig = [0.5] * 128

    print(f"Voice signature chosen as {spk_sig}")

    # cluster voice signature using k-means.
    voice_signatures = [[0,cosine_dist(spk_sig, res['spk'])] for res in result_data if 'spk' in res]
    km = KMeans(n_clusters = args.speakers).fit(voice_signatures)

    # Join consecutive results which belong to the same cluster
    speaker_ordered_transcript = []
    current_speaker_results = []
    current_speaker = -1 # unknown speaker
    for res in result_data:
        # check and set speaker
        if 'spk' in res:
            similarity = cosine_dist(spk_sig, res['spk'])
            predicted_speaker_id = km.predict([[0, similarity]])
            if current_speaker == -1:
                # current speaker not set yet but given
                current_speaker = predicted_speaker_id
            elif current_speaker != predicted_speaker_id:
                # there has likely been a change of speakers
                speaker_ordered_transcript.append({
                    "speaker": current_speaker,
                    "texts": current_speaker_results
                })
                current_speaker_results = []
                current_speaker = predicted_speaker_id

        current_speaker_results.append(res)

    speaker_ordered_transcript.append({
                    "speaker": current_speaker,
                    "texts": current_speaker_results
                })

    # Restore punctuation for each speaker block if enabled
    speaker_texts = []
    for i, res in enumerate(speaker_ordered_transcript):
        spoken = ''
        for recognized in res['texts']:
            text = recognized['text']
            spoken += text + ' '
        speaker_texts.append(spoken)

    if args.punctuation_model:
        logging.info("Reconstructing punctuation and casing")
        punctuated_texts = recasepunc.generate_predictions(None, args.punctuation_model, speaker_texts)
    else:
        punctuated_texts = speaker_texts

    srt_entries = []
    # limit line length
    _LINE_LENGTH = 13
    for i, res in enumerate(speaker_ordered_transcript):
        for recognized in res['texts']:
            words = punctuated_texts[i].split()
            lines = []
            lines.append(f"[Speaker {res['speaker']}]")
            for j in range(0, len(words), _LINE_LENGTH):
                   line = words[j : j + _LINE_LENGTH]
                   lines.append(' '.join(line))

        # find start and end times
        start_seconds = None
        for r in res['texts']:
            if 'result' not in r:
                continue

            start_seconds = r['result'][0]['start']
            break

        end_seconds = None
        for r in reversed(res['texts']):
            if 'result' not in r:
                continue

            end_seconds = r['result'][-1]['end']
            break


        s = srt.Subtitle(index=i,
            content = '\n'.join(lines),
            start=datetime.timedelta(seconds=start_seconds),
            end=datetime.timedelta(seconds=end_seconds))
        srt_entries.append(s)

    # Write output
    out_path = f"{audio_path}.srt"
    logging.info(f"Writing result to {out_path}")
    with open(out_path, 'w') as f:
        f.write(srt.compose(srt_entries))


def main():

    pool = multiprocessing.Pool(args.threads)
    for out in pool.imap_unordered(transcribe_audio, args.audio):
        pass

main()

