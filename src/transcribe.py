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

import sys
import wave
import json
import logging
import os
import argparse
import srt
import datetime
import multiprocessing
import tempfile
import subprocess

import recasepunc
from vosk import Model, KaldiRecognizer, SpkModel
import numpy as np
from sklearn.cluster import KMeans
import librosa

# must be in __main__
from recasepunc import WordpieceTokenizer

# disable vosk logs
from vosk import SetLogLevel

parser = argparse.ArgumentParser(description='Transcribe interviews.')
parser.add_argument('--language-model', type=str,
                    help='The path to the vosk language model')
parser.add_argument('--punctuation-model', type=str,
                    help='The path to the punctuation model')
parser.add_argument('--speakers', type=int, default=2,
                    help='Number of speakers during the interview')
parser.add_argument('--speech-enhancement-on', action='store_true',
                    help="Use speechbrain's sppech enhancement. It usually does NOT enhance ASR accuracy")
parser.add_argument('--ffmpeg-strategy', type=str, choices=['MERGE', 'LEFT', 'RIGHT', 'AUTO'], default='AUTO',
                    help="Strategy to use for creating a mono file using ffmpeg. If you use stereo microphones, you can select from 'MERGE', 'LEFT' and 'RIGHT'. Using the appropriate channel can improve accuracy.")
parser.add_argument('--ffmpeg-noise-reduction', action='store_true', default=True,
                    help="Apply ffmpeg's NR filter.")
parser.add_argument('--ffmpeg-compress', action='store_true', default=True,
                    help="Apply ffmpeg's compand filter for compression.")
parser.add_argument('--no-ffmpeg-compress', dest='ffmpeg_compress', action='store_false')
parser.add_argument('--no-ffmpeg-noise-reduction', dest='ffmpeg_noise_reduction', action='store_false')
parser.add_argument('--log-level', type=str, default="INFO",
                    help='Set the loglevel')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='Set the number of threads allowed to use for processing multiple audio files')
parser.add_argument('audio', nargs = '+', help = 'The audio files to transcribe with the given settings')


def diarize_speakers(audio_path, nspeakers):
    from simple_diarizer.diarizer import Diarizer
    diar = Diarizer(
        embed_model='ecapa', # supported types: ['xvec', 'ecapa']
        cluster_method='sc', # supported types: ['ahc', 'sc']
        window=1.5, # size of window to extract embeddings (in seconds)
        period=0.75)

    segments = diar.diarize(audio_path, num_speakers=nspeakers)
    return concat_speaker_segments(segments)

def concat_speaker_segments(segments):
    current_speaker = segments[0]['label']
    result = []
    current_start = 0
    current_end = -1
    for i, segment in enumerate(segments):
        if segment['label'] != current_speaker:
            # speaker change detected - concatenate all previous segment times
            seg = {
                'speaker': current_speaker,
                'start': current_start,
                'end': segment['start']
            }

            result.append(seg)
            current_start = segments[i-1]['end']
            current_speaker = segment['label']

        current_end = segment['end']

    # last segment must still be added
    final_start_time = segments[-2]['end']
    final_seg = {
        'speaker': segments[-1]['label'],
        'start': final_start_time,
        'end': segments[-1]['end']
    }

    result.append(final_seg)

    return result

def transcribe_audio(audio_path):
    wf = wave.open(audio_path, "rb")

    logging.info("Diarizing speakers into segments...")

    segments = diarize_speakers(audio_path, args.speakers)
    segment_texts = []
    with tempfile.TemporaryDirectory() as tmp:
        logging.info(f"Writing segments as wave file to {tmp}")
        for i, segment in enumerate(segments):
            logging.info(f"Preprocessing segment {i + 1}/{len(segments)}")
            filename = f'{tmp}/{i}.wav'
            segment['path'] = filename
            start_frame = int(segment['start'] * wf.getframerate())
            end_frame = int(segment['end'] * wf.getframerate())
            length = end_frame - start_frame
            segment_wave = wave.open(filename, 'wb')
            segment_wave.setframerate(wf.getframerate())
            segment_wave.setnframes(length)
            segment_wave.setsampwidth(wf.getsampwidth())
            segment_wave.setnchannels(wf.getnchannels())
            wf.setpos(start_frame)
            data = wf.readframes(length)
            segment_wave.writeframes(data)
            segment_wave.close()

            if args.speech_enhancement_on:
                logging.info(f"Applying speech enhancement to {filename}")
                from speechbrain.pretrained import SepformerSeparation as separator
                from speechbrain.dataio.dataio import write_audio
                import torchaudio

                enhancer_model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='../models/sepformer-whamr-enhancement')
                enhanced_speech = enhancer_model.separate_file(filename)
                torchaudio.save(filename, enhanced_speech[:, :, 0].detach().cpu(), 8000)


        for i, segment in enumerate(segments):
            rec = KaldiRecognizer(model, 16000)
            # rec.SetWords(True)
            logging.info(f"Processing segment {segment}")

            channel_to_use = args.ffmpeg_strategy

            if args.ffmpeg_strategy == 'LEFT':
                channel_to_use = 0
            elif args.ffmpeg_strategy == 'RIGHT':
                channel_to_use = 1
            elif args.ffmpeg_strategy == 'AUTO':
                channel_to_use = determine_dominant_channel(segment['path'])
                logging.info(f"Determined channel {channel_to_use}")
            else:
                assert(False)

            if channel_to_use is None:
                channel_to_use = 'MERGE'

            pipe = prepare_through_ffmpeg(segment['path'], args.ffmpeg_compress, args.ffmpeg_noise_reduction, channel_to_use)

            recognized = ''

            logging.info(f"ASR segment {i + 1}/{len(segments)}")

            while True:
                data = pipe.stdout.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    logging.debug(f"VOSK intermediate result {res}")
                    recognized += res['text'] + ' '

            # add remaining transcription at end of file
            res = json.loads(rec.FinalResult())
            recognized += res['text']
            segment_texts.append(recognized)

        # copy temporary directory to save path
        try:
            from shutil import copytree
            copytree(tmp, f'{audio_path}-segments')
        except Exception as e:
            logging.error("Could not save segmented audio data")
            logging.error(e)


    if args.punctuation_model:
        logging.info("Reconstructing punctuation and casing")
        punctuated_texts = recasepunc.generate_predictions(None, args.punctuation_model, segment_texts)
    else:
        punctuated_texts = segment_texts

    srt_entries = []
    # limit line length
    _LINE_LENGTH = 13
    for i, segment in enumerate(segments):
        words = punctuated_texts[i].split()
        lines = []
        speaker = segment['speaker']
        lines.append(f"[Speaker {speaker}]")
        for j in range(0, len(words), _LINE_LENGTH):
               line = words[j : j + _LINE_LENGTH]
               lines.append(' '.join(line))

        # find start and end times
        start_seconds = segment['start']
        end_seconds = segment['end']

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



def determine_dominant_channel(file_path):
    ''' determine the domainant channel from a wave file. Returns the channel id (int) or None. It only
        returns None if it is no stereo recording.
    '''

    with wave.open(file_path, 'rb') as wf:
        if wf.getnchannels() < 2:
            return None

        # use librosa to compute rms and pick channel with strongest signal
        strongest_rms = -1
        strongest_channel = 0
        sound, sample_rate = librosa.load(file_path, mono=False)
        for i in range(0, wf.getnchannels()):
            S, phase_left = librosa.magphase(librosa.stft(sound[i]))
            rms = librosa.feature.rms(S=S)

            current_mean_rms = np.mean(rms)
            if current_mean_rms > strongest_rms:
                strongest_rms = current_mean_rms
                strongest_channel = i

            logging.debug(f"{i}: rms = {current_mean_rms}")

        return strongest_channel


def prepare_through_ffmpeg(input_path, enable_compression, enable_noise_reduction, channel_to_use):
    ffmpeg_options = ['ffmpeg']
    if args.log_level.upper() != "DEBUG":
        # make sure to still show errors! Otherwise we won't notice bad ffmpeg commands.
        ffmpeg_options.extend(['-loglevel', 'error'])

    ffmpeg_options.extend(['-i', input_path, '-ar', '16000', '-acodec', 'pcm_s16le', '-f', 'wav'])
    filter_options = []
    if channel_to_use == 'MERGE':
        ffmpeg_options.extend(['-ac', '1'])
    elif type(channel_to_use) == int:
        filter_options.append(f"pan=mono|c0=c{channel_to_use}")
    else:
        assert(False)

    if enable_compression:
        filter_options.append("compand=.3|.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2")

    if enable_noise_reduction:
        filter_options.append("afftdn=nr=10:nf=-40")

    if len(filter_options) > 0:
        filter_string = ','.join(filter_options)
        ffmpeg_options.extend(['-af', filter_string])

    ffmpeg_options.append('-')

    logging.debug(f"Executing ffmpeg command {' '.join(ffmpeg_options)}")

    # write to pipe
    process = subprocess.Popen(ffmpeg_options, stdout=subprocess.PIPE)

    return process


def main():
    global args
    args = parser.parse_args()

    # some basic logging settings for vosk
    SetLogLevel(-1)

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logging.basicConfig(level=numeric_level)

    if not os.path.exists(args.language_model):
        print ("Please download the language model from https://alphacephei.com/vosk/models and unpack as {} in the current folder.".format(args.language_model))
        exit (1)

    global model
    model = Model(model_path=args.language_model)

    pool = multiprocessing.Pool(args.threads)
    for out in pool.imap_unordered(transcribe_audio, args.audio):
        pass

if __name__ == '__main__':
    main()

