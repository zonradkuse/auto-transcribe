# auto-transcribe

auto-transcribe is a small project built on top of vosk-api to automatically transcribe interviews.

auto-transcribe produces srt files. srt files are human-readable and compatible with many tools which work with audio/video.

I needed a tool that does not require uploading the interview data. Strict data protection guidelines in social sciences usually do now allow uploading participant data to third-party servers if they are not fully anonymized.
For this reason, interviews in social sciences are still commonly transcribed manually. This process is extremely expensive and can at least be supported by automatic transcription.

## Features

- Transcription of audio data in many languages
- Speaker diarization (works okayish but **will** need manual processing)
- Multiprocessing

## Installation

First install dependencies using poetry

```bash
poetry install
```

Then download the models you would like to use. Checkout the models folder for more info.

## Usage

```
usage: transcribe.py [-h] [--language-model LANGUAGE_MODEL] [--punctuation-model PUNCTUATION_MODEL] [--speakers SPEAKERS] [--speech-enhancement-on] [--store-segment-audio STORE_SEGMENT_AUDIO] [--log-level LOG_LEVEL] [--threads THREADS] audio [audio ...]

Transcribe interviews.

positional arguments:
  audio                 The audio files to transcribe with the given settings

optional arguments:
  -h, --help            show this help message and exit
  --language-model LANGUAGE_MODEL
                        The path to the vosk language model
  --punctuation-model PUNCTUATION_MODEL
                        The path to the punctuation model
  --speakers SPEAKERS   Number of speakers during the interview
  --speech-enhancement-on
                        Use speechbrain's sppech enhancement. It usually does NOT enhance ASR accuracy.
  --store-segment-audio STORE_SEGMENT_AUDIO
                        Path to store segmented audio files to.
  --log-level LOG_LEVEL
                        Set the loglevel
  --threads THREADS     Set the number of threads allowed to use for processing multiple audio files
```

### Conversion using ffmpeg
Conversion is now optionla. However, you can convert your interview files to 16 bit 16kHz PCM mono wave

```bash
ffmpeg -i in.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
```

Occasionally, minor preprocessing (denoising and compressing) improves recognition

```bash
ffmpeg -i in.wav -af  "highpass=f=60,lowpass=f=10000,acompressor=threshold=-20dB:ratio=15:attack=0.01:release=0.01,volume=+15dB,afftdn=nf=-20" -acodec pcm_s16le -ac 1 -ar 16000 out.wav
```

Finally, start recognition.

```bash
poetry run python src/transcribe.py --language-model models/vosk/vosk-model-de-0.21 --punctuation-model models/repunc/vosk-recasepunc-de-0.21/checkpoint in.wav
```

You can adjust the number of expected speakers by adding `--speakers <n>` to the call. The default is 2.

## Limitations

- Make sure that the audio quality is really good. Poor audio quality directly influences the transcipt's accuracy. Poor quality audio **will** result in non-sense transcripts. Pre-processing of the audio files is certainly possible (and occasionally improves recognition) but needs manual settings.
- The transcript accuracy depends a lot on the language model. For the current German vosk language model the word-error-rate on their podcast testing data is about 20%. Expect to edit the result!


