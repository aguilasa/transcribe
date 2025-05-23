import logging
import os
import re
from pathlib import Path

import colorlog
import torch
import unicodedata
import whisper
from transformers import pipeline
from yt_dlp import YoutubeDL

# Configuração do colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s]%(reset)s %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))
logger = colorlog.getLogger("transcriber")
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Altere para DEBUG se quiser mais detalhes


class AudioTranscriber:
    def __init__(self, model_size="medium", language=None, download_dir="downloads", output_dir="transcriptions"):
        self.model_size = model_size
        self.language = language
        self.model = None
        self.download_dir = download_dir
        self.output_dir = output_dir

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            logger.info(f"Created download directory: {self.download_dir}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        self._load_model()

    def _load_model(self):
        logger.info(f"Loading Whisper model '{self.model_size}' on CPU...")
        try:
            self.model = whisper.load_model(self.model_size).to("cpu")
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.model = None

    def normalize_filename(self, filename):
        filename = filename.lower()
        filename = filename.replace(" ", "-")
        filename = re.sub(r'[^a-z0-9\-_\.]', '', filename)
        filename = ''.join(c for c in unicodedata.normalize('NFD', filename)
                           if unicodedata.category(c) != 'Mn')
        return filename

    def download_youtube_audio(self, url, format='mp3', quality='192'):
        """
        Downloads audio from a YouTube video using ID for filename and then renames to normalized title.
        If the file already exists, skips download.

        Args:
            url: YouTube video URL
            format: Audio format (mp3, m4a, wav, etc.)
            quality: Audio quality in kbps

        Returns:
            Path to the downloaded audio file or None if download failed
        """
        logger.info(f"Downloading audio from YouTube: {url}")

        # First, get video info to determine the ID and title
        ydl_info_opts = {
            'quiet': True,
            'skip_download': True,
        }
        try:
            with YoutubeDL(ydl_info_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    info = info['entries'][0]
                video_id = info.get('id', 'unknown_id')
                video_title = info.get('title', 'unknown_video')
        except Exception as e:
            logger.error(f"Failed to extract video info: {str(e)}")
            return None

        # Build the expected file path
        temp_file = os.path.join(self.download_dir, f"{video_id}.{format}")
        normalized_title = self.normalize_filename(video_title)
        final_file = os.path.join(self.download_dir, f"{normalized_title}.{format}")

        # Check if the normalized file already exists
        if os.path.exists(final_file):
            logger.info(f"File already exists: {final_file}. Skipping download.")
            return final_file

        # Check if the temp file (by ID) already exists
        if os.path.exists(temp_file):
            logger.info(f"File already exists: {temp_file}. Renaming to normalized title.")
            try:
                os.rename(temp_file, final_file)
                logger.info(f"File renamed from {temp_file} to {final_file}")
            except Exception as e:
                logger.error(f"Error renaming file: {str(e)}")
                final_file = temp_file
            return final_file

        # If not, proceed to download
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': format,
                'preferredquality': quality,
            }],
            'outtmpl': os.path.join(self.download_dir, '%(id)s.%(ext)s'),
            'verbose': True,
            'quiet': False,
            'no_warnings': False,
            'noplaylist': True,
            'extract_flat': False,
            'writethumbnail': False,
            'writesubtitles': False,
            'writedescription': False,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Starting download: {video_title}")
                ydl.download([url])

            # After download, rename to normalized title
            if os.path.exists(temp_file):
                try:
                    os.rename(temp_file, final_file)
                    logger.info(f"File renamed from {temp_file} to {final_file}")
                except Exception as e:
                    logger.error(f"Error renaming file: {str(e)}")
                    final_file = temp_file
                logger.info(f"Download completed: {final_file}")
                return final_file
            else:
                logger.error(f"Downloaded file not found: {temp_file}")
                return None

        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            logger.debug("Error details:", exc_info=True)
            return None

    def get_youtube_title(self, url):
        """
        Gets the title of a YouTube video without downloading it.

        Args:
            url: YouTube video URL

        Returns:
            The video title or None if it fails.
        """
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,  # Don't extract all information
            'skip_download': True,  # Don't download
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if 'entries' in info:
                    info = info['entries'][0]
                return info.get('title', None)
        except Exception as e:
            logger.error(f"Failed to extract video title: {str(e)}")
            return None

    def transcribe_youtube(self, url, audio_format='mp3', audio_quality='192'):
        audio_file = self.download_youtube_audio(url, format=audio_format, quality=audio_quality)
        if not audio_file:
            logger.error("Could not download YouTube audio. Transcription aborted.")
            return [], None
        without_timestamps = self.transcribe(audio_file)
        base_filename = Path(audio_file).stem
        return without_timestamps, base_filename

    def transcribe(self, file_path):
        if self.model is None:
            logger.error("Model is not loaded. Cannot transcribe.")
            return []
        logger.info(f"Starting transcription for: {file_path}")
        try:
            result = self.model.transcribe(file_path, language=self.language, fp16=False)
            text = result["text"]
            formatted_text = self.format_text(text)
            paragraphs = self.split_into_paragraphs(formatted_text)
            with_timestamps = []
            without_timestamps = []
            if "segments" in result:
                for segment in result["segments"]:
                    time = segment.get("start", 0)
                    minutes = int(time // 60)
                    seconds = int(time % 60)
                    segment_text = self.format_text(segment["text"])
                    with_timestamps.append(f"[{minutes:02d}:{seconds:02d}] {segment_text}")
                    without_timestamps.append(segment_text)
            else:
                with_timestamps = paragraphs
                without_timestamps = paragraphs
            file_name = Path(file_path).stem
            out_path = os.path.join(self.output_dir, f"{file_name}_transcription.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription of file: {file_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n\n".join(with_timestamps))
                f.write("\n\n" + "=" * 50 + "\n")
                f.write(f"End of transcription - Model used: {self.model_size}")
            logger.info(f"Transcription saved to: {out_path}")
            return without_timestamps  # Retorna apenas a versão sem tempos
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return []

    @staticmethod
    def format_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        if not text.rstrip().endswith(('.', '!', '?')):
            text = text.rstrip() + '.'
        text = re.sub(r'([.!?])\s*([A-ZÀ-Ú])', r'\1\n\n\2', text)
        text = '. '.join(s.capitalize() for s in text.split('. '))
        return text

    @staticmethod
    def split_into_paragraphs(text, max_words=30):
        words = text.split()
        paragraphs = []
        current_paragraph = []
        for word in words:
            current_paragraph.append(word)
            if (word.endswith(('.', '!', '?')) and len(current_paragraph) >= max_words):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        return paragraphs

    @staticmethod
    def display_transcription(final_text):
        print("\nTranscription:")
        print("=" * 50)
        for paragraph in final_text:
            print(paragraph)
            print()
        print("=" * 50)


class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        logger.info(f"Loading summarization model '{model_name}' on CPU...")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device="cpu"
            )
            logger.info("Summarization model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {str(e)}")
            self.summarizer = None

    def summarize(self, text, max_tokens=130, min_tokens=30):
        if self.summarizer is None:
            logger.error("Summarizer is not loaded. Cannot summarize.")
            return "Unable to generate summary."
        try:
            summary = self.summarizer(
                text,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary."


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cleared.")

    language = None
    output_dir = "transcriptions"
    transcriber = AudioTranscriber(model_size="medium", language=language, output_dir=output_dir)

    youtube_url = "https://www.youtube.com/watch?v=2X2SO3Y-af8"

    # 1. Obtenha o título do vídeo para gerar o nome base dos arquivos
    video_title = transcriber.get_youtube_title(youtube_url)
    if not video_title:
        logger.error("Could not get video title. Aborting.")
        exit(1)
    base_filename = transcriber.normalize_filename(video_title)

    # 2. Defina os caminhos dos arquivos
    transcription_with_times_path = os.path.join(transcriber.output_dir, f"{base_filename}_transcription.txt")
    transcription_without_times_path = os.path.join(transcriber.output_dir,
                                                    f"{base_filename}_transcription_without_times.txt")
    summary_path = os.path.join(transcriber.output_dir, f"{base_filename}_summary.txt")

    # 3. Verifique se a transcrição sem tempos já existe
    if os.path.exists(transcription_without_times_path):
        logger.info(f"Transcription file already exists: {transcription_without_times_path}. Skipping transcription.")
        with open(transcription_without_times_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            without_timestamps = [line.strip() for line in lines if
                                  line.strip() and not line.startswith("Transcription")]
    else:
        # 4. Se não existe, faça a transcrição e salve a versão sem tempos
        without_timestamps, base_filename = transcriber.transcribe_youtube(youtube_url)
        if without_timestamps:
            with open(transcription_without_times_path, "w", encoding="utf-8") as f:
                f.write("Transcription without timestamps\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n\n".join(without_timestamps))
            logger.info(f"Transcription without timestamps saved to: {transcription_without_times_path}")
        else:
            logger.error("No transcription was generated.")
            exit(1)

    # 5. Sumarização usando a versão sem tempos
    complete_text = " ".join(without_timestamps)
    logger.info(f"Text length for summarization: {len(complete_text)}")


    # Função para escolher o modelo de sumarização
    def get_summarizer_for_text(text):
        if len(text) < 3500:
            model_name = "facebook/bart-large-cnn"
        else:
            model_name = "facebook/led-base-16384"
        logger.info(f"Using summarization model: {model_name}")
        return TextSummarizer(model_name=model_name)


    summarizer = get_summarizer_for_text(complete_text)
    min_chars = 100
    max_chars = 15000  # LED suporta textos bem maiores

    if not complete_text or len(complete_text.strip()) < min_chars:
        summary = "Texto muito curto para gerar um resumo."
        logger.warning("Texto muito curto para gerar um resumo.")
    else:
        if len(complete_text) > max_chars:
            logger.info(f"Texto muito grande para sumarização, cortando para {max_chars} caracteres.")
            complete_text = complete_text[:max_chars]
        try:
            summary = summarizer.summarize(complete_text)
        except Exception as e:
            logger.error(f"Exception during summary generation: {str(e)}")
            summary = "Unable to generate summary due to an error."

    # 6. Salva o sumário
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Summary of transcription\n")
        f.write("=" * 50 + "\n\n")
        f.write(summary)
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("Process completed successfully!")
