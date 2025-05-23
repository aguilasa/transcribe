import os
import re
import unicodedata
import logging
from pathlib import Path

import whisper
import torch
from transformers import pipeline
from yt_dlp import YoutubeDL
import colorlog

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

        Args:
            url: YouTube video URL
            format: Audio format (mp3, m4a, wav, etc.)
            quality: Audio quality in kbps

        Returns:
            Path to the downloaded audio file or None if download failed
        """
        logger.info(f"Downloading audio from YouTube: {url}")

        # Use video ID for initial download to avoid any character issues
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
                # First extract video information
                info = ydl.extract_info(url, download=False)

                # Check if it's a playlist and get the first item if it is
                if 'entries' in info:
                    logger.warning("URL contains multiple videos. Using the first one.")
                    info = info['entries'][0]

                video_id = info.get('id', 'unknown_id')
                video_title = info.get('title', 'unknown_video')
                duration = info.get('duration', 0)  # Duration in seconds

                logger.info(f"Starting download: {video_title}")
                logger.info(f"Video ID: {video_id}")
                logger.info(f"Duration: {duration // 60}:{duration % 60:02d}")

                # Perform the download using the ID
                ydl.download([url])

                # Path to the downloaded file (using ID)
                temp_file = os.path.join(self.download_dir, f"{video_id}.{format}")

                # Check if the file exists
                if not os.path.exists(temp_file):
                    logger.warning(f"File with ID {video_id}.{format} not found. Looking for alternatives...")

                    # Look for any recently created file with the correct format
                    import time
                    current_time = time.time()
                    recent_files = [f for f in os.listdir(self.download_dir)
                                    if f.endswith(f'.{format}') and
                                    os.path.getmtime(os.path.join(self.download_dir, f)) >
                                    (current_time - 60)]  # Files created in the last minute

                    if recent_files:
                        temp_file = os.path.join(self.download_dir, recent_files[0])
                        logger.info(f"Found alternative file: {temp_file}")
                    else:
                        logger.error(f"Could not find downloaded file")
                        return None

                # Create normalized title for the final filename
                normalized_title = self.normalize_filename(video_title)
                final_file = os.path.join(self.download_dir, f"{normalized_title}.{format}")

                # Rename the file
                try:
                    os.rename(temp_file, final_file)
                    logger.info(f"File renamed from {temp_file} to {final_file}")
                except Exception as e:
                    logger.error(f"Error renaming file: {str(e)}")
                    # If rename fails, use the original file
                    final_file = temp_file

                logger.info(f"Download completed: {final_file}")
                return final_file

        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            logger.debug("Error details:", exc_info=True)
            return None

    def transcribe_youtube(self, url, audio_format='mp3', audio_quality='192'):
        audio_file = self.download_youtube_audio(url, format=audio_format, quality=audio_quality)
        if not audio_file:
            logger.error("Could not download YouTube audio. Transcription aborted.")
            return []
        return self.transcribe(audio_file)

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
            final_text = []
            if "segments" in result:
                for segment in result["segments"]:
                    time = segment.get("start", 0)
                    minutes = int(time // 60)
                    seconds = int(time % 60)
                    segment_text = self.format_text(segment["text"])
                    final_text.append(f"[{minutes:02d}:{seconds:02d}] {segment_text}")
            else:
                final_text = paragraphs
            file_name = Path(file_path).stem
            out_path = os.path.join(self.output_dir, f"{file_name}_transcription.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription of file: {file_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write("\n\n".join(final_text))
                f.write("\n\n" + "=" * 50 + "\n")
                f.write(f"End of transcription - Model used: {self.model_size}")
            logger.info(f"Transcription saved to: {out_path}")
            return final_text
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

    # Exemplo com arquivo local:
    # audio_path = "/path/to/your/audio.mp3"
    # transcription = transcriber.transcribe(audio_path)

    # Exemplo com YouTube:
    youtube_url = "https://www.youtube.com/watch?v=2X2SO3Y-af8"
    transcription = transcriber.transcribe_youtube(youtube_url)

    if transcription:
        transcriber.display_transcription(transcription)
        complete_text = " ".join([re.sub(r'\[\d{2}:\d{2}\]\s*', '', t) for t in transcription])
        logger.info("Generating summary...")
        summarizer = TextSummarizer()
        summary = summarizer.summarize(complete_text)
        print("\nSummary of transcribed text:")
        print("=" * 50)
        print(summary)
        print("=" * 50)
        # Normaliza o nome do arquivo para o resumo
        file_name = Path(transcriber.download_dir).joinpath(Path(youtube_url).stem)
        summary_path = f"{file_name}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Summary of transcription\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("Process completed successfully!")
    else:
        logger.error("No transcription was generated.")
