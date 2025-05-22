import whisper
import torch
from pathlib import Path
import re
from transformers import pipeline
from yt_dlp import YoutubeDL
import os

class AudioTranscriber:
    def __init__(self, model_size="medium", language=None, download_dir="downloads", output_dir="transcriptions"):
        """
        Initializes the transcriber with the specified Whisper model size and language.
        If language is None, Whisper will auto-detect.

        Args:
            model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code ('en', 'pt', etc.) or None for auto-detection
            download_dir: Directory to save downloaded audio files
            output_dir: Directory to save transcription files
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self.download_dir = download_dir
        self.output_dir = output_dir

        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._load_model()

    def _load_model(self):
        """Loads the Whisper model on CPU."""
        print(f"[INFO] Loading Whisper model '{self.model_size}' on CPU...")
        try:
            self.model = whisper.load_model(self.model_size).to("cpu")
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            self.model = None

    def download_youtube_audio(self, url, format='mp3', quality='192'):
        """
        Downloads audio from a YouTube video.

        Args:
            url: YouTube video URL
            format: Audio format (mp3, m4a, wav, etc.)
            quality: Audio quality in kbps

        Returns:
            Path to the downloaded audio file or None if download failed
        """
        print(f"[INFO] Downloading audio from YouTube: {url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': format,
                'preferredquality': quality,
            }],
            'outtmpl': os.path.join(self.download_dir, '%(title)s.%(ext)s'),
            'verbose': False,
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
                video_title = info['title']
                duration = info.get('duration', 0)  # Duration in seconds

                print(f"[INFO] Starting download: {video_title}")
                print(f"[INFO] Duration: {duration//60}:{duration%60:02d}")

                # Perform download
                ydl.download([url])

                # The file will have the format extension, not the original
                output_file = os.path.join(self.download_dir, f"{video_title}.{format}")
                print(f"[INFO] Download completed: {output_file}")
                return output_file

        except Exception as e:
            print(f"[ERROR] Failed to download video: {str(e)}")
            return None

    def transcribe_youtube(self, url, audio_format='mp3', audio_quality='192'):
        """
        Downloads and transcribes audio from a YouTube video.

        Args:
            url: YouTube video URL
            audio_format: Audio format to download
            audio_quality: Audio quality in kbps

        Returns:
            List of transcribed segments or empty list if failed
        """
        audio_file = self.download_youtube_audio(url, format=audio_format, quality=audio_quality)
        if not audio_file:
            print("[ERROR] Could not download YouTube audio. Transcription aborted.")
            return []

        return self.transcribe(audio_file)

    def transcribe(self, file_path):
        """
        Transcribes the given audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            List of formatted transcription segments
        """
        if self.model is None:
            print("[ERROR] Model is not loaded. Cannot transcribe.")
            return []

        print(f"[INFO] Starting transcription for: {file_path}")
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

            print(f"[INFO] Transcription saved to: {out_path}")
            return final_text

        except Exception as e:
            print(f"[ERROR] Error during transcription: {str(e)}")
            return []

    @staticmethod
    def format_text(text):
        """
        Formats text for better readability.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        if not text.rstrip().endswith(('.', '!', '?')):
            text = text.rstrip() + '.'
        text = re.sub(r'([.!?])\s*([A-ZÀ-Ú])', r'\1\n\n\2', text)
        text = '. '.join(s.capitalize() for s in text.split('. '))
        return text

    @staticmethod
    def split_into_paragraphs(text, max_words=30):
        """
        Splits text into paragraphs based on a maximum number of words.
        """
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
        """
        Displays the transcription in a formatted way in the console.
        """
        print("\nTranscription:")
        print("=" * 50)
        for paragraph in final_text:
            print(paragraph)
            print()
        print("=" * 50)

class TextSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the summarizer with the specified Hugging Face model.
        """
        print(f"[INFO] Loading summarization model '{model_name}' on CPU...")
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device="cpu"
            )
            print("[INFO] Summarization model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load summarization model: {str(e)}")
            self.summarizer = None

    def summarize(self, text, max_tokens=130, min_tokens=30):
        """
        Summarizes the given text.
        """
        if self.summarizer is None:
            print("[ERROR] Summarizer is not loaded. Cannot summarize.")
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
            print(f"[ERROR] Error generating summary: {str(e)}")
            return "Unable to generate summary."

if __name__ == "__main__":
    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[INFO] CUDA memory cleared.")

    language = None  # Set to "en" for English, "pt" for Portuguese, or None for auto-detect
    output_dir = "transcriptions"  # Set the desired output directory for transcriptions
    transcriber = AudioTranscriber(model_size="medium", language=language, output_dir=output_dir)

    # Example usage with local file
    # audio_path = "/path/to/your/audio.mp3"  # Replace with your local audio file
    # transcription = transcriber.transcribe(audio_path)

    # Example usage with YouTube URL
    youtube_url = "https://www.youtube.com/live/p95VYdLmbHY"
    transcription = transcriber.transcribe_youtube(youtube_url)

    if transcription:
        transcriber.display_transcription(transcription)

        # Remove timestamps for summarization
        complete_text = " ".join([re.sub(r'\[\d{2}:\d{2}\]\s*', '', t) for t in transcription])

        print("\n[INFO] Generating summary...")
        summarizer = TextSummarizer()
        summary = summarizer.summarize(complete_text)

        print("\nSummary of transcribed text:")
        print("=" * 50)
        print(summary)
        print("=" * 50)

        # Get the filename from the last transcription
        file_name = Path(transcriber.download_dir).joinpath(Path(youtube_url).stem)
        summary_path = f"{file_name}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Summary of transcription\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)

        print(f"\n[INFO] Summary saved to: {summary_path}")
        print("[INFO] Process completed successfully!")
    else:
        print("[ERROR] No transcription was generated.")