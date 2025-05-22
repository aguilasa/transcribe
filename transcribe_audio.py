import whisper
import torch
from pathlib import Path
import re
from transformers import pipeline

def summarize_text(text, max_tokens=130, min_tokens=30):
    """
    Summarizes text using a pre-trained Hugging Face model on CPU.
    """
    # Load summarization pipeline forcing CPU
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device="cpu"  # Force CPU usage
    )

    try:
        # Generate summary
        summary = summarizer(
            text,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Unable to generate summary."

def transcribe_audio(file_path, model_size="medium", language=None):
    """
    Transcribes audio using CPU
    """
    try:
        print(f"Loading {model_size} model on CPU...")
        # Force CPU usage
        model = whisper.load_model(model_size).to("cpu")

        print("Starting transcription...")
        # If language is not specified, Whisper will auto-detect it
        result = model.transcribe(file_path, language=language, fp16=False)

        # Format text
        text = result["text"]
        formatted_text = format_text(text)

        # Split into paragraphs
        paragraphs = split_into_paragraphs(formatted_text)

        # Prepare final text with timestamps
        final_text = []
        if "segments" in result:
            for segment in result["segments"]:
                time = segment.get("start", 0)
                minutes = int(time // 60)
                seconds = int(time % 60)
                segment_text = format_text(segment["text"])
                final_text.append(f"[{minutes:02d}:{seconds:02d}] {segment_text}")
        else:
            final_text = paragraphs

        # Save transcription
        file_name = Path(file_path).stem
        with open(f"{file_name}_transcription.txt", "w", encoding="utf-8") as f:
            f.write(f"Transcription of file: {file_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n\n".join(final_text))
            f.write("\n\n" + "=" * 50 + "\n")
            f.write(f"End of transcription - Model used: {model_size}")

        print(f"\nTranscription saved to: {file_name}_transcription.txt")
        return final_text

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return []

def format_text(text):
    """
    Formats text for better readability
    """
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Add punctuation at the end of sentences if needed
    if not text.rstrip().endswith(('.', '!', '?')):
        text = text.rstrip() + '.'

    # Fix spaces after punctuation
    text = re.sub(r'([.!?])\s*([A-ZÀ-Ú])', r'\1\n\n\2', text)

    # Ensure sentences start with capital letter
    text = '. '.join(s.capitalize() for s in text.split('. '))

    return text

def split_into_paragraphs(text, max_words=30):
    """
    Splits text into paragraphs based on maximum number of words
    """
    words = text.split()
    paragraphs = []
    current_paragraph = []

    for word in words:
        current_paragraph.append(word)

        # Check if it's the end of a sentence and if word limit is reached
        if (word.endswith(('.', '!', '?')) and
            len(current_paragraph) >= max_words):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    # Add the last paragraph if any
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return paragraphs

def display_transcription(final_text):
    """
    Displays the transcription in a formatted way in the console
    """
    print("\nTranscription:")
    print("=" * 50)
    for paragraph in final_text:
        print(paragraph)
        print()
    print("=" * 50)

# Example usage
if __name__ == "__main__":
    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA memory cleared")

    audio_path = "/home/ingmar/PycharmProjects/transcribe/downloads/action_plan.mp3"

    # Detect language or specify it manually
    # language = "en"  # For English
    # language = "pt"  # For Portuguese
    language = None    # For auto-detection

    print(f"Processing audio file: {audio_path}")
    print(f"Language setting: {'Auto-detect' if language is None else language}")

    try:
        # Transcribe audio
        print(f"Starting transcription with {model_size} model...")
        final_text = transcribe_audio(audio_path, model_size="medium", language=language)

        if final_text:
            # Display transcription
            display_transcription(final_text)

            # Prepare text for summary (remove timestamps if they exist)
            complete_text = " ".join([re.sub(r'\[\d{2}:\d{2}\]\s*', '', t) for t in final_text])

            print("\nGenerating summary...")
            summary = summarize_text(complete_text)

            print("\nSummary of transcribed text:")
            print("=" * 50)
            print(summary)
            print("=" * 50)

            # Save summary
            file_name = Path(audio_path).stem
            with open(f"{file_name}_summary.txt", "w", encoding="utf-8") as f:
                f.write("Summary of transcription\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary)

            print(f"\nSummary saved to: {file_name}_summary.txt")
            print("Process completed successfully!")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("Please check the file path and ensure all dependencies are installed.")