
import os
import time
import tempfile

from pathlib import Path
from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

TRANSCRIPTION_MODEL = os.getenv(
    "GEMINI_TRANSCRIPTION_MODEL",
    "gemini-3.5-transcribe"
)

VIDEO_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 120000}
)


SUPPORTED_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm"
}


def transcribe_audio(remote_file, mime_type):
    """Transcribe an uploaded audio recording."""

    interaction = client.interactions.create(
        model=TRANSCRIPTION_MODEL,
        input=[
            {
                "type": "audio",
                "uri": remote_file.uri,
                "mime_type": remote_file.mime_type or mime_type
            }
        ]
    )

    return interaction.output_text


def wait_for_video_processing(remote_file):
    """Wait until Gemini finishes processing an uploaded video."""

    max_attempts = 120

    for attempt in range(max_attempts):

        state = getattr(remote_file, "state", None)

        state_name = getattr(
            state,
            "name",
            str(state)
        ).upper()

        if state_name.endswith("ACTIVE"):
            return remote_file

        if state_name.endswith("FAILED"):
            raise RuntimeError(
                "Gemini failed to process the video. "
                "Try another MP4 file."
            )

        if state_name.endswith("PROCESSING") or state_name.endswith("PENDING"):
            time.sleep(3)

            remote_file = client.files.get(
                name=remote_file.name
            )
            continue

        # Handle an unavailable or unexpected processing state.
        time.sleep(3)

        remote_file = client.files.get(
            name=remote_file.name
        )

    raise TimeoutError(
        "Video processing took too long. "
        "Try a shorter video or upload it again."
    )


def transcribe_video(remote_file, mime_type):
    """Extract spoken debate content from an uploaded video."""

    transcription_prompt = """
You are a professional debate transcription assistant.

Transcribe all audible speech in the uploaded debate video.

Follow these instructions carefully:

1. Preserve the original spoken language.
2. Support both English and Kiswahili.
3. Do not translate the speech.
4. Do not summarize or paraphrase what the speakers say.
5. Separate speakers where their voices can reasonably be distinguished.
6. Use labels such as Speaker 1, Speaker 2, and Speaker 3.
7. Preserve the meaning and order of each statement.
8. Include direct rebuttals and interruptions when audible.
9. Do not invent words, arguments, names, or statements.
10. Mark unclear speech as [Unclear] rather than guessing.
11. Return only the transcript.

The purpose is to provide an accurate transcript for a debate
rebuttal prediction system.
"""

    interaction = client.interactions.create(
        model=VIDEO_MODEL,
        input=[
            {
                "type": "video",
                "uri": remote_file.uri,
                "mime_type": remote_file.mime_type or mime_type
            },
            {
                "type": "text",
                "text": transcription_prompt
            }
        ]
    )

    return interaction.output_text


def transcribe_media(uploaded_file):
    """
    Transcribe an uploaded audio or video debate recording.
    Returns the extracted spoken content as text.
    """

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            "Unsupported audio or video format."
        )

    mime_type = SUPPORTED_MIME_TYPES[extension]

    file_data = uploaded_file.getvalue()
    file_size = len(file_data)

    if file_size > 100 * 1024 * 1024:
        raise ValueError(
            "File is too large. Please upload a file below 100 MB."
        )

    if file_size == 0:
        raise ValueError(
            "The uploaded file is empty."
        )

    temp_path = None

    try:
        # Save the uploaded recording temporarily.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(file_data)
            temp_path = temp_file.name

        # Upload the recording to Gemini.
        remote_file = client.files.upload(
            file=temp_path
        )

        if extension in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:

            transcript = transcribe_audio(
                remote_file,
                mime_type
            )

        else:

            # Wait for Gemini to finish processing the video.
            remote_file = wait_for_video_processing(
                remote_file
            )

            # Extract the spoken debate content.
            transcript = transcribe_video(
                remote_file,
                mime_type
            )

        if not transcript or not transcript.strip():
            raise ValueError(
                "No speech could be extracted from this recording."
            )

        return transcript.strip()

    finally:
        # Remove the temporary local file.
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
