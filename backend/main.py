"""Final integration entry point for the MythLens backend.

Stable functions developed in Colab should be moved into their module folders
and wired together here during integration.
"""

from backend.arabic.normalization import detect_language, normalize_egyptian_arabic
from backend.claims.extraction import extract_claims, generate_medical_query
from backend.ingestion.transcription import process_text_input, process_video_input, summarize_video_transcript, transcribe_video


def main() -> None:
    print("MythLens backend structure is ready.")


# Member 1 API surface matching TEAM_WORKFLOW.md
transcribe_video = transcribe_video
normalize_egyptian_arabic = normalize_egyptian_arabic
detect_language = detect_language
extract_claims = extract_claims
generate_medical_query = generate_medical_query
process_text_input = process_text_input
process_video_input = process_video_input
summarize_video_transcript = summarize_video_transcript


if __name__ == "__main__":
    main()
