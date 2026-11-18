
import json
import streamlit as st

from modules.rebuttal_engine import generate_rebuttals
from modules.media_transcriber import transcribe_media
from modules.debate_analyzer import analyze_debate


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Speak - Debate Rebuttal Predictor",
    page_icon="🎙️",
    layout="wide"
)


# APPLICATION HEADER
st.title("🎙️ Speak")
st.subheader("AI-Powered Debate Rebuttal Predictor")

st.write(
    "Analyze debate motions, opponent arguments and recorded "
    "debates to identify speakers, extract arguments and predict "
    "likely rebuttals."
)

st.caption(
    "Supports English and Kiswahili | Powered by Google Gemini"
)

st.divider()


# INPUT SECTION
st.header("1. Enter Debate Information")

debate_input = st.text_area(
    "Debate motion or opponent's argument",
    placeholder=(
        "Example: Is education really an equalizer "
        "if schools are resourced differently?"
    ),
    height=150
)

st.markdown("### Or upload a debate recording")

uploaded_media = st.file_uploader(
    "Upload audio or video",
    type=[
        "mp3",
        "wav",
        "m4a",
        "ogg",
        "flac",
        "mp4",
        "mov",
        "webm"
    ],
    help="Maximum file size: 100 MB."
)

if uploaded_media:
    st.success(
        f"Uploaded successfully: {uploaded_media.name}"
    )

    st.caption(
        "Your recording will be sent to Google Gemini for processing."
    )


col1, col2 = st.columns(2)

with col1:
    debate_side = st.selectbox(
        "Your debate side",
        [
            "Proposition",
            "Opposition",
            "Not specified"
        ]
    )

with col2:
    output_language = st.selectbox(
        "Output language",
        [
            "Auto-detect",
            "English",
            "Kiswahili"
        ]
    )


# GENERATE ANALYSIS AND PREDICTIONS
st.divider()

st.header("2. Analyze Debate")

if st.button(
    "Analyze Debate",
    type="primary",
    use_container_width=True
):

    if not debate_input.strip() and not uploaded_media:
        st.warning(
            "Please enter debate information or upload a recording."
        )

    else:

        # Clear previous results to avoid displaying stale information.
        st.session_state.pop("transcript", None)
        st.session_state.pop("debate_analysis", None)
        st.session_state.pop("prediction_result", None)

        try:

            final_debate_input = debate_input.strip()
            transcript = ""

            # STEP 1: TRANSCRIBE AUDIO OR VIDEO
            if uploaded_media:

                with st.spinner(
                    "Speak is transcribing your recording..."
                ):

                    transcript = transcribe_media(
                        uploaded_media
                    )

                st.session_state["transcript"] = transcript

                st.success(
                    "Transcription completed successfully!"
                )

                if final_debate_input:

                    final_debate_input = (
                        f"Recorded debate transcript:\n"
                        f"{transcript}\n\n"
                        f"Additional debate information:\n"
                        f"{final_debate_input}"
                    )

                else:

                    final_debate_input = (
                        f"Recorded debate transcript:\n{transcript}"
                    )

        except Exception as error:

            st.error(
                "Speak could not transcribe the recording. "
                "Please check the file and try again."
            )

            st.exception(error)

            final_debate_input = ""

        # Continue only if input or transcription is available.
        if final_debate_input:

            # STEP 2: IDENTIFY SPEAKERS AND EXTRACT ARGUMENTS
            try:

                with st.spinner(
                    "Speak is identifying speakers and analyzing arguments..."
                ):

                    analysis = analyze_debate(
                        final_debate_input
                    )

                st.session_state["debate_analysis"] = analysis

                st.success(
                    "Speaker identification and argument analysis completed!"
                )

            except Exception as error:

                st.error(
                    "Speaker identification and argument analysis failed."
                )

                st.exception(error)

            # STEP 3: GENERATE PREDICTED REBUTTALS
            try:

                with st.spinner(
                    "Speak is predicting likely rebuttals..."
                ):

                    result = generate_rebuttals(
                        debate_input=final_debate_input,
                        debate_side=debate_side,
                        output_language=output_language
                    )

                st.session_state["prediction_result"] = result

                st.success(
                    "Predictions generated successfully!"
                )

            except Exception as error:

                st.error(
                    "Speak could not generate rebuttal predictions."
                )

                st.exception(error)


# DISPLAY TRANSCRIPT
transcript = st.session_state.get("transcript")

if transcript:

    st.divider()

    st.header("3. Extracted Debate Transcript")

    with st.expander("View full transcript", expanded=False):

        st.text_area(
            "Transcribed speech",
            value=transcript,
            height=250,
            disabled=True
        )

        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name="speak_transcript.txt",
            mime="text/plain"
        )


# DISPLAY DEBATE ANALYSIS
analysis = st.session_state.get("debate_analysis")

if analysis:

    st.divider()

    st.header("4. Speaker Identification and Argument Analysis")

    st.caption(
        "Speaker labels and debate sides are AI inferences from "
        "the available text. They are not verified voice identities."
    )

    # DEBATE MOTION AND SUMMARY
    st.subheader("Debate Overview")

    st.markdown(
        f"**Debate motion:** "
        f"{analysis.get('debate_motion', 'Not specified')}"
    )

    st.write(
        analysis.get(
            "summary",
            "No summary available."
        )
    )

    # SPEAKER IDENTIFICATION
    st.divider()

    st.subheader("Identified Speakers")

    speakers = analysis.get("speakers", [])

    if speakers:

        for speaker in speakers:

            with st.container(border=True):

                st.markdown(
                    f"**{speaker.get('speaker_id', 'Unknown')} — "
                    f"{speaker.get('speaker_label', 'Unknown speaker')}**"
                )

                st.write(
                    f"**Debate side:** "
                    f"{speaker.get('side', 'Unknown')}"
                )

                st.write(
                    speaker.get(
                        "position_summary",
                        "No position summary available."
                    )
                )

    else:

        st.info("No speakers were identified.")


    # ARGUMENT EXTRACTION
    st.divider()

    st.subheader("Extracted Arguments")

    arguments = analysis.get("arguments", [])

    if arguments:

        for argument in arguments:

            with st.container(border=True):

                st.markdown(
                    f"### {argument.get('argument_id', 'Unknown argument')}"
                )

                st.write(
                    f"**Speaker:** "
                    f"{argument.get('speaker_id', 'Unknown')}"
                )

                st.write(
                    f"**Side:** "
                    f"{argument.get('side', 'Unknown')}"
                )

                st.write(
                    f"**Argument type:** "
                    f"{argument.get('argument_type', 'Not specified')}"
                )

                st.markdown("**Claim**")

                st.write(
                    argument.get("claim", "")
                )

                st.markdown("**Reasoning**")

                st.write(
                    argument.get("reasoning", "")
                )

                st.markdown("**Evidence**")

                st.write(
                    argument.get(
                        "evidence",
                        "Not provided"
                    )
                )

                responds_to = argument.get("responds_to")

                if responds_to and responds_to != "None":

                    st.write(
                        f"**Responds to:** {responds_to}"
                    )

    else:

        st.info("No arguments were extracted.")


    # DIRECT REBUTTAL IDENTIFICATION
    st.divider()

    st.subheader("Direct Rebuttals Identified")

    direct_rebuttals = analysis.get("rebuttals", [])

    if direct_rebuttals:

        for rebuttal in direct_rebuttals:

            with st.container(border=True):

                st.markdown(
                    f"### {rebuttal.get('rebuttal_id', 'Rebuttal')}"
                )

                st.write(
                    f"**Speaker:** "
                    f"{rebuttal.get('speaker_id', 'Unknown')}"
                )

                st.write(
                    f"**Side:** "
                    f"{rebuttal.get('side', 'Unknown')}"
                )

                st.write(
                    f"**Target argument:** "
                    f"{rebuttal.get('target_argument_id', 'Not identified')}"
                )

                st.markdown("**Rebuttal**")

                st.write(
                    rebuttal.get("rebuttal", "")
                )

                st.markdown("**Explanation**")

                st.write(
                    rebuttal.get("explanation", "")
                )

    else:

        st.info(
            "No direct rebuttals were identified in the debate."
        )


    # DOWNLOAD DEBATE ANALYSIS
    st.divider()

    analysis_json = json.dumps(
        analysis,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Debate Analysis (JSON)",
        data=analysis_json,
        file_name="speak_debate_analysis.json",
        mime="application/json",
        use_container_width=True
    )


# DISPLAY PREDICTIONS
result = st.session_state.get("prediction_result")

if result:

    st.divider()

    st.header("5. Predicted Rebuttals")

    st.write(
        f"**Debate topic:** "
        f"{result.get('topic', 'Not specified')}"
    )

    st.write(
        f"**Detected language:** "
        f"{result.get('language', 'Not specified')}"
    )

    st.caption(
        "These are AI-generated predictions of plausible "
        "opposing arguments, not guaranteed statements."
    )

    rebuttals = result.get("rebuttals", [])

    for index, rebuttal in enumerate(rebuttals, start=1):

        with st.container(border=True):

            st.subheader(f"Prediction {index}")

            confidence = rebuttal.get(
                "confidence",
                "Not specified"
            )

            st.markdown(
                f"**Confidence:** {confidence}"
            )

            st.markdown("**Predicted argument**")

            st.write(
                rebuttal.get("argument", "")
            )

            st.markdown("**Why this argument is likely**")

            st.write(
                rebuttal.get("reasoning", "")
            )

            st.markdown("**Suggested response strategy**")

            st.write(
                rebuttal.get("response_strategy", "")
            )


    # LINGUISTIC PATTERNS
    st.divider()

    st.header("6. Key Linguistic Patterns")

    patterns = result.get(
        "linguistic_patterns",
        {}
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Likely Keywords")

        keywords = patterns.get("keywords", [])

        if keywords:

            for keyword in keywords:
                st.write(f"- {keyword}")

        else:

            st.write("No keywords returned.")

    with col2:

        st.subheader("Rhetorical Strategies")

        strategies = patterns.get(
            "rhetorical_strategies",
            []
        )

        if strategies:

            for strategy in strategies:
                st.write(f"- {strategy}")

        else:

            st.write("No strategies returned.")


    # DOWNLOAD PREDICTIONS
    st.divider()

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Predictions (JSON)",
        data=json_data,
        file_name="speak_predictions.json",
        mime="application/json",
        use_container_width=True
    )


# FOOTER
st.divider()

st.caption(
    "Speak | Debate Rebuttal Predictor | AI Capstone Prototype"
)
