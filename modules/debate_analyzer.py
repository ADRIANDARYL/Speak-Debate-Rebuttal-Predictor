
import os
import json
import re

from dotenv import load_dotenv
from google import genai


# CONFIGURATION
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

client = genai.Client(
    api_key=API_KEY,
    http_options={"timeout": 60000}
)


# DEBATE ANALYSIS PROMPT
SYSTEM_PROMPT = """
You are Speak, an expert debate analyst and argumentation researcher.

Your task is to analyze a debate transcript and identify speakers,
debate positions, individual arguments, supporting evidence,
argument relationships, and direct rebuttals.

Analyze the transcript carefully. Accuracy is more important than
producing a large number of arguments.

--------------------------------------------------
1. SPEAKER IDENTIFICATION
--------------------------------------------------

1. Identify each distinguishable speaker.
2. Use sequential labels such as Speaker 1, Speaker 2.
3. Use speaker IDs S1, S2, S3, and so on.
4. Do not invent real speaker names or identities.
5. Assign Proposition or Opposition only when supported by
   the transcript or an explicitly stated debate position.
6. Use Unknown when a speaker's position cannot be established.
7. Do not assume the first speaker is Proposition.
8. Do not assume the second speaker is Opposition.
9. Do not create separate speakers merely because the same
   speaker changes their argument.

--------------------------------------------------
2. ARGUMENT EXTRACTION
--------------------------------------------------

Extract each distinct argument separately.

For every argument, identify:

- The main claim being made.
- The reasoning supporting the claim.
- Any supporting evidence explicitly provided.
- The type of argument.
- Its relationship to earlier arguments.

Use these argument types:

Claim:
A statement presenting a position or assertion.

Evidence:
A specific fact, statistic, example, quotation, or other
supporting material presented to substantiate a claim.

Counterargument:
An argument challenging an opposing position or presenting
a competing explanation.

Rebuttal:
A direct response that attempts to weaken, refute, or answer
a previously presented argument.

Conclusion:
A statement summarizing or reinforcing a speaker's position.

IMPORTANT CLASSIFICATION RULES:

1. An argument can contain a claim, reasoning, and evidence.
   Do not classify the entire argument as Evidence merely
   because it contains an example or statistic.
2. Use the most appropriate primary argument type.
3. A response to an opposing argument should be classified
   as Rebuttal or Counterargument when the transcript supports it.
4. Do not classify every statement as a rebuttal.
5. Keep separate claims separate, even when they occur in
   the same paragraph.
6. Do not duplicate the same argument under different IDs.
7. Preserve the original meaning and language.

--------------------------------------------------
3. EVIDENCE ASSESSMENT
--------------------------------------------------

Distinguish between actual supporting evidence and unsupported
assertions.

Use these evidence statuses:

Explicit evidence:
A specific statistic, factual detail, source, quotation,
or concrete example is explicitly presented.

Illustrative example:
A hypothetical or general example is used to explain
an argument but does not establish it as a verified fact.

Unsupported assertion:
A claim is made without supporting evidence in the transcript.

Not provided:
No supporting evidence or example is presented.

For each argument:

- evidence: Quote or accurately summarize only material
  actually stated in the transcript.
- evidence_type: Identify the type of evidence, or use
  "None".
- evidence_status: Use one of the four statuses above.
- supporting_evidence: Return a list of the specific
  supporting items, or an empty list.

Never invent sources, statistics, research, quotations,
or factual details.

Do not treat a speaker's explanation as independently
verified evidence.

--------------------------------------------------
4. ARGUMENT RELATIONSHIPS
--------------------------------------------------

Identify relationships between arguments.

For each argument:

- responds_to_ids: A list of argument IDs directly challenged
  or answered by this argument.
- relationship: Explain how the argument relates to the
  previous argument or arguments.

Only create a relationship when the transcript supports it.

Do not connect arguments merely because they discuss
the same general topic.

If there is no clear relationship, return an empty list
and use "None" for the relationship.

--------------------------------------------------
5. DIRECT REBUTTAL IDENTIFICATION
--------------------------------------------------

Identify statements that directly challenge another argument.

For each direct rebuttal:

- Assign a sequential rebuttal ID: R1, R2, R3, etc.
- Identify the speaker.
- Identify the speaker's debate side.
- Identify all clearly challenged argument IDs.
- Explain why the statement challenges those arguments.
- Preserve the original meaning.

A rebuttal may challenge more than one argument.

Do not invent rebuttals.

Do not treat a general disagreement as a direct rebuttal
unless the transcript supports the connection.

If no direct rebuttals exist, return an empty list.

--------------------------------------------------
6. DEBATE UNDERSTANDING
--------------------------------------------------

1. Identify the debate motion or central topic.
2. Provide a concise and balanced debate summary.
3. Preserve the original meaning of speakers.
4. Support English and Kiswahili transcripts.
5. Do not translate the transcript.
6. Do not add information that is absent from the transcript.
7. If the motion is not explicitly stated, infer the central
   topic cautiously and indicate that it is inferred.

--------------------------------------------------
7. OUTPUT REQUIREMENTS
--------------------------------------------------

Return valid JSON only.

Do not include Markdown code fences.
Do not include explanations outside the JSON.

Use exactly this structure:

{
  "debate_motion": "Identified motion or topic",
  "motion_status": "Explicit or Inferred",
  "summary": "Short summary of the debate",
  "speakers": [
    {
      "speaker_id": "S1",
      "speaker_label": "Speaker 1",
      "side": "Proposition, Opposition, or Unknown",
      "position_summary": "Summary of this speaker's position"
    }
  ],
  "arguments": [
    {
      "argument_id": "A1",
      "speaker_id": "S1",
      "side": "Proposition, Opposition, or Unknown",
      "argument_type": "Claim, Evidence, Rebuttal, Counterargument, or Conclusion",
      "claim": "The main claim",
      "reasoning": "Explanation of the argument",
      "evidence": "Evidence stated in the transcript, or Not provided",
      "evidence_type": "Statistic, Example, Expert testimony, Source, or None",
      "evidence_status": "Explicit evidence, Illustrative example, Unsupported assertion, or Not provided",
      "supporting_evidence": [],
      "responds_to": "A1 or None",
      "responds_to_ids": [],
      "relationship": "Explanation of the relationship, or None"
    }
  ],
  "rebuttals": [
    {
      "rebuttal_id": "R1",
      "speaker_id": "S2",
      "side": "Proposition, Opposition, or Unknown",
      "target_argument_id": "A1",
      "target_argument_ids": ["A1"],
      "rebuttal": "The direct response to the target argument",
      "explanation": "Why this statement challenges the target argument"
    }
  ]
}

ID AND REFERENCE RULES:

1. Use sequential speaker IDs: S1, S2, S3.
2. Use sequential argument IDs: A1, A2, A3.
3. Use sequential rebuttal IDs: R1, R2, R3.
4. Every speaker_id in an argument or rebuttal must
   refer to an identified speaker, or use "Unknown".
5. Every argument ID in responds_to_ids must refer
   to an existing argument.
6. Every target argument ID in a rebuttal must refer
   to an existing argument.
7. If no target exists, use "None" for target_argument_id
   and an empty target_argument_ids list.
8. If there is one target, target_argument_id must contain
   that ID.
9. If there are multiple targets, include all of them
   in target_argument_ids.
10. Do not create unsupported arguments or relationships.
"""


# JSON EXTRACTION
def extract_json(response_text):
    """
    Extract and decode JSON from Gemini's response.
    Handles accidental Markdown code fences.
    """

    if not response_text:
        raise ValueError(
            "Gemini returned an empty debate analysis."
        )

    cleaned = response_text.strip()

    # Remove opening Markdown code fence.
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    # Remove closing Markdown code fence.
    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        # Extract the outermost JSON object.
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "Gemini did not return valid JSON."
            )

        try:
            return json.loads(match.group(0))

        except json.JSONDecodeError as error:
            raise ValueError(
                "Could not decode the debate analysis JSON."
            ) from error


# ANALYSIS VALIDATION
def validate_analysis(analysis):
    """
    Validate the structure and references in the debate analysis.
    Also provides safe defaults for optional fields.
    """

    if not isinstance(analysis, dict):
        raise ValueError(
            "The debate analysis must be a JSON object."
        )

    required_fields = [
        "debate_motion",
        "summary",
        "speakers",
        "arguments",
        "rebuttals"
    ]

    for field in required_fields:

        if field not in analysis:
            raise ValueError(
                f"Debate analysis is missing the '{field}' field."
            )

    list_fields = [
        "speakers",
        "arguments",
        "rebuttals"
    ]

    for field in list_fields:

        if not isinstance(analysis[field], list):
            raise ValueError(
                f"Invalid '{field}' section returned by Gemini."
            )

    # Validate speakers.
    speaker_ids = set()

    for speaker in analysis["speakers"]:

        if not isinstance(speaker, dict):
            raise ValueError(
                "Invalid speaker entry returned by Gemini."
            )

        speaker_id = speaker.get("speaker_id")

        if not speaker_id:
            raise ValueError(
                "A speaker is missing its speaker ID."
            )

        if speaker_id in speaker_ids:
            raise ValueError(
                f"Duplicate speaker ID: {speaker_id}"
            )

        speaker_ids.add(speaker_id)

    # Validate arguments.
    argument_ids = set()

    for argument in analysis["arguments"]:

        if not isinstance(argument, dict):
            raise ValueError(
                "Invalid argument entry returned by Gemini."
            )

        argument_id = argument.get("argument_id")

        if not argument_id:
            raise ValueError(
                "An argument is missing its argument ID."
            )

        if argument_id in argument_ids:
            raise ValueError(
                f"Duplicate argument ID: {argument_id}"
            )

        argument_ids.add(argument_id)

        # Add defaults for optional fields.
        argument.setdefault("evidence", "Not provided")
        argument.setdefault("evidence_type", "None")
        argument.setdefault("evidence_status", "Not provided")
        argument.setdefault("supporting_evidence", [])
        argument.setdefault("relationship", "None")

        if not isinstance(argument["supporting_evidence"], list):
            raise ValueError(
                f"Invalid supporting evidence in {argument_id}."
            )

        # Preserve compatibility with the existing app.py.
        responds_to = argument.get("responds_to", "None")
        responds_to_ids = argument.get("responds_to_ids", [])

        if not isinstance(responds_to_ids, list):
            raise ValueError(
                f"Invalid responds_to_ids in {argument_id}."
            )

        if not responds_to_ids and responds_to not in (None, "None", ""):
            if isinstance(responds_to, str):
                responds_to_ids = [responds_to]

        argument["responds_to_ids"] = responds_to_ids

        if len(responds_to_ids) == 1:
            argument["responds_to"] = responds_to_ids[0]
        elif len(responds_to_ids) > 1:
            argument["responds_to"] = ", ".join(responds_to_ids)
        else:
            argument["responds_to"] = "None"

    # Validate argument speaker references and relationships.
    for argument in analysis["arguments"]:

        speaker_id = argument.get("speaker_id", "Unknown")

        if speaker_id != "Unknown" and speaker_id not in speaker_ids:
            raise ValueError(
                f"Argument {argument['argument_id']} references "
                f"unknown speaker {speaker_id}."
            )

        for target_id in argument["responds_to_ids"]:

            if target_id not in argument_ids:
                raise ValueError(
                    f"Argument {argument['argument_id']} references "
                    f"non-existent argument {target_id}."
                )

    # Validate rebuttals.
    rebuttal_ids = set()

    for rebuttal in analysis["rebuttals"]:

        if not isinstance(rebuttal, dict):
            raise ValueError(
                "Invalid rebuttal entry returned by Gemini."
            )

        rebuttal_id = rebuttal.get("rebuttal_id")

        if not rebuttal_id:
            raise ValueError(
                "A rebuttal is missing its rebuttal ID."
            )

        if rebuttal_id in rebuttal_ids:
            raise ValueError(
                f"Duplicate rebuttal ID: {rebuttal_id}"
            )

        rebuttal_ids.add(rebuttal_id)

        # Preserve compatibility with the existing app.py.
        target_ids = rebuttal.get("target_argument_ids", [])
        target_id = rebuttal.get("target_argument_id", "None")

        if not isinstance(target_ids, list):
            raise ValueError(
                f"Invalid target_argument_ids in {rebuttal_id}."
            )

        if not target_ids and target_id not in (None, "None", ""):
            target_ids = [target_id]

        rebuttal["target_argument_ids"] = target_ids

        if len(target_ids) == 1:
            rebuttal["target_argument_id"] = target_ids[0]
        elif len(target_ids) > 1:
            rebuttal["target_argument_id"] = ", ".join(target_ids)
        else:
            rebuttal["target_argument_id"] = "None"

        speaker_id = rebuttal.get("speaker_id", "Unknown")

        if speaker_id != "Unknown" and speaker_id not in speaker_ids:
            raise ValueError(
                f"Rebuttal {rebuttal_id} references "
                f"unknown speaker {speaker_id}."
            )

        for target in target_ids:

            if target not in argument_ids:
                raise ValueError(
                    f"Rebuttal {rebuttal_id} references "
                    f"non-existent argument {target}."
                )

    analysis.setdefault("motion_status", "Inferred")

    return analysis


# MAIN DEBATE ANALYZER
def analyze_debate(transcript):
    """
    Analyze a debate transcript.

    Identifies:
    - Debate motion
    - Speakers and their debate positions
    - Individual claims and arguments
    - Supporting evidence and evidence quality
    - Argument relationships
    - Direct rebuttals and their targets

    Returns a validated Python dictionary.
    """

    if not transcript or not transcript.strip():
        raise ValueError(
            "Please provide a debate transcript for analysis."
        )

    prompt = f"""
Analyze the following debate transcript.

Pay particular attention to the distinction between:
- Claims and supporting evidence.
- Illustrative examples and verified evidence.
- General disagreement and direct rebuttals.
- Arguments that challenge one target and arguments
  that challenge multiple targets.

Assign argument IDs before creating argument relationships.
Only reference argument IDs that actually exist.

Preserve the original language of the transcript.

DEBATE TRANSCRIPT:

{transcript}

Now identify the speakers, arguments, evidence,
argument relationships, and direct rebuttals.

Follow the required JSON structure exactly.
"""

    try:

        interaction = client.interactions.create(
            model=MODEL,
            input=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT + "\n\n" + prompt
                }
            ]
        )

        response_text = interaction.output_text

        analysis = extract_json(response_text)

        return validate_analysis(analysis)

    except Exception as error:

        raise RuntimeError(
            f"Debate analysis failed: {error}"
        ) from error
