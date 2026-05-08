"""Run LLM judge alignment over the participant-audio evaluation package.

The script ingests the 10-minute source transcript and generated summary transcript
from each ``Podcast_XX_<duration>`` folder, runs the shared conversation-summary
judge prompt with Gemini or DeepSeek, and correlates the judge's item and
dimension scores against the human survey responses.

References:
    [1] M. Stone, "Cross-validatory choice and assessment of statistical
        predictions," Journal of the Royal Statistical Society: Series B
        (Methodological), vol. 36, no. 2, pp. 111-133, 1974.
    [2] R. A. Fisher, "On the probable error of a coefficient of correlation
        deduced from a small sample," Metron, vol. 1, pp. 3-32, 1921.
    [3] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap.
        New York, NY, USA: Chapman and Hall, 1993.
    [4] D. T. Campbell and D. W. Fiske, "Convergent and discriminant
        validation by the multitrait-multimethod matrix," Psychological
        Bulletin, vol. 56, no. 2, pp. 81-105, 1959, doi: 10.1037/h0046016.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google import genai
from google.genai import types
from openai import OpenAI

from card_framework.benchmark.reference_free.conversation_judge import (
    DIMENSION_ITEMS,
    DIMENSION_NAMES,
    ITEM_NAMES,
    ConversationSummaryJudge,
    ConversationSummaryJudgeConfig,
)
from card_framework.providers.google_genai_provider import GoogleGenAIProvider
from card_framework.shared.env import load_repo_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSES_CSV = (
    REPO_ROOT / "Podcast Summary Evaluation (Responses) - Form Responses 1.csv"
)
DEFAULT_ASSIGNMENTS_CSV = (
    REPO_ROOT / "research" / "human_eval_analysis" / "participant_assignments.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "research" / "human_eval_analysis"
DEFAULT_TRANSCRIPT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "participant_audio_v3_transcripts"
DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_VLLM_API_KEY = "EMPTY"
DEEPSEEK_MAX_OUTPUT_TOKENS = 384_000
VLLM_CONTEXT_TOKEN_HEADROOM = 16_384
GEMINI_SCORE_CACHE_SCHEMA_VERSION = 1
CV_REPORT_SCHEMA_VERSION = 1
JUDGE_PROMPT_CACHE_VERSION = "conversation_summary_judge_item_scores_audio_v4"
PROMPT_PROFILE_NOTES = {
    "default": "",
    "human_calibrated": """
Score the dimensions independently:
- Factualness is the only dimension that should absorb unsupported claims,
  contradictions, or missing source context.
- Do not lower Naturalness or Speaker Grammar Similarity merely because a claim
  is factually unsupported, unless the factual issue directly creates an awkward
  jump, incoherent wording, or wrong speaker attribution.

Naturalness should approximate how survey listeners used N1-N3:
- N1 is about whether the candidate is easy to follow as a short podcast segment,
  not whether it covers every source detail.
- N2 is about whether the candidate's ordering feels locally conversational and
  not abruptly rearranged; do not require perfect transcript chronology under
  compression.
- N3 is about wording/readability. Penalize obvious stitching: forced pivots
  such as unsupported "speaking of" jumps, interjections that interrupt sentence
  grammar, robotic echo backchannels, repeated laugh tokens, list-like summaries
  that stop sounding conversational, or abrupt endings.
- Short 30-second summaries may feel teaser-like; do not lower N solely for
  brevity if the wording and turn order are smooth.

Speaker Grammar Similarity should approximate S1-S3 from the survey:
- Reward recognizable casual podcast voice, speaker-labelled turns, fragments,
  fillers, idioms, informal grammar, and backchannels when they fit the turn.
- Do not require exact source wording or full filler preservation after
  compression.
- Penalize generic narrator style, over-polished formal wording, style cues that
  are attached to the wrong speaker, or inserted backchannels/interjections that
  make the wrong speaker seem to say or react to the wrong thing.
""".strip(),
    "human_calibrated_lenient_style": """
Keep dimensions independent:
- Put unsupported claims and omissions in Factualness only.
- Do not let Factualness errors drag down Naturalness or Speaker Grammar
  Similarity unless they visibly make the summary wording incoherent or attach
  a claim/reaction to the wrong speaker.

Naturalness anchors:
- Human raters mostly used N1-N3 as surface conversational readability. A short
  summary can score high even when it omits nuance.
- Give 0.75-1.00 when the turn sequence is understandable, locally coherent,
  and the wording sounds like a plausible podcast exchange.
- Use 0.25-0.50 only for clear awkwardness: a forced transition that breaks the
  topic, an interjection inserted mid-phrase so the sentence becomes unnatural,
  robotic echo/backchannel repetition, long laugh tokens, or a stitched/list-like
  recap that stops feeling like a conversation.

Speaker Grammar Similarity anchors:
- Be lenient and human-like. Start from 0.75 when speaker labels are stable,
  content remains with the right speaker, and the text has any casual podcast
  style markers.
- Reward direct address, names, slang, fragments, fillers, informal grammar,
  repeated emphatic phrases, and source-like catchphrases such as "100%",
  "yeah", "right", "man", or host/guest banter. These can outweigh imperfect
  compression.
- Do not require exact source wording or full preservation of every filler.
- Lower S2/S3 only when the summary becomes generic narrator prose, strips nearly
  all speaker-specific style, gives one speaker the other's stance, or uses an
  interjection/backchannel that clearly belongs to the wrong speaker.
""".strip(),
    "human_calibrated_style_sensitive": """
Separate dimensions first:
- Factual errors and omissions belong in Factualness only.
- Naturalness and Speaker Grammar Similarity should not be lowered merely
  because a claim is unsupported unless it also creates a visible wording,
  turn-order, or speaker-attribution problem.

Naturalness calibration:
- Favor the human listener view: a summary can be easy to follow and naturally
  ordered even when it is compressed or factually imperfect.
- Give N1/N2 0.75-1.00 when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- Penalize N3 and sometimes N2 for visible synthetic stitching: forced pivots,
  abrupt topic jumps, backchannels inserted inside a phrase, echo backchannels
  that repeat the previous words mechanically, long laugh tokens, or a rushed
  list of unrelated topics.

Speaker Grammar Similarity calibration:
- Be style-sensitive rather than universally lenient. Reward distinctive source
  voice markers: direct address by name, repeated emphatic phrases, slang,
  fragments, fillers, informal grammar, host/guest banter, and catchphrases
  like "100%", "yeah", "right", "man", or topic-specific phrasing.
- Penalize generic summary voice where both speakers sound interchangeable,
  polished, or narrator-like.
- Penalize unnatural generated backchannels such as generic "that's really cool",
  "that's amazing", or word-for-word echo snippets when they do not match the
  speaker's role or interrupt grammar.
- S3 should stay high when the main content stays with the correct speaker, but
  should drop if reactions/interjections make the wrong speaker seem to hold a
  stance or if source-speaker roles are blurred.
""".strip(),
    "human_calibrated_style_sensitive_factual_duration_v1": """
Separate dimensions first:
- Factual errors and omissions belong in Factualness only.
- Naturalness and Speaker Grammar Similarity should not be lowered merely
  because a claim is unsupported unless it also creates a visible wording,
  turn-order, or speaker-attribution problem.

Factualness calibration:
- F1 is support/contradiction. Keep F1 high when claims are directly supported,
  strongly implied, or reasonable compressed paraphrases of the source. Lower
  F1 for explicit inventions, contradictions, or a claim/reaction attached to
  the wrong speaker.
- F2 is main-topic retention relative to the requested duration. A 30-second
  summary may omit most source topics and still score high if it preserves the
  central beat. A 1-minute summary should keep the main story arc. A 5-minute
  summary should preserve more of the major topic path, but not every detail.
- F3 is duration-appropriate compression, not an overall quality score and not
  a completeness meter. Keep F3 high when omissions are reasonable for the
  target duration. Lower F3 mainly when the summary omits context needed to
  understand its own central point, pads a short target, drops the main arc for
  a longer target, or replaces the main arc with an unsupported one.
- If a summary has one unsupported detail, penalize F1 for that detail, but do
  not automatically collapse F2/F3 unless the unsupported detail becomes the
  summary's main arc.

Naturalness calibration:
- Favor the human listener view: a summary can be easy to follow and naturally
  ordered even when it is compressed or factually imperfect.
- Give N1/N2 0.75-1.00 when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- Penalize N3 and sometimes N2 for visible synthetic stitching: forced pivots,
  abrupt topic jumps, backchannels inserted inside a phrase, echo backchannels
  that repeat the previous words mechanically, long laugh tokens, or a rushed
  list of unrelated topics.

Speaker Grammar Similarity calibration:
- Be style-sensitive rather than universally lenient. Reward distinctive source
  voice markers: direct address by name, repeated emphatic phrases, slang,
  fragments, fillers, informal grammar, host/guest banter, and catchphrases
  like "100%", "yeah", "right", "man", or topic-specific phrasing.
- Penalize generic summary voice where both speakers sound interchangeable,
  polished, or narrator-like.
- Penalize unnatural generated backchannels such as generic "that's really cool",
  "that's amazing", or word-for-word echo snippets when they do not match the
  speaker's role or interrupt grammar.
- S3 should stay high when the main content stays with the correct speaker, but
  should drop if reactions/interjections make the wrong speaker seem to hold a
  stance or if source-speaker roles are blurred.
""".strip(),
    "human_calibrated_style_sensitive_factual_duration_speaker_guard_v1": """
Separate dimensions first:
- Factual errors and omissions belong in Factualness only.
- Naturalness and Speaker Grammar Similarity should not be lowered merely
  because a claim is unsupported unless it also creates a visible wording,
  turn-order, or speaker-attribution problem.
- High Factualness or Naturalness is not evidence for high Speaker Grammar
  Similarity. Speaker scores require speaker-specific evidence.

Factualness calibration:
- F1 is support/contradiction. Keep F1 high when claims are directly supported,
  strongly implied, or reasonable compressed paraphrases of the source. Lower
  F1 for explicit inventions, contradictions, or a claim/reaction attached to
  the wrong speaker.
- F2 is main-topic retention relative to the requested duration. A 30-second
  summary may omit most source topics and still score high if it preserves the
  central beat. A 1-minute summary should keep the main story arc. A 5-minute
  summary should preserve more of the major topic path, but not every detail.
- F3 is duration-appropriate compression, not an overall quality score and not
  a completeness meter. Keep F3 high when omissions are reasonable for the
  target duration. Lower F3 mainly when the summary omits context needed to
  understand its own central point, pads a short target, drops the main arc for
  a longer target, or replaces the main arc with an unsupported one.
- If a summary has one unsupported detail, penalize F1 for that detail, but do
  not automatically collapse F2/F3 unless the unsupported detail becomes the
  summary's main arc.

Naturalness calibration:
- Favor the human listener view: a summary can be easy to follow and naturally
  ordered even when it is compressed or factually imperfect.
- Give N1/N2 high scores when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- Penalize N3 and sometimes N2 for visible synthetic stitching: forced pivots,
  abrupt topic jumps, backchannels inserted inside a phrase, echo backchannels
  that repeat the previous words mechanically, long laugh tokens, or a rushed
  list of unrelated topics.
- Do not lower N1/N2 merely because the speaker style is generic; put generic
  speaker voice in Speaker Grammar Similarity unless it also harms flow.

Speaker Grammar Similarity calibration:
- Be style-sensitive rather than universally lenient. Reward distinctive source
  voice markers: direct address by name, repeated emphatic phrases, slang,
  fragments, fillers, informal grammar, host/guest banter, and catchphrases
  like "100%", "yeah", "right", "man", or topic-specific phrasing.
- S1 is recognizable persona/role, not just the presence of speaker labels.
  Penalize generic summary voice where both speakers sound interchangeable,
  polished, or narrator-like.
- S2 is verbal texture. Do not require exact source wording, but lower S2 when
  speaker-specific rhythm is replaced by canned supportive lines, formal prose,
  or mechanical word echoes.
- S3 is ownership and role consistency. Keep S3 high when main content stays
  with the correct speaker, but lower it if reactions/interjections make the
  wrong speaker seem to hold a stance, blur host/guest roles, or interrupt
  grammar in a way that changes who is speaking.
- Stable labels plus generic wording usually support S3 more than S1/S2. Do not
  use correct labels alone to make all three speaker items high.

Before finalizing, verify that S1, S2, and S3 each have separate evidence. If
Speaker Grammar Similarity is uniformly high, confirm the summary preserves
distinct persona, verbal texture, and correct ownership rather than only
speaker labels and factual content.
""".strip(),
    "human_calibrated_style_sensitive_guarded_v1": """
Separate dimensions first:
- Factual errors and omissions belong in Factualness only.
- Naturalness and Speaker Grammar Similarity should not be lowered merely
  because a claim is unsupported unless it also creates a visible wording,
  turn-order, or speaker-attribution problem.

Factualness calibration:
- F1 is support/contradiction. Keep F1 high when claims are directly supported,
  strongly implied, or reasonable compressed paraphrases. Lower it for explicit
  inventions, contradictions, or a claim/reaction attached to the wrong speaker.
- F2 is main-topic retention relative to the requested duration. A 30-second
  summary may omit most source topics and still score high if it preserves the
  central beat. A 1-minute summary should keep the main story arc. A 5-minute
  summary should preserve more of the major topic path, but not every detail.
- F3 is duration-appropriate compression, not an overall quality score and not
  a completeness meter. Keep F3 high when the amount omitted is reasonable for
  the target duration. Lower F3 mainly when the summary omits context needed to
  understand its own central point, pads a short target, drops the main arc for
  a longer target, or replaces the main arc with an unsupported one.
- If a summary has one unsupported detail, penalize F1 for that detail, but do
  not automatically collapse F2/F3 unless the unsupported detail becomes the
  summary's main arc.

Naturalness calibration:
- Favor the human listener view: a summary can be easy to follow and naturally
  ordered even when it is compressed or factually imperfect.
- Give N1/N2 high scores when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- N3 is wording and stitching. Penalize visible synthetic stitching such as
  forced pivots, abrupt topic jumps, broken XML insertion points, mechanical
  echo backchannels, long laugh tokens, rushed unrelated lists, or narrator-like
  recap prose.
- Do not over-penalize one generic backchannel or one slightly awkward
  transition. Lower N3 strongly only when awkward stitching is repeated,
  grammar-breaking, or makes the conversation hard to follow.

Speaker Grammar Similarity calibration:
- Be style-sensitive rather than universally lenient. Reward distinctive source
  voice markers: direct address by name, repeated emphatic phrases, slang,
  fragments, fillers, informal grammar, host/guest banter, and catchphrases
  like "100%", "yeah", "right", "man", or topic-specific phrasing.
- S2 should not require exact fillers, catchphrases, or source wording. Score S2
  high when the candidate preserves a casual role-consistent voice, even with
  generic but plausible phrasing.
- Penalize generic summary voice where both speakers sound interchangeable,
  polished, or narrator-like. Penalize repeated canned supportive reactions or
  word-for-word echo snippets when they do not match the speaker's role or
  interrupt grammar.
- S3 is correct ownership. Keep S3 high when the main claims, reactions, and
  stances stay attached to the right speaker. Do not lower S3 merely because
  S1/S2 style is weak or details are omitted. Lower S3 for role swaps,
  misleading interjections, or blurred speaker ownership.

Before finalizing scores, sanity-check whether F3, N3, S2, and S3 each have
item-specific evidence. Do not let a single severe factual error flatten every
Naturalness or Speaker Grammar Similarity item.
""".strip(),
    "human_calibrated_style_sensitive_guarded_speaker_restore_v1": """
Separate dimensions first:
- Factual errors and omissions belong in Factualness only.
- Naturalness and Speaker Grammar Similarity should not be lowered merely
  because a claim is unsupported unless it also creates a visible wording,
  turn-order, or speaker-attribution problem.

Factualness calibration:
- F1 is support/contradiction. Keep F1 high when claims are directly supported,
  strongly implied, or reasonable compressed paraphrases. Lower it for explicit
  inventions, contradictions, or a claim/reaction attached to the wrong speaker.
- F2 is main-topic retention relative to the requested duration. A 30-second
  summary may omit most source topics and still score high if it preserves the
  central beat. A 1-minute summary should keep the main story arc. A 5-minute
  summary should preserve more of the major topic path, but not every detail.
- F3 is duration-appropriate compression, not an overall quality score and not
  a completeness meter. Keep F3 high when the amount omitted is reasonable for
  the target duration. Lower F3 mainly when the summary omits context needed to
  understand its own central point, pads a short target, drops the main arc for
  a longer target, or replaces the main arc with an unsupported one.
- If a summary has one unsupported detail, penalize F1 for that detail, but do
  not automatically collapse F2/F3 unless the unsupported detail becomes the
  summary's main arc.

Naturalness calibration:
- Favor the human listener view: a summary can be easy to follow and naturally
  ordered even when it is compressed or factually imperfect.
- Give N1/N2 high scores when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- N3 is wording and stitching. Penalize visible synthetic stitching such as
  forced pivots, abrupt topic jumps, broken XML insertion points, mechanical
  echo backchannels, long laugh tokens, rushed unrelated lists, or narrator-like
  recap prose.
- Do not over-penalize one generic backchannel or one slightly awkward
  transition. Lower N3 strongly only when awkward stitching is repeated,
  grammar-breaking, or makes the conversation hard to follow.

Speaker Grammar Similarity calibration:
- Be style-sensitive rather than universally lenient. Reward distinctive source
  voice markers: direct address by name, repeated emphatic phrases, slang,
  fragments, fillers, informal grammar, host/guest banter, and catchphrases
  like "100%", "yeah", "right", "man", or topic-specific phrasing.
- S1 is speaker recognizability. Penalize generic summary voice where both
  speakers sound interchangeable, polished, or narrator-like, even if the facts
  are mostly correct.
- S2 is verbal texture. Do not require exact fillers, catchphrases, or source
  wording, but lower S2 when the candidate replaces speaker-specific rhythm with
  canned supportive lines, polished generic prose, or mechanical word echoes.
- S3 is role and ownership. Keep S3 high when main claims, reactions, and
  stances stay attached to the right speaker, but lower it when interjections or
  reactions make the wrong speaker seem to hold a stance, blur host/guest roles,
  or interrupt grammar in a way that changes who is speaking.
- Penalize unnatural generated backchannels such as generic "that's really
  cool", "that's amazing", or word-for-word echo snippets when they do not match
  the speaker's role or interrupt the sentence.

Before finalizing scores, sanity-check whether F3, N3, S2, and S3 each have
item-specific evidence. Do not let duration leniency erase speaker-style or
speaker-ownership defects.
""".strip(),
    "judge_reliability_v1": """
Do not use any external human scores, artifact IDs, response statistics, or
participant comments. Score only from the raw source transcript, candidate
summary XML, target duration, and the rubric.

Use a human-rater mental model:
- Human listeners often rate a compressed podcast as natural when it is easy to
  follow locally, even if it omits nuance. Keep omission penalties in
  Factualness unless the omission causes an abrupt jump.
- Do not over-penalize 30-second summaries for being teaser-like when the target
  duration is 30 seconds and the summary has a coherent arc.
- Naturalness should focus on flow and wording: coherent turn sequence,
  plausible reactions, smooth pivots, and non-robotic interjections.
- Penalize unnatural XML interjections when they interrupt grammar, repeat a word
  mechanically, arrive before the triggering information, or sound like generic
  filler pasted into the wrong place.

Speaker Grammar Similarity should focus on speaker identity and attachment:
- S1 asks whether each speaker remains recognizable as a distinct host/guest or
  conversational role. Distinct turns and role-consistent voice can score well
  without exact source wording.
- S2 asks whether style cues are preserved when they matter: casual fragments,
  fillers, slang, direct address, uncertainty, emphatic repetition, or topic-
  specific phrasing. Generic polished prose should score lower.
- S3 asks whether content and style cues stay attached to the correct speaker.
  Keep S3 high when the main claims/stylistic reactions are assigned correctly;
  lower it for role swaps or misleading interjections.

Use the full 0.00/0.25/0.50/0.75/1.00 range. Avoid rating every summary in the
middle. Good but imperfect summaries should often receive 0.75; clearly awkward
or generic summaries should receive 0.25 or 0.50.
""".strip(),
    "human_listener_tolerant_ranked": """
Do not use human labels, artifact IDs, response statistics, participant comments,
or any source outside the provided transcript, summary, target duration, and
rubric. Emulate a survey listener who rewards a plausible podcast-summary
experience, not an expert auditor trying to mark every small transcript defect.

Use a 0.75 baseline for a summary that is understandable, mostly supported, and
role-consistent. Move upward for strong coverage, smooth local flow, and
distinct speaker voices. Move downward only for defects that a normal listener
would likely notice without doing a line-by-line forensic comparison.

Factualness calibration:
- Keep F1 high when claims are directly supported or are reasonable compressed
  paraphrases of the transcript. Lower F1 for explicit contradictions or clearly
  invented events, not for ordinary synthesis.
- Score F2/F3 relative to the target duration. A 30-second summary can omit most
  details and still score well if it preserves the central arc. A 5-minute
  summary should preserve more of the major topic sequence.
- Do not move factual penalties into Naturalness or Speaker Grammar Similarity
  unless the factual issue causes an obvious jump or wrong-speaker reaction.

Naturalness calibration:
- Reward concise summaries that form a coherent beginning, middle, and ending,
  even when they are short or omit nuance.
- Treat one isolated awkward XML interjection, echo, or clipped phrase as a
  minor local issue when the surrounding conversation remains easy to follow.
- Penalize repeated generic backchannels, forced "speaking of" pivots, padded
  recap endings, or interjections placed mid-phrase when those patterns make the
  summary sound stitched together instead of conversational.
- Do not give high Naturalness just because the summary is long, polished, or
  complete-looking; long summaries can feel synthetic when they over-explain or
  close with narrator-like recap prose.

Speaker Grammar Similarity calibration:
- Reward distinct speaker roles, stable speaker labels, direct address, casual
  fragments, emphatic phrasing, and source-like oddities when those cues fit the
  original speaker. Odd phrasing is not a style flaw if it reflects the source.
- A supportive interviewer can use simple supportive language without losing all
  speaker similarity, especially in an emotional interview, as long as the main
  speaker's story and stance stay attached correctly.
- Penalize generic narrator voice, interchangeable speakers, repeated canned
  reactions such as "that's really cool" or "that's amazing", exact-word echo
  snippets used mechanically, and reactions that make the wrong speaker seem to
  hold a stance.
- Keep S3 high when the main claims and roles stay attached to the correct
  speaker; lower it for role swaps, misleading interjections, or blurred speaker
  ownership.

Use the full 0.00/0.25/0.50/0.75/1.00 scale, but reserve 0.00 or 0.25 for clear
failures. Most plausible but imperfect summaries should land at 0.50 or 0.75 on
the affected item rather than collapsing to 0.25 across the whole dimension.
""".strip(),
    "predicted_human_listener_v1": """
Your job is to predict how a normal survey listener would answer the nine
Likert guide items after comparing the provided source transcript and candidate
summary. Do not optimize for a perfectly objective expert audit. Do not use
artifact IDs, human labels, participant comments, response statistics, or any
external data.

General rater-behavior prior:
- Human listeners are usually generous when a summary has a clear central arc,
  stable speaker labels, and no obvious contradiction.
- F1, N1, S1, and S3 should often remain high for plausible summaries because
  listeners notice direct contradiction, followability, recognizable roles, and
  speaker attachment more than subtle compression defects.
- F3, N3, and S2 are the more discriminating items. Use them to reflect lossy
  compression, stitched wording, over-polished narrator voice, canned reactions,
  and missing speaker-specific texture.
- Avoid all-high scoring. A polished summary with repeated generic backchannels
  or a padded recap can still be only mixed on N3/S2 even if F1 is high.

Factualness:
- F1 asks whether the summary obviously contradicts or invents source events.
  Compressed paraphrase, emotional synthesis, and source-supported inference are
  not failures.
- F2 asks whether the main topic sequence remains. Target duration matters:
  30-second summaries can keep only the central beat; 5-minute summaries should
  retain more of the episode's topic path.
- F3 asks whether the amount removed feels appropriate. Penalize summaries that
  feel teaser-like for a longer target or over-expanded with filler for a short
  target.

Naturalness:
- N1 is broad followability. If a listener can understand the segment from
  start to finish, N1 should usually be 0.75 or 1.00.
- N2 is local conversational ordering, not strict transcript chronology.
- N3 is where awkwardness belongs: repeated generic "yeah exactly" style
  backchannels, echo snippets that repeat mechanically, interjections inside
  broken syntax, forced topic pivots, abrupt endings, or narrator-like recaps.
  A single small artifact should usually lower only N3 by one step.

Speaker Grammar Similarity:
- S1 is recognizability of speaker role or persona. Speaker labels, informal
  tone, direct address, stance, and clear host/guest roles all count.
- S2 is preservation of distinctive verbal texture. Reward source-like oddities,
  slang, fragments, emphatic phrases, uncertainty, and casual rhythm when they
  fit. Penalize generic polished prose and repeated canned supportive lines.
- S3 is correct ownership. Keep it high when main claims and reactions stay
  attached to the right speaker; lower it for role swaps or misleading
  interjections.

Use only 0.00, 0.25, 0.50, 0.75, or 1.00 for each item score. Use 0.00 rarely,
only for a clear item-level failure. Prefer 0.50 for mixed and 0.75 for good but
noticeably imperfect.
""".strip(),
    "listener_factual_style_sensitive_v1": """
Do not use human labels, artifact IDs, participant comments, response
statistics, or external data. Score only from the provided source transcript,
summary XML, target duration, and rubric.

Apply different judgment behavior by dimension:

Factualness should approximate a normal survey listener rather than a forensic
fact checker.
- Keep F1 high when claims are directly supported, strongly implied, or are
  reasonable compressed paraphrases. Lower F1 only for clear contradiction,
  clearly invented events, or a reaction that changes who did what.
- F2 and F3 are target-duration-aware. A 30-second summary may keep only the
  central beat; a 1-minute summary should keep the main story arc; a 5-minute
  summary should preserve more major topic sequence and not replace it with
  filler.
- Do not let factual omissions automatically lower Naturalness or Speaker
  Grammar Similarity.

Naturalness should be style-sensitive, not universally lenient.
- Give N1/N2 high scores when the local turn sequence is understandable and the
  summary reads like a plausible podcast excerpt.
- Penalize N3 for visible synthetic stitching: forced pivots, repeated generic
  backchannels, exact echo snippets, interjections inside broken syntax, long
  laugh tokens, rushed list-like recaps, or narrator-like closing summaries.
- Do not reward length or polish by itself. A long summary can be less natural
  than a short one if it pads the exchange or over-explains the arc.
- Do not let one isolated awkward interjection dominate N1/N2 when the overall
  exchange remains easy to follow.

Speaker Grammar Similarity should focus on role, voice, and attachment.
- Reward distinctive source voice markers: direct address, slang, fragments,
  fillers, informal grammar, emphatic phrases, uncertainty, host/guest banter,
  and odd source-like phrasing.
- Penalize generic narrator voice, over-polished wording, interchangeable
  speakers, repeated canned supportive reactions, and mechanical echo snippets.
- S3 should stay high when main content and reactions remain attached to the
  right speaker, but should drop for role swaps or misleading interjections.

Use only 0.00, 0.25, 0.50, 0.75, or 1.00 for item scores. Use 0.00 rarely.
Use 0.50 for mixed, 0.75 for good with minor issues, and 1.00 only when the item
has no meaningful flaw under the dimension-specific guidance above.
""".strip(),
    "synthetic_anchor_listener_v1": """
Do not use human labels, artifact IDs, participant comments, response
statistics, or external data. The examples below are synthetic scale anchors;
they are not evaluation artifacts and should not be matched by topic. Use them
only to calibrate item-level severity.

Synthetic scale anchors:

Anchor A, concise good summary:
- The summary keeps the central source arc, has stable speaker labels, uses
  plausible casual wording, and has only ordinary compression omissions.
- Typical scores: F1=1.00, F2=0.75, F3=0.75, N1=1.00, N2=0.75, N3=0.75,
  S1=0.75, S2=0.75, S3=0.75.

Anchor B, supported but synthetic-sounding summary:
- Main facts are supported, but the summary is padded with recap prose, repeated
  generic backchannels, or polished narrator-like turns that make speakers less
  distinct.
- Typical scores: F1=1.00, F2=0.75, F3=0.50, N1=0.75, N2=0.50, N3=0.25,
  S1=0.50, S2=0.25, S3=0.75.

Anchor C, wrong-speaker or invented-event summary:
- The summary invents a central event, contradicts the source, or assigns a key
  stance or reaction to the wrong speaker.
- Typical scores: F1=0.25, F2=0.50, F3=0.50, N1=0.50, N2=0.50, N3=0.25,
  S1=0.25, S2=0.25, S3=0.25.

Anchor D, short target-duration teaser:
- For a 30-second target, the summary keeps a coherent central beat and sounds
  conversational, but necessarily omits most nuance.
- Typical scores: F1=1.00, F2=0.75, F3=0.75, N1=1.00, N2=0.75, N3=0.75,
  S1=0.75, S2=0.50, S3=0.75.

When scoring the real candidate, choose the closest severity pattern by item,
not by topic. Keep dimensions independent:
- Factualness handles support, contradiction, and compression appropriateness.
- Naturalness handles flow, wording, local turn order, and synthetic stitching.
- Speaker Grammar Similarity handles recognizability, verbal texture, and
  correct speaker ownership.

Use only 0.00, 0.25, 0.50, 0.75, or 1.00 for item scores. Reserve 0.00 for
near-total item failure.
""".strip(),
    "qwen_item_separated_listener_v1": """
Do not use human labels, artifact IDs, participant comments, response
statistics, or external data. Score only from the source transcript, summary
XML, target duration, and rubric.

First decide the item scores independently. Do not make all items in one
dimension move together. Each item has a separate job:

Factualness:
- F1 is support/contradiction only. Use 0.75 or 1.00 when the central events are
  supported, even if the summary is compressed or stylistically awkward. Lower
  it for clear invented events, contradictions, or wrong-speaker claims.
- F2 is important-topic retention. For 30 seconds, reward keeping the central
  beat. For 1 minute, reward keeping the main story arc. For 5 minutes, reward
  keeping the major topic sequence rather than padding with generic wrap-up.
- F3 is compression appropriateness. Do not use F3 as a general quality score.
  A concise short target can score high; a long target should lose F3 when it
  omits major arcs or fills space with repetition.

Naturalness:
- N1 is broad followability. Vary it: use 1.00 for a very easy coherent segment,
  0.75 for understandable with minor artifacts, 0.50 for a segment that takes
  effort to follow, and 0.25 for broken or confusing flow.
- N2 is local conversational order. It does not require exact transcript
  chronology, but topic jumps and misplaced turns should lower it.
- N3 is surface wording and stitching. Put repeated canned backchannels,
  mechanical echoes, broken XML insertion points, abrupt endings, and narrator
  recap prose here.

Speaker Grammar Similarity:
- S1 is recognizable speaker role/persona. Speaker labels, host/guest roles,
  stance, casual tone, and direct address count even without exact wording.
- S2 is verbal texture. Reward fragments, slang, filler-like rhythm, emphatic
  phrasing, and source-like oddities. Penalize polished generic prose and canned
  supportive reactions.
- S3 is correct ownership only. Keep S3 high when main claims and reactions are
  attached to the right speaker. Do not lower S3 just because S1/S2 style is weak
  or because details are omitted. Lower S3 for role swaps, misleading
  interjections, or a stance assigned to the wrong speaker.

Use only 0.00, 0.25, 0.50, 0.75, or 1.00. Avoid constant item scores across all
summaries: N1, S3, and F3 should still vary when the candidate actually differs
on those item-specific criteria.
""".strip(),
    "qwen_duration_guarded_style_v1": """
Do not use human labels, artifact IDs, participant comments, response
statistics, or external data. Score only from the source transcript, summary
XML, target duration, and rubric.

Use the human-listener frame, then apply these guardrails:
- Do not score as a forensic transcript auditor. A compressed summary can be
  good when it preserves the central podcast experience for the requested
  duration.
- Do not let one item's defect automatically pull down neighboring items. Make
  a separate evidence decision for F1, F2, F3, N1, N2, N3, S1, S2, and S3.
- Use 0.75 for good-but-imperfect items. Use 0.50 for mixed evidence. Reserve
  0.25 or 0.00 for clear item-level failures.

Factualness guardrails:
- F1 is support/contradiction. Keep F1 high for directly supported,
  strongly implied, or reasonable compressed paraphrases. Lower it for explicit
  inventions, contradictions, or a claim/reaction attached to the wrong speaker.
- F2 is main-topic retention. For 30 seconds, retaining the central beat is
  enough for a high score. For 1 minute, expect the main story arc. For 5
  minutes, expect more of the major topic path, but not every detail.
- F3 is duration-appropriate compression, not an overall quality score and not
  a completeness meter. Keep F3 at 0.75 when the amount omitted is reasonable
  for the target duration. Lower F3 when a longer target drops a major arc, a
  short target becomes padded, or compression removes context needed to
  understand the central point.

Naturalness guardrails:
- N1 is broad followability. Score high when a listener can follow the segment
  from start to finish, even if some facts are incomplete.
- N2 is local conversational order, not exact source chronology. Penalize
  misplaced turns, sudden unsupported pivots, or ordering that makes reactions
  arrive before their cause.
- N3 is wording and stitching. Penalize broken XML insertion points, repeated
  canned backchannels, mechanical echo snippets, long laugh tokens, abrupt
  endings, and narrator-like recap prose.

Speaker Grammar Similarity guardrails:
- S1 is recognizable speaker role/persona. Distinct host/guest roles, direct
  address, stance, casual tone, and stable labels can score well without exact
  source wording.
- S2 is verbal texture. Reward source-like fragments, slang, filler rhythm,
  emphatic phrasing, informal grammar, and odd but role-consistent turns.
  Penalize polished generic prose and repeated canned supportive reactions.
- S3 is correct ownership. Keep S3 high when main claims and reactions remain
  attached to the right speaker; lower it for role swaps, misleading
  interjections, or blurred speaker ownership.

Before finalizing, briefly sanity-check whether each item score has
item-specific evidence. If a dimension looks uniformly low or high, revise only
when the individual items justify it.
""".strip(),
    "in_sample_gold_anchor": """
This run is intentionally calibrated against the same human-response set used
for reporting. Treat the provided artifact-level human mean scores as gold
calibration anchors for this exact artifact and output those item_scores exactly.
This is useful for verifying the correlation pipeline and for estimating an
upper bound, but it is not an out-of-sample judge reliability estimate.
""".strip(),
}
COMMENT_COLUMN = (
    "Provide any additional qualitative remarks or observations regarding the podcast "
    "summary."
)
LIKERT_TO_NORMALIZED = {
    1: 0.0,
    2: 0.25,
    3: 0.5,
    4: 0.75,
    5: 1.0,
}
ITEM_TO_DIMENSION = {
    item_name: dimension_name
    for dimension_name, item_names in DIMENSION_ITEMS.items()
    for item_name in item_names
}
_ITEM_CODE_PATTERN = re.compile(r"\[([FNS][1-3])\s*-")
_FOLDER_PATTERN = re.compile(r"^Podcast_(?P<podcast>\d{2})_(?P<duration>30s|1m|5m)$")
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"}


@dataclass(slots=True, frozen=True)
class Assignment:
    """One participant-to-artifact assignment row."""

    participant_id: str
    participant_email: str
    podcast_id: str
    duration: str
    folder_link: str

    @property
    def artifact_key(self) -> str:
        """Return the normalized artifact key used for manifests."""
        return build_artifact_key(self.podcast_id, self.duration)


@dataclass(slots=True, frozen=True)
class ParticipantArtifact:
    """Local participant-audio artifact metadata used by the judge."""

    artifact_key: str
    podcast_id: str
    duration: str
    source_transcript_path: Path
    summary_transcript_path: Path
    source_audio_path: Path | None
    summary_audio_path: Path | None
    speaker_sample_audio_paths: tuple[Path, ...]
    local_source_transcript_path: Path
    local_summary_transcript_path: Path

    @property
    def target_duration_label(self) -> str:
        """Return the human-readable target-duration label."""
        if self.duration == "30s":
            return "30 seconds"
        if self.duration == "1m":
            return "1 minute"
        if self.duration == "5m":
            return "5 minutes"
        return self.duration


@dataclass(slots=True, frozen=True)
class HumanResponse:
    """Parsed human survey response with normalized item and dimension scores."""

    participant_id: str
    participant_email: str
    assignment: Assignment
    item_scores: dict[str, float]
    dimension_scores: dict[str, float]
    comment: str


def build_artifact_key(podcast_id: str, duration: str) -> str:
    """Build the canonical key for one podcast-duration artifact."""
    podcast_number = re.search(r"(\d{1,2})", podcast_id)
    if podcast_number is None:
        raise ValueError(f"Could not parse podcast id from {podcast_id!r}.")
    return f"Podcast_{int(podcast_number.group(1)):02d}_{duration.strip()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-csv", type=Path, default=DEFAULT_RESPONSES_CSV)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS_CSV)
    parser.add_argument(
        "--participant-audio-root",
        type=Path,
        required=True,
        help="Folder containing Podcast_XX_<duration> participant-audio subfolders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transcript-output-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--judge-provider",
        choices=("gemini", "deepseek", "vllm"),
        default="gemini",
    )
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Thinking toggle for DeepSeek and vLLM. Ignored by Gemini provider.",
    )
    parser.add_argument(
        "--gemini-thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="low",
        help="Gemini 3 thinking level for Gemini judge calls.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="",
        help="DeepSeek OpenAI-compatible base URL. Defaults to DEEPSEEK_BASE_URL or https://api.deepseek.com.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="",
        help=(
            "OpenAI-compatible vLLM base URL. Defaults to VLLM_URL, "
            "CARD_FRAMEWORK_VLLM_URL, VLLM_BASE_URL, or CARD_FRAMEWORK_VLLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--vllm-api-key",
        default="",
        help="vLLM API key. Defaults to VLLM_API_KEY, CARD_FRAMEWORK_VLLM_API_KEY, or EMPTY.",
    )
    parser.add_argument(
        "--deepseek-reasoning-effort",
        choices=("high", "max"),
        default="high",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=tuple(PROMPT_PROFILE_NOTES),
        default="default",
        help="Optional extra scoring guidance injected into the judge prompt.",
    )
    parser.add_argument(
        "--allow-label-leakage",
        action="store_true",
        help="Allow prompt profiles that inject human score anchors. Do not use for judge reliability.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("none", "summary", "source-and-summary", "human-folder"),
        default="source-and-summary",
        help=(
            "Audio attachments to send with the raw source/summary transcripts. "
            "'human-folder' attaches all audio files shown in the human rater folder."
        ),
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum judge completion tokens. Defaults to the largest practical "
            "provider limit instead of the old 8192 cap."
        ),
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        help=(
            "Best-effort generation seed for providers that support it. Gemini and "
            "vLLM accept this; DeepSeek's current Chat API does not document a seed "
            "parameter, so DeepSeek runs should use --judge-temperature 0 and cached "
            "traces for reproducibility."
        ),
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help=(
            "Optional judge sampling temperature. Use 0 for more stable judge reruns. "
            "When omitted, provider defaults are preserved except vLLM, which keeps "
            "its existing temperature=0 default."
        ),
    )
    parser.add_argument("--max-judge-attempts", type=int, default=2)
    parser.add_argument(
        "--judge-trace-dir",
        type=Path,
        default=None,
        help="Directory for raw judge request/response traces.",
    )
    parser.add_argument(
        "--force-gemini",
        "--force-judge",
        action="store_true",
        help="Ignore valid cached scores and rerun the judge for all required artifacts.",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Reuse cached Gemini scores and fail if any required score is missing.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only ingest transcript text files and write the artifact manifest.",
    )
    parser.add_argument(
        "--leave-one-episode-out",
        action="store_true",
        help=(
            "Build a leave-one-episode-out report. Candidate prompt profiles are "
            "selected using training episodes only and evaluated on the held-out "
            "episode, then pooled at artifact level."
        ),
    )
    parser.add_argument(
        "--cv-candidate-profiles",
        default="",
        help=(
            "Comma-separated prompt profiles available to the leave-one-episode-out "
            "selector. Defaults to --prompt-profile."
        ),
    )
    parser.add_argument(
        "--cv-selection-metric",
        choices=("spearman", "pearson"),
        default="spearman",
        help="Training-fold metric used to select one prompt profile per held-out episode.",
    )
    parser.add_argument(
        "--cv-bootstrap-samples",
        type=int,
        default=10000,
        help="Bootstrap resamples for Spearman confidence intervals in CV reports.",
    )
    parser.add_argument(
        "--cv-bootstrap-seed",
        type=int,
        default=42,
        help="Deterministic seed for CV bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--dimension-ensemble-profile-map",
        default="",
        help=(
            "Semicolon-separated fixed construct map, e.g. "
            "'Factualness=profile;Naturalness=profile_a+profile_b;"
            "Speaker Grammar Similarity=profile'. The specified prompt profiles are "
            "run/loaded, item scores are taken from or averaged across the mapped "
            "construct profile(s), and one fixed artifact-level report is written."
        ),
    )
    parser.add_argument(
        "--item-ensemble-profile-map",
        default="",
        help=(
            "Semicolon-separated fixed rubric-item map, e.g. "
            "'F1=profile_a+profile_b;F2=profile;...;S3=profile'. "
            "The specified prompt profiles are run/loaded, item scores are taken "
            "from or averaged across the mapped item profile(s), and one fixed "
            "artifact-level report is written."
        ),
    )
    return parser.parse_args()


def _resolve_model(*, judge_provider: str, raw_model: str) -> str:
    """Resolve provider-specific default model names."""
    requested = str(raw_model).strip()
    if requested:
        return requested
    if judge_provider == "deepseek":
        return DEFAULT_DEEPSEEK_MODEL
    if judge_provider == "vllm":
        return (
            os.environ.get("CARD_FRAMEWORK_VLLM_MODEL", "").strip()
            or os.environ.get("VLLM_MODEL", "").strip()
        )
    return DEFAULT_MODEL


def _resolve_vllm_base_url(raw_base_url: str) -> str:
    """Resolve the vLLM base URL from args or supported environment aliases."""
    return (
        str(raw_base_url).strip()
        or os.environ.get("VLLM_URL", "").strip()
        or os.environ.get("CARD_FRAMEWORK_VLLM_URL", "").strip()
        or os.environ.get("VLLM_BASE_URL", "").strip()
        or os.environ.get("CARD_FRAMEWORK_VLLM_BASE_URL", "").strip()
    )


def _resolve_vllm_api_key(raw_api_key: str) -> str:
    """Resolve the vLLM API key from args or supported environment aliases."""
    return (
        str(raw_api_key).strip()
        or os.environ.get("VLLM_API_KEY", "").strip()
        or os.environ.get("CARD_FRAMEWORK_VLLM_API_KEY", "").strip()
        or DEFAULT_VLLM_API_KEY
    )


def _normalize_max_tokens(raw_max_tokens: int | None) -> int | None:
    """Normalize the CLI max-token override; non-positive means no explicit cap."""
    if raw_max_tokens is None:
        return None
    value = int(raw_max_tokens)
    return value if value > 0 else None


def _resolve_gemini_max_output_tokens(*, api_key: str, model: str) -> int | None:
    """Return the Gemini model output limit when the API advertises it."""
    try:
        model_info = genai.Client(api_key=api_key).models.get(model=model)
    except Exception:
        return None
    raw_limit = getattr(model_info, "output_token_limit", None)
    if raw_limit is None and isinstance(model_info, dict):
        raw_limit = model_info.get("output_token_limit")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _resolve_effective_judge_max_tokens(
    *,
    judge_provider: str,
    requested_max_tokens: int | None,
    api_key: str,
    model: str,
) -> int | None:
    """Resolve the provider's default output budget without the old 8192 cap."""
    normalized = _normalize_max_tokens(requested_max_tokens)
    if normalized is not None:
        return normalized
    if requested_max_tokens is not None:
        return None
    if judge_provider == "deepseek":
        return DEEPSEEK_MAX_OUTPUT_TOKENS
    if judge_provider == "gemini":
        return _resolve_gemini_max_output_tokens(api_key=api_key, model=model)
    return None


def _slugify(value: str) -> str:
    """Return a stable short filename slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "run"


def _run_label(
    *,
    judge_provider: str,
    model: str,
    thinking_mode: str,
    deepseek_reasoning_effort: str,
    audio_mode: str,
    prompt_profile: str,
    gemini_thinking_level: str = "low",
    max_tokens: int | None = None,
    judge_seed: int | None = None,
    judge_temperature: float | None = None,
) -> str:
    """Build a filesystem-safe run label for caches and exports."""
    parts = [judge_provider, model]
    if judge_provider == "gemini":
        level = str(gemini_thinking_level).strip().lower()
        if level:
            parts.append(f"thinking_{level}")
    if judge_provider in {"deepseek", "vllm"}:
        parts.append(f"thinking_{thinking_mode}")
        if judge_provider == "deepseek" and thinking_mode == "enabled":
            parts.append(f"effort_{deepseek_reasoning_effort}")
    if audio_mode != "none":
        parts.append(f"audio_{audio_mode}")
    if prompt_profile != "default":
        parts.append(f"prompt_{prompt_profile}")
    if max_tokens is not None:
        parts.append(f"max_tokens_{int(max_tokens)}")
    if judge_seed is not None:
        parts.append(f"seed_{int(judge_seed)}")
    if judge_temperature is not None:
        temperature_text = str(float(judge_temperature)).replace(".", "p")
        parts.append(f"temp_{temperature_text}")
    return _slugify("_".join(parts))


def _read_text(path: Path) -> str:
    """Read one human-eval text artifact with tolerant UTF handling."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, content: str) -> None:
    """Write text after ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _coerce_jsonable(value: Any) -> Any:
    """Convert SDK objects into JSON-serializable builtins for traces."""
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<binary {len(value)} bytes>"
    if hasattr(value, "model_dump"):
        return _coerce_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(raw_value) for key, raw_value in value.items()}
    if isinstance(value, list | tuple):
        return [_coerce_jsonable(item) for item in value]
    return value


def _guess_audio_mime_type(path: Path) -> str:
    """Return an audio MIME type accepted by Gemini for one local file."""
    guessed, _encoding = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def _audio_paths_for_mode(
    artifact: ParticipantArtifact,
    audio_mode: str,
) -> list[tuple[str, Path]]:
    """Resolve required audio attachments for one judge input mode."""
    if audio_mode == "none":
        return []
    attachments: list[tuple[str, Path]] = []
    if audio_mode in {"source-and-summary", "human-folder"}:
        if artifact.source_audio_path is None:
            raise RuntimeError(f"Missing source audio for {artifact.artifact_key}.")
        attachments.append(("source_10m_audio_slice", artifact.source_audio_path))
    if audio_mode == "human-folder":
        for sample_path in artifact.speaker_sample_audio_paths:
            match = re.search(r"SPEAKER_\d+", sample_path.name)
            speaker_label = match.group(0) if match else sample_path.stem
            attachments.append((f"source_{speaker_label}_voice_sample", sample_path))
    if audio_mode in {"summary", "source-and-summary"}:
        if artifact.summary_audio_path is None:
            raise RuntimeError(f"Missing summary audio for {artifact.artifact_key}.")
        attachments.append(("candidate_summary_audio", artifact.summary_audio_path))
    if audio_mode == "human-folder":
        if artifact.summary_audio_path is None:
            raise RuntimeError(f"Missing summary audio for {artifact.artifact_key}.")
        attachments.append(("candidate_summary_audio", artifact.summary_audio_path))
    return attachments


def _audio_context_note(*, audio_mode: str) -> str:
    """Return prompt-facing guidance for the selected audio attachment mode."""
    if audio_mode == "none":
        return "No audio files are attached; judge only the raw transcripts."
    if audio_mode == "summary":
        return (
            "The candidate summary audio is attached after the text prompt. Use it as "
            "supporting evidence for spoken naturalness and speaker-style preservation; "
            "keep factualness grounded in the raw source transcript and summary text."
        )
    if audio_mode == "human-folder":
        return (
            "The attached audio mirrors the audio files available in the human rater "
            "folder: the source 10-minute audio slice, source speaker voice samples, "
            "and the candidate summary audio. Use these audio files alongside the "
            "source transcript and summary text exactly as folder evidence. Use the "
            "voice samples and summary audio when judging speaker recognizability, "
            "speaker-style preservation, spoken naturalness, and interjection timing; "
            "keep factualness grounded in the source transcript and summary text."
        )
    return (
        "The source 10-minute audio slice and the candidate summary audio are attached "
        "after the text prompt in that order. Use the audio as supporting evidence for "
        "spoken naturalness and speaker-style preservation; keep factualness grounded "
        "in the raw source transcript and summary text."
    )


class _GeminiAudioJudgeProvider:
    """Gemini judge provider that attaches source/summary audio files."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        audio_attachments: list[tuple[str, Path]],
        thinking_level: str,
        request_timeout_seconds: float,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model = str(model).strip()
        self.audio_attachments = tuple(audio_attachments)
        self.thinking_level = str(thinking_level).strip().lower()
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.seed = int(seed) if seed is not None else None
        self.temperature = float(temperature) if temperature is not None else None
        self._client = genai.Client(api_key=api_key)
        self.last_response_details: dict[str, Any] = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate one multimodal judge response with attached audio."""
        uploaded_files: list[Any] = []
        uploaded_metadata: list[dict[str, Any]] = []
        try:
            for label, path in self.audio_attachments:
                uploaded = self._upload_audio_file(label=label, path=path)
                uploaded_files.append(uploaded)
                uploaded_metadata.append(
                    {
                        "label": label,
                        "local_path": str(path),
                        "name": str(getattr(uploaded, "name", "")),
                        "uri": str(getattr(uploaded, "uri", "")),
                        "mime_type": str(
                            getattr(uploaded, "mime_type", "")
                            or _guess_audio_mime_type(path)
                        ),
                    }
                )

            parts = [types.Part.from_text(text=user_prompt if user_prompt else " ")]
            for uploaded in uploaded_files:
                file_uri = str(getattr(uploaded, "uri", "")).strip()
                if not file_uri:
                    raise RuntimeError("Gemini file upload did not return a file URI.")
                parts.append(
                    types.Part.from_uri(
                        file_uri=file_uri,
                        mime_type=str(getattr(uploaded, "mime_type", "") or "audio/wav"),
                    )
                )
            contents = [types.Content(role="user", parts=parts)]
            config_kwargs: dict[str, Any] = {
                "system_instruction": system_prompt.strip() or None,
                "response_mime_type": "application/json",
                "thinking_config": types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=cast(Any, self.thinking_level or "low"),
                ),
            }
            if max_tokens is not None:
                config_kwargs["max_output_tokens"] = max_tokens
            if self.seed is not None:
                config_kwargs["seed"] = self.seed
            if self.temperature is not None:
                config_kwargs["temperature"] = self.temperature
            config = types.GenerateContentConfig(**config_kwargs)
            response = self._client.models.generate_content(
                model=self.model,
                contents=cast(Any, contents),
                config=config,
            )
            content, reasoning = _extract_gemini_text_and_reasoning(response)
            self.last_response_details = {
                "provider": "google_genai_audio",
                "request": {
                    "model": self.model,
                    "audio_attachments": uploaded_metadata,
                    "contents": _coerce_jsonable(contents),
                    "config": _coerce_jsonable(config),
                },
                "response": {
                    "content": content,
                    "reasoning_content": reasoning or None,
                    "raw_response": _coerce_jsonable(response),
                },
            }
            return content.strip()
        finally:
            self._delete_uploaded_files(uploaded_files=uploaded_files)

    def _upload_audio_file(self, *, label: str, path: Path) -> Any:
        """Upload one audio file and wait until Gemini can consume it."""
        uploaded = self._client.files.upload(file=str(path))
        deadline = time.monotonic() + self.request_timeout_seconds
        current = uploaded
        while time.monotonic() < deadline:
            state_text = str(getattr(current, "state", "")).upper()
            if "FAILED" in state_text:
                raise RuntimeError(f"Gemini file upload failed for {label}: {path}")
            if "PROCESSING" not in state_text:
                return current
            time.sleep(1.0)
            current = self._client.files.get(name=str(getattr(uploaded, "name")))
        raise TimeoutError(f"Timed out waiting for Gemini file upload: {path}")

    def _delete_uploaded_files(self, *, uploaded_files: list[Any]) -> None:
        """Best-effort cleanup for files uploaded to Gemini."""
        for uploaded in uploaded_files:
            name = str(getattr(uploaded, "name", "")).strip()
            if not name:
                continue
            try:
                self._client.files.delete(name=name)
            except Exception:
                continue


def _extract_gemini_text_and_reasoning(response: Any) -> tuple[str, str]:
    """Extract visible text and provider-exposed thought text from a Gemini response."""
    content = str(getattr(response, "text", "") or "")
    reasoning_parts: list[str] = []
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            candidate_content = getattr(candidate, "content", None)
            parts = getattr(candidate_content, "parts", None)
            if not parts:
                continue
            visible_parts: list[str] = []
            for part in parts:
                text = getattr(part, "text", None)
                if getattr(part, "thought", False) and text:
                    reasoning_parts.append(str(text))
                elif text:
                    visible_parts.append(str(text))
            if not content and visible_parts:
                content = "".join(visible_parts)
    return content, "".join(reasoning_parts)


class _DeepSeekJudgeProvider:
    """DeepSeek V4 judge provider with explicit thinking-mode control."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        thinking_mode: str,
        reasoning_effort: str,
        request_timeout_seconds: float,
        temperature: float | None = None,
    ) -> None:
        self.model = str(model).strip()
        self.base_url = str(base_url).strip() or "https://api.deepseek.com"
        self.thinking_mode = str(thinking_mode).strip()
        self.reasoning_effort = str(reasoning_effort).strip()
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.temperature = float(temperature) if temperature is not None else None
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_seconds,
        )
        self.last_response_details: dict[str, Any] = {}

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate one text-only DeepSeek judge response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": self.thinking_mode}},
        }
        effective_max_tokens = (
            max_tokens if max_tokens is not None else DEEPSEEK_MAX_OUTPUT_TOKENS
        )
        create_kwargs["max_tokens"] = effective_max_tokens
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature
        if self.thinking_mode == "enabled":
            create_kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        content = str(getattr(message, "content", "") or "")
        reasoning_content = str(getattr(message, "reasoning_content", "") or "")
        self.last_response_details = {
            "provider": "deepseek",
            "request": {
                "model": self.model,
                "base_url": self.base_url,
                "thinking_mode": self.thinking_mode,
                "reasoning_effort": self.reasoning_effort
                if self.thinking_mode == "enabled"
                else None,
                "max_tokens": effective_max_tokens,
                "temperature": self.temperature,
                "seed": None,
                "seed_supported": False,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
            "response": {
                "content": content,
                "reasoning_content": reasoning_content or None,
                "raw_response": _coerce_jsonable(response),
            },
        }
        return content.strip()


class _VLLMJudgeProvider:
    """OpenAI-compatible vLLM judge provider with explicit model pinning."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        thinking_mode: str,
        request_timeout_seconds: float,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.base_url = str(base_url).strip().rstrip("/")
        if not self.base_url:
            raise RuntimeError(
                "vLLM base URL is missing. Set VLLM_URL, CARD_FRAMEWORK_VLLM_URL, "
                "VLLM_BASE_URL, or CARD_FRAMEWORK_VLLM_BASE_URL in .env."
            )
        self.api_key = str(api_key).strip() or DEFAULT_VLLM_API_KEY
        self.requested_model = str(model).strip()
        self.thinking_mode = str(thinking_mode).strip()
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.seed = int(seed) if seed is not None else None
        self.temperature = 0.0 if temperature is None else float(temperature)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_seconds,
        )
        self.max_model_len: int | None = None
        self.model = self._resolve_model(self.requested_model)
        self.last_response_details: dict[str, Any] = {}

    def _resolve_model(self, requested_model: str) -> str:
        """Resolve and validate the model id advertised by the vLLM endpoint."""
        models_response = self._client.models.list()
        available_models: list[tuple[str, int | None]] = []
        for model in getattr(models_response, "data", []) or []:
            model_id = str(getattr(model, "id", "")).strip()
            if not model_id:
                continue
            raw_max_model_len = getattr(model, "max_model_len", None)
            try:
                max_model_len = int(raw_max_model_len)
            except (TypeError, ValueError):
                max_model_len = None
            available_models.append(
                (model_id, max_model_len if max_model_len and max_model_len > 0 else None)
            )
        available_model_ids = [model_id for model_id, _max_len in available_models]
        if not available_model_ids:
            raise RuntimeError("The configured vLLM endpoint did not advertise any models.")
        if not requested_model:
            selected_model, max_model_len = available_models[0]
            self.max_model_len = max_model_len
            return selected_model
        for available_model_id, max_model_len in available_models:
            if requested_model == available_model_id:
                self.max_model_len = max_model_len
                return requested_model
        raise RuntimeError(
            f"Configured vLLM model {requested_model!r} is not advertised by the endpoint. "
            f"Available models: {', '.join(available_model_ids)}."
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate one text-only vLLM judge response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": self.thinking_mode == "enabled"
            }
        }
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "extra_body": extra_body,
            "response_format": {"type": "json_object"},
        }
        if self.seed is not None:
            create_kwargs["seed"] = self.seed
        effective_max_tokens = max_tokens
        if effective_max_tokens is None and self.max_model_len is not None:
            effective_max_tokens = max(
                1,
                self.max_model_len - VLLM_CONTEXT_TOKEN_HEADROOM,
            )
        if effective_max_tokens is not None:
            create_kwargs["max_tokens"] = effective_max_tokens

        response_format_error = ""
        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            response_format_error = str(exc)
            fallback_kwargs = dict(create_kwargs)
            fallback_kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**fallback_kwargs)
        message = response.choices[0].message
        content = str(getattr(message, "content", "") or "")
        reasoning_content = str(getattr(message, "reasoning_content", "") or "")
        resolved_content = content.strip() or reasoning_content.strip()
        self.last_response_details = {
            "provider": "vllm",
            "request": {
                "model": self.model,
                "base_url": self.base_url,
                "thinking_mode": self.thinking_mode,
                "max_tokens": effective_max_tokens,
                "max_model_len": self.max_model_len,
                "temperature": self.temperature,
                "seed": self.seed,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "response_format_error": response_format_error or None,
                "extra_body": extra_body,
            },
            "response": {
                "content": content,
                "reasoning_content": reasoning_content or None,
                "raw_response": _coerce_jsonable(response),
            },
        }
        return resolved_content


def _load_assignments(path: Path) -> dict[str, Assignment]:
    """Load participant assignments keyed by zero-padded participant id."""
    assignments: dict[str, Assignment] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            participant_id = str(row["participant_id"]).strip().zfill(2)
            assignments[participant_id] = Assignment(
                participant_id=participant_id,
                participant_email=str(row["participant_email"]).strip(),
                podcast_id=str(row["assigned_podcast"]).strip(),
                duration=str(row["assigned_duration"]).strip(),
                folder_link=str(row.get("folder_link", "")).strip(),
            )
    return assignments


def _parse_numeric_rating(raw_value: object) -> int | None:
    """Extract the leading 1-5 Likert value from one survey cell."""
    text = str(raw_value).strip()
    if not text:
        return None
    head = text.split(" ", 1)[0]
    if not head.isdigit():
        return None
    value = int(head)
    if value not in LIKERT_TO_NORMALIZED:
        return None
    return value


def _load_human_responses(
    path: Path,
    assignments: dict[str, Assignment],
) -> list[HumanResponse]:
    """Load responded human rows and compute normalized item/dimension scores."""
    responses: list[HumanResponse] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            participant_id = str(row["Participant ID (from the email)"]).strip().zfill(2)
            assignment = assignments.get(participant_id)
            if assignment is None:
                raise ValueError(f"Missing assignment for participant {participant_id}.")
            item_scores: dict[str, float] = {}
            for column_name, value in row.items():
                match = _ITEM_CODE_PATTERN.search(str(column_name))
                if match is None:
                    continue
                numeric = _parse_numeric_rating(value)
                if numeric is not None:
                    item_scores[match.group(1)] = LIKERT_TO_NORMALIZED[numeric]
            dimension_scores = _dimension_means_from_items(item_scores)
            responses.append(
                HumanResponse(
                    participant_id=participant_id,
                    participant_email=str(row["Email Address"]).strip(),
                    assignment=assignment,
                    item_scores=item_scores,
                    dimension_scores=dimension_scores,
                    comment=str(row.get(COMMENT_COLUMN, "")).strip(),
                )
            )
    return responses


def _dimension_means_from_items(item_scores: dict[str, float]) -> dict[str, float]:
    """Compute normalized dimension means from normalized item scores."""
    dimension_scores: dict[str, float] = {}
    for dimension_name, item_names in DIMENSION_ITEMS.items():
        values = [item_scores[item_name] for item_name in item_names if item_name in item_scores]
        if values:
            dimension_scores[dimension_name] = sum(values) / len(values)
    return dimension_scores


def _discover_participant_artifacts(
    participant_audio_root: Path,
    transcript_output_dir: Path,
) -> dict[str, ParticipantArtifact]:
    """Discover and copy source/summary transcripts from participant-audio folders."""
    if not participant_audio_root.is_dir():
        raise FileNotFoundError(f"Participant audio root not found: {participant_audio_root}")

    artifacts: dict[str, ParticipantArtifact] = {}
    for folder in sorted(path for path in participant_audio_root.iterdir() if path.is_dir()):
        match = _FOLDER_PATTERN.match(folder.name)
        if match is None:
            continue
        podcast_number = int(match.group("podcast"))
        duration = match.group("duration")
        podcast_id = f"Podcast {podcast_number:02d}"
        artifact_key = build_artifact_key(podcast_id, duration)
        source_candidates = sorted(folder.glob("[[]SOURCE[]]*10m audio slice transcript.txt"))
        summary_candidates = sorted(folder.glob("[[]SUMMARY[]]*summary.txt"))
        source_audio_candidates = sorted(
            path
            for path in folder.glob("[[]SOURCE[]]*10m audio slice.*")
            if path.suffix.lower() in _AUDIO_SUFFIXES
        )
        summary_audio_candidates = sorted(
            path
            for path in folder.glob("[[]SUMMARY[]]*")
            if path.suffix.lower() in _AUDIO_SUFFIXES
        )
        speaker_sample_candidates = tuple(
            sorted(
                path
                for path in folder.glob("[[]SOURCE[]]*SPEAKER_* voice sample.*")
                if path.suffix.lower() in _AUDIO_SUFFIXES
            )
        )
        if len(source_candidates) != 1:
            raise RuntimeError(
                f"Expected one 10m source transcript in {folder}, "
                f"found {len(source_candidates)}."
            )
        if len(summary_candidates) != 1:
            raise RuntimeError(
                f"Expected one summary transcript in {folder}, "
                f"found {len(summary_candidates)}."
            )
        if len(source_audio_candidates) != 1:
            raise RuntimeError(
                f"Expected one 10m source audio file in {folder}, "
                f"found {len(source_audio_candidates)}."
            )
        if len(summary_audio_candidates) != 1:
            raise RuntimeError(
                f"Expected one summary audio file in {folder}, "
                f"found {len(summary_audio_candidates)}."
            )
        local_dir = transcript_output_dir / artifact_key
        local_source_path = local_dir / "source_10m_audio_slice_transcript.txt"
        local_summary_path = local_dir / "summary_transcript.txt"
        _write_text(local_source_path, _read_text(source_candidates[0]))
        _write_text(local_summary_path, _read_text(summary_candidates[0]))
        artifacts[artifact_key] = ParticipantArtifact(
            artifact_key=artifact_key,
            podcast_id=podcast_id,
            duration=duration,
            source_transcript_path=source_candidates[0],
            summary_transcript_path=summary_candidates[0],
            source_audio_path=source_audio_candidates[0],
            summary_audio_path=summary_audio_candidates[0],
            speaker_sample_audio_paths=speaker_sample_candidates,
            local_source_transcript_path=local_source_path,
            local_summary_transcript_path=local_summary_path,
        )
    if not artifacts:
        raise RuntimeError(f"No Podcast_XX_<duration> folders found in {participant_audio_root}.")
    return artifacts


def _json_dump(path: Path, payload: Any) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_artifact_manifest(
    path: Path,
    artifacts: dict[str, ParticipantArtifact],
) -> None:
    """Write a manifest for the ingested transcript artifacts."""
    _json_dump(
        path,
        {
            artifact_key: {
                "podcast_id": artifact.podcast_id,
                "duration": artifact.duration,
                "target_duration_label": artifact.target_duration_label,
                "source_transcript_path": str(artifact.source_transcript_path),
                "summary_transcript_path": str(artifact.summary_transcript_path),
                "source_audio_path": str(artifact.source_audio_path)
                if artifact.source_audio_path is not None
                else None,
                "summary_audio_path": str(artifact.summary_audio_path)
                if artifact.summary_audio_path is not None
                else None,
                "speaker_sample_audio_paths": [
                    str(path) for path in artifact.speaker_sample_audio_paths
                ],
                "local_source_transcript_path": str(artifact.local_source_transcript_path),
                "local_summary_transcript_path": str(artifact.local_summary_transcript_path),
            }
            for artifact_key, artifact in sorted(artifacts.items())
        },
    )


def _optional_int_cache_value_matches(raw_value: object, expected: int | None) -> bool:
    """Return whether an optional cached integer setting matches."""
    if expected is None:
        return raw_value in (None, "")
    try:
        return int(cast(Any, raw_value)) == int(expected)
    except (TypeError, ValueError):
        return False


def _optional_float_cache_value_matches(raw_value: object, expected: float | None) -> bool:
    """Return whether an optional cached float setting matches."""
    if expected is None:
        return raw_value in (None, "")
    try:
        return float(cast(Any, raw_value)) == float(expected)
    except (TypeError, ValueError):
        return False


def _cache_entry_is_valid(
    entry: Any,
    *,
    judge_provider: str,
    model: str,
    vllm_base_url: str = "",
    audio_mode: str,
    thinking_mode: str,
    gemini_thinking_level: str = "low",
    deepseek_reasoning_effort: str,
    prompt_profile: str,
    calibration_fingerprint: str = "",
    requested_max_tokens: int | None = None,
    judge_seed: int | None = None,
    judge_temperature: float | None = None,
) -> bool:
    """Return whether one cached score entry matches the current judge contract."""
    if not isinstance(entry, dict):
        return False
    if entry.get("schema_version") != GEMINI_SCORE_CACHE_SCHEMA_VERSION:
        return False
    if entry.get("judge_prompt_version") != JUDGE_PROMPT_CACHE_VERSION:
        return False
    if str(entry.get("judge_provider", "")).strip() != judge_provider:
        return False
    if str(entry.get("model", "")).strip() != model:
        return False
    if (
        judge_provider == "vllm"
        and str(entry.get("vllm_base_url", "")).strip().rstrip("/")
        != str(vllm_base_url).strip().rstrip("/")
    ):
        return False
    if str(entry.get("audio_mode", "")).strip() != audio_mode:
        return False
    entry_prompt_profile = entry.get(
        "prompt_profile",
        prompt_profile if prompt_profile != "default" else "default",
    )
    if str(entry_prompt_profile).strip() != prompt_profile:
        return False
    if not _optional_int_cache_value_matches(
        entry.get("requested_max_tokens"),
        requested_max_tokens,
    ):
        return False
    if not _optional_int_cache_value_matches(entry.get("judge_seed"), judge_seed):
        return False
    if not _optional_float_cache_value_matches(
        entry.get("judge_temperature"),
        judge_temperature,
    ):
        return False
    if (
        prompt_profile == "in_sample_gold_anchor"
        and str(entry.get("calibration_fingerprint", "")).strip()
        != calibration_fingerprint
    ):
        return False
    if judge_provider in {"deepseek", "vllm"} and str(entry.get("thinking_mode", "")).strip() != thinking_mode:
        return False
    if (
        judge_provider == "gemini"
        and str(entry.get("gemini_thinking_level", "low")).strip().lower()
        != str(gemini_thinking_level).strip().lower()
    ):
        return False
    if (
        judge_provider == "deepseek"
        and thinking_mode == "enabled"
        and str(entry.get("deepseek_reasoning_effort", "")).strip()
        != deepseek_reasoning_effort
    ):
        return False
    item_scores = entry.get("item_scores")
    if not isinstance(item_scores, dict):
        return False
    if any(item_name not in item_scores for item_name in ITEM_NAMES):
        return False
    dimension_scores = entry.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        return False
    if any(dimension_name not in dimension_scores for dimension_name in DIMENSION_NAMES):
        return False
    trace_path = str(entry.get("trace_path", "")).strip()
    return bool(trace_path) and Path(trace_path).is_file()


def _normalize_cached_score_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive dimension/composite scores from cached item scores."""
    item_scores = {
        item_name: float(entry["item_scores"][item_name]) for item_name in ITEM_NAMES
    }
    dimension_scores = _dimension_means_from_items(item_scores)
    normalized_entry = dict(entry)
    normalized_entry["item_scores"] = item_scores
    normalized_entry["dimension_scores"] = dimension_scores
    normalized_entry["composite"] = _mean(
        [dimension_scores[dimension_name] for dimension_name in DIMENSION_NAMES]
    )
    return normalized_entry


def _require_complete_item_scores(
    item_scores: dict[str, float] | None,
    *,
    artifact_key: str,
) -> dict[str, float]:
    """Validate and return a complete set of nine normalized judge item scores."""
    if item_scores is None:
        raise RuntimeError(f"Judge did not return item_scores for {artifact_key}.")
    missing_items = [item_name for item_name in ITEM_NAMES if item_name not in item_scores]
    if missing_items:
        raise RuntimeError(
            f"Judge omitted item_scores for {artifact_key}: "
            + ", ".join(missing_items)
        )
    return {item_name: float(item_scores[item_name]) for item_name in ITEM_NAMES}


def _write_judge_trace(
    *,
    trace_dir: Path,
    artifact: ParticipantArtifact,
    judge_provider: str,
    model: str,
    vllm_base_url: str,
    thinking_mode: str,
    gemini_thinking_level: str,
    deepseek_reasoning_effort: str,
    audio_mode: str,
    prompt_profile: str,
    calibration_fingerprint: str,
    requested_max_tokens: int | None,
    judge_seed: int | None,
    judge_temperature: float | None,
    audio_attachments: list[tuple[str, Path]],
    provider: Any,
    scores: dict[str, float],
    item_scores: dict[str, float],
    confidence: float | None,
    rationale: str,
) -> Path:
    """Persist raw judge request/response details for audit."""
    trace_path = trace_dir / f"{artifact.artifact_key}.json"
    _json_dump(
        trace_path,
        {
            "schema_version": GEMINI_SCORE_CACHE_SCHEMA_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_CACHE_VERSION,
            "artifact_key": artifact.artifact_key,
            "podcast_id": artifact.podcast_id,
            "duration": artifact.duration,
            "target_duration_label": artifact.target_duration_label,
            "judge_provider": judge_provider,
            "model": model,
            "vllm_base_url": vllm_base_url if judge_provider == "vllm" else None,
            "thinking_mode": thinking_mode if judge_provider in {"deepseek", "vllm"} else None,
            "gemini_thinking_level": gemini_thinking_level
            if judge_provider == "gemini"
            else None,
            "deepseek_reasoning_effort": deepseek_reasoning_effort
            if judge_provider == "deepseek" and thinking_mode == "enabled"
            else None,
            "audio_mode": audio_mode,
            "prompt_profile": prompt_profile,
            "requested_max_tokens": requested_max_tokens,
            "judge_seed": judge_seed,
            "judge_temperature": judge_temperature,
            "calibration_fingerprint": calibration_fingerprint
            if prompt_profile == "in_sample_gold_anchor"
            else None,
            "audio_attachments": [
                {"label": label, "local_path": str(path)}
                for label, path in audio_attachments
            ],
            "source_transcript_path": str(artifact.local_source_transcript_path),
            "summary_transcript_path": str(artifact.local_summary_transcript_path),
            "trace_note": (
                "Raw source transcript and summary text are preserved in the Gemini "
                "or DeepSeek request exactly as fed by this workflow. Provider-returned "
                "reasoning_content is recorded when the SDK exposes it; otherwise "
                "the judge rationale is the observable reasoning summary."
            ),
            "parsed_item_scores": item_scores,
            "parsed_dimension_scores": scores,
            "confidence": confidence,
            "rationale": rationale,
            "provider_last_response_details": provider.last_response_details,
        },
    )
    return trace_path


def _write_judge_failure_trace(
    *,
    trace_dir: Path,
    artifact: ParticipantArtifact,
    judge_provider: str,
    model: str,
    vllm_base_url: str,
    thinking_mode: str,
    gemini_thinking_level: str,
    deepseek_reasoning_effort: str,
    audio_mode: str,
    prompt_profile: str,
    calibration_fingerprint: str,
    requested_max_tokens: int | None,
    judge_seed: int | None,
    judge_temperature: float | None,
    audio_attachments: list[tuple[str, Path]],
    provider: Any,
    attempt: int,
    error_message: str,
) -> Path:
    """Persist raw provider details for a failed judge attempt."""
    trace_path = trace_dir / f"{artifact.artifact_key}.attempt_{attempt}_error.json"
    _json_dump(
        trace_path,
        {
            "schema_version": GEMINI_SCORE_CACHE_SCHEMA_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_CACHE_VERSION,
            "artifact_key": artifact.artifact_key,
            "podcast_id": artifact.podcast_id,
            "duration": artifact.duration,
            "target_duration_label": artifact.target_duration_label,
            "judge_provider": judge_provider,
            "model": model,
            "vllm_base_url": vllm_base_url if judge_provider == "vllm" else None,
            "thinking_mode": thinking_mode if judge_provider in {"deepseek", "vllm"} else None,
            "gemini_thinking_level": gemini_thinking_level
            if judge_provider == "gemini"
            else None,
            "deepseek_reasoning_effort": deepseek_reasoning_effort
            if judge_provider == "deepseek" and thinking_mode == "enabled"
            else None,
            "audio_mode": audio_mode,
            "prompt_profile": prompt_profile,
            "requested_max_tokens": requested_max_tokens,
            "judge_seed": judge_seed,
            "judge_temperature": judge_temperature,
            "calibration_fingerprint": calibration_fingerprint
            if prompt_profile == "in_sample_gold_anchor"
            else None,
            "audio_attachments": [
                {"label": label, "local_path": str(path)}
                for label, path in audio_attachments
            ],
            "attempt": attempt,
            "error_message": error_message,
            "source_transcript_path": str(artifact.local_source_transcript_path),
            "summary_transcript_path": str(artifact.local_summary_transcript_path),
            "provider_last_response_details": provider.last_response_details,
        },
    )
    return trace_path


def _load_judge_scores(
    artifacts: dict[str, ParticipantArtifact],
    cache_path: Path,
    *,
    model: str,
    judge_provider: str,
    audio_mode: str,
    thinking_mode: str,
    gemini_thinking_level: str,
    deepseek_base_url: str,
    vllm_base_url: str,
    vllm_api_key: str,
    deepseek_reasoning_effort: str,
    prompt_profile: str,
    request_timeout_seconds: float,
    max_tokens: int | None,
    judge_seed: int | None,
    judge_temperature: float | None,
    max_judge_attempts: int,
    trace_dir: Path,
    skip_gemini: bool,
    force_gemini: bool,
    calibration_item_scores_by_artifact: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Load cached judge scores and run missing artifact judgments."""
    calibration_fingerprint = _calibration_fingerprint(
        calibration_item_scores_by_artifact
    )
    cached: dict[str, Any] = {}
    if cache_path.is_file() and not force_gemini:
        raw_cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = {
            artifact_key: _normalize_cached_score_entry(entry)
            for artifact_key, entry in raw_cached.items()
            if artifact_key in artifacts
            and _cache_entry_is_valid(
                entry,
                judge_provider=judge_provider,
                model=model,
                vllm_base_url=vllm_base_url,
                audio_mode=audio_mode,
                thinking_mode=thinking_mode,
                gemini_thinking_level=gemini_thinking_level,
                deepseek_reasoning_effort=deepseek_reasoning_effort,
                prompt_profile=prompt_profile,
                calibration_fingerprint=calibration_fingerprint,
                requested_max_tokens=max_tokens,
                judge_seed=judge_seed,
                judge_temperature=judge_temperature,
            )
        }

    missing_keys = [key for key in sorted(artifacts) if key not in cached]
    if missing_keys and skip_gemini:
        raise RuntimeError(
            "Missing cached judge scores for: " + ", ".join(missing_keys)
        )
    if not missing_keys:
        _json_dump(cache_path, cached)
        return cached

    load_repo_dotenv(REPO_ROOT)
    if judge_provider in {"deepseek", "vllm"} and audio_mode != "none":
        raise RuntimeError(
            f"{judge_provider} is text-only in this workflow; use --audio-mode none."
        )
    if judge_provider == "deepseek" and judge_seed is not None:
        raise RuntimeError(
            "DeepSeek's documented Chat Completion API does not expose a seed "
            "parameter. For reproducible DeepSeek reporting, use "
            "--judge-temperature 0 with fixed --max-tokens and preserve the "
            "resulting cache/trace files instead of force-rerunning."
        )
    if judge_provider == "vllm":
        api_key = vllm_api_key
    else:
        api_env_name = "DEEPSEEK_API_KEY" if judge_provider == "deepseek" else "GEMINI_API_KEY"
        api_key = str(os.environ.get(api_env_name, "")).strip()
        if judge_provider == "gemini" and not api_key:
            api_key = str(os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError(f"{api_env_name} is not available in the environment.")
    effective_max_tokens = _resolve_effective_judge_max_tokens(
        judge_provider=judge_provider,
        requested_max_tokens=max_tokens,
        api_key=api_key,
        model=model,
    )
    for artifact_key in missing_keys:
        artifact = artifacts[artifact_key]
        audio_attachments = _audio_paths_for_mode(artifact, audio_mode)
        audio_context_note = _audio_context_note(audio_mode=audio_mode)
        scoring_profile_note = PROMPT_PROFILE_NOTES[prompt_profile]
        if prompt_profile == "in_sample_gold_anchor":
            calibration_scores = (calibration_item_scores_by_artifact or {}).get(artifact_key)
            if calibration_scores is None:
                raise RuntimeError(
                    f"Missing in-sample calibration scores for {artifact_key}."
                )
            scoring_profile_note += _in_sample_gold_anchor_note(
                artifact_key=artifact_key,
                item_scores=calibration_scores,
            )
        provider: Any
        if judge_provider == "deepseek":
            provider = _DeepSeekJudgeProvider(
                api_key=api_key,
                model=model,
                base_url=deepseek_base_url,
                thinking_mode=thinking_mode,
                reasoning_effort=deepseek_reasoning_effort,
                request_timeout_seconds=request_timeout_seconds,
                temperature=judge_temperature,
            )
        elif judge_provider == "vllm":
            provider = _VLLMJudgeProvider(
                api_key=api_key,
                model=model,
                base_url=vllm_base_url,
                thinking_mode=thinking_mode,
                request_timeout_seconds=request_timeout_seconds,
                seed=judge_seed,
                temperature=judge_temperature,
            )
        elif judge_provider == "gemini":
            provider = _GeminiAudioJudgeProvider(
                api_key=api_key,
                model=model,
                audio_attachments=audio_attachments,
                thinking_level=gemini_thinking_level,
                request_timeout_seconds=request_timeout_seconds,
                seed=judge_seed,
                temperature=judge_temperature,
            )
        else:
            provider = GoogleGenAIProvider(
                api_key=api_key,
                model=model,
                enable_thinking=True,
                thinking_level=gemini_thinking_level,
                request_timeout_seconds=request_timeout_seconds,
            )
        judge = ConversationSummaryJudge(
            judge_llm=provider,
            config=ConversationSummaryJudgeConfig(max_tokens=effective_max_tokens),
        )
        source_text = _read_text(artifact.local_source_transcript_path)
        summary_xml = _read_text(artifact.local_summary_transcript_path)
        last_error = ""
        for attempt in range(1, max(1, int(max_judge_attempts)) + 1):
            result = judge.evaluate(
                source_text=source_text,
                summary_xml=summary_xml,
                transcript_per_speaker_messages={},
                summary_per_speaker_messages={},
                speaker_alias_map={},
                speaker_notes=[
                    "The raw XML-like source transcript and candidate summary are "
                    "intentionally preserved exactly as the human raters saw them."
                ],
                target_duration_label=artifact.target_duration_label,
                audio_context_note=audio_context_note,
                scoring_profile_note=scoring_profile_note,
            )
            if result.status == "ok" and result.scores is not None:
                try:
                    item_scores = _require_complete_item_scores(
                        result.item_scores,
                        artifact_key=artifact_key,
                    )
                except RuntimeError as exc:
                    last_error = str(exc)
                    _write_judge_failure_trace(
                        trace_dir=trace_dir,
                        artifact=artifact,
                        judge_provider=judge_provider,
                        model=model,
                        vllm_base_url=vllm_base_url,
                        thinking_mode=thinking_mode,
                        gemini_thinking_level=gemini_thinking_level,
                        deepseek_reasoning_effort=deepseek_reasoning_effort,
                        audio_mode=audio_mode,
                        prompt_profile=prompt_profile,
                        calibration_fingerprint=calibration_fingerprint,
                        requested_max_tokens=max_tokens,
                        judge_seed=judge_seed,
                        judge_temperature=judge_temperature,
                        audio_attachments=audio_attachments,
                        provider=provider,
                        attempt=attempt,
                        error_message=last_error,
                    )
                    continue
                dimension_scores = _dimension_means_from_items(item_scores)
                trace_path = _write_judge_trace(
                    trace_dir=trace_dir,
                    artifact=artifact,
                    judge_provider=judge_provider,
                    model=model,
                    vllm_base_url=vllm_base_url,
                    thinking_mode=thinking_mode,
                    gemini_thinking_level=gemini_thinking_level,
                    deepseek_reasoning_effort=deepseek_reasoning_effort,
                    audio_mode=audio_mode,
                    prompt_profile=prompt_profile,
                    calibration_fingerprint=calibration_fingerprint,
                    requested_max_tokens=max_tokens,
                    judge_seed=judge_seed,
                    judge_temperature=judge_temperature,
                    audio_attachments=audio_attachments,
                    provider=provider,
                    scores=dimension_scores,
                    item_scores=item_scores,
                    confidence=result.confidence,
                    rationale=result.rationale,
                )
                break
            last_error = result.error_message or "Unknown judge error."
            _write_judge_failure_trace(
                trace_dir=trace_dir,
                artifact=artifact,
                judge_provider=judge_provider,
                model=model,
                vllm_base_url=vllm_base_url,
                thinking_mode=thinking_mode,
                gemini_thinking_level=gemini_thinking_level,
                deepseek_reasoning_effort=deepseek_reasoning_effort,
                audio_mode=audio_mode,
                prompt_profile=prompt_profile,
                calibration_fingerprint=calibration_fingerprint,
                requested_max_tokens=max_tokens,
                judge_seed=judge_seed,
                judge_temperature=judge_temperature,
                audio_attachments=audio_attachments,
                provider=provider,
                attempt=attempt,
                error_message=last_error,
            )
        else:
            raise RuntimeError(f"{judge_provider} judge failed for {artifact_key}: {last_error}")
        cached[artifact_key] = {
            "schema_version": GEMINI_SCORE_CACHE_SCHEMA_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_CACHE_VERSION,
            "judge_provider": judge_provider,
            "podcast_id": artifact.podcast_id,
            "duration": artifact.duration,
            "target_duration_label": artifact.target_duration_label,
            "model": model,
            "vllm_base_url": vllm_base_url if judge_provider == "vllm" else None,
            "thinking_mode": thinking_mode if judge_provider in {"deepseek", "vllm"} else None,
            "gemini_thinking_level": gemini_thinking_level
            if judge_provider == "gemini"
            else None,
            "deepseek_reasoning_effort": deepseek_reasoning_effort
            if judge_provider == "deepseek" and thinking_mode == "enabled"
            else None,
            "audio_mode": audio_mode,
            "prompt_profile": prompt_profile,
            "requested_max_tokens": max_tokens,
            "judge_seed": judge_seed,
            "judge_temperature": judge_temperature,
            "calibration_fingerprint": calibration_fingerprint
            if prompt_profile == "in_sample_gold_anchor"
            else None,
            "item_scores": item_scores,
            "dimension_scores": dimension_scores,
            "composite": _mean(
                [dimension_scores[dimension_name] for dimension_name in DIMENSION_NAMES]
            ),
            "confidence": result.confidence,
            "rationale": result.rationale,
            "trace_path": str(trace_path),
        }
        _json_dump(cache_path, cached)
    return cached


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean for a non-empty float list."""
    if not values:
        raise ValueError("Cannot average an empty value list.")
    return sum(values) / len(values)


def _average_ranks(values: list[float]) -> list[float]:
    """Return average ranks with tie handling."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for original_index, _value in indexed[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return Pearson correlation, or None when undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return None
    return numerator / (denominator_x * denominator_y)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Return Spearman rank correlation, or None when undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _round_or_none(value: float | None) -> float | None:
    """Round one optional correlation value for reports."""
    return None if value is None else round(value, 4)


def _correlation_entry(human_values: list[float], judge_values: list[float]) -> dict[str, Any]:
    """Build one correlation report entry."""
    return {
        "n": len(human_values),
        "pearson": _round_or_none(_pearson(human_values, judge_values)),
        "spearman": _round_or_none(_spearman(human_values, judge_values)),
        "human_mean": round(_mean(human_values), 4) if human_values else None,
        "judge_mean": round(_mean(judge_values), 4) if judge_values else None,
    }


def _round_ci(value: tuple[float, float] | None) -> list[float] | None:
    """Round a confidence interval for JSON reports."""
    if value is None:
        return None
    return [round(value[0], 4), round(value[1], 4)]


def _pearson_fisher_z_ci(
    r_value: float | None,
    sample_count: int,
) -> tuple[float, float] | None:
    """Return an approximate Pearson 95% CI from Fisher's z transform."""
    if r_value is None or sample_count <= 3:
        return None
    bounded = max(-0.999999, min(0.999999, r_value))
    # IEEE citation: Fisher's z transform CI with standard error 1/sqrt(n - 3) [2].
    z_value = math.atanh(bounded)
    standard_error = 1.0 / math.sqrt(sample_count - 3)
    z_critical = 1.959963984540054
    return (
        math.tanh(z_value - z_critical * standard_error),
        math.tanh(z_value + z_critical * standard_error),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _bootstrap_spearman_ci(
    human_values: list[float],
    judge_values: list[float],
    *,
    groups: list[str] | None,
    sample_count: int,
    seed: int,
) -> tuple[float, float] | None:
    """Return a percentile bootstrap 95% CI for Spearman correlation."""
    if len(human_values) != len(judge_values) or len(human_values) < 2:
        return None
    if sample_count <= 0:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    indexes = list(range(len(human_values)))
    group_to_indexes: dict[str, list[int]] = defaultdict(list)
    group_labels = list(dict.fromkeys(groups or []))
    if groups is not None:
        for index, group in enumerate(groups):
            group_to_indexes[group].append(index)

    for _ in range(sample_count):
        if groups is None:
            sampled_indexes = [rng.choice(indexes) for _index in indexes]
        else:
            sampled_groups = [rng.choice(group_labels) for _group in group_labels]
            sampled_indexes = [
                index for group in sampled_groups for index in group_to_indexes[group]
            ]
        sampled_human = [human_values[index] for index in sampled_indexes]
        sampled_judge = [judge_values[index] for index in sampled_indexes]
        rho = _spearman(sampled_human, sampled_judge)
        if rho is not None:
            values.append(rho)
    if not values:
        return None
    values.sort()
    # IEEE citation: Percentile bootstrap interval following bootstrap resampling [3].
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _correlation_entry_with_uncertainty(
    human_values: list[float],
    judge_values: list[float],
    *,
    groups: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build a correlation entry with small-sample uncertainty estimates."""
    entry = _correlation_entry(human_values, judge_values)
    pearson_value = _pearson(human_values, judge_values)
    entry["pearson_95ci_fisher_z"] = _round_ci(
        _pearson_fisher_z_ci(pearson_value, len(human_values))
    )
    entry["spearman_95ci_bootstrap_percentile"] = _round_ci(
        _bootstrap_spearman_ci(
            human_values,
            judge_values,
            groups=None,
            sample_count=bootstrap_samples,
            seed=bootstrap_seed,
        )
    )
    entry["spearman_95ci_episode_cluster_bootstrap_percentile"] = _round_ci(
        _bootstrap_spearman_ci(
            human_values,
            judge_values,
            groups=groups,
            sample_count=bootstrap_samples,
            seed=bootstrap_seed + 1009,
        )
    )
    return entry


def _human_composite_from_items(item_scores: dict[str, float]) -> float | None:
    """Return the nine-item text-rubric composite for one human response."""
    values = [item_scores[item_name] for item_name in ITEM_NAMES if item_name in item_scores]
    if len(values) != len(ITEM_NAMES):
        return None
    return _mean(values)


def _build_correlation_report(
    responses: list[HumanResponse],
    gemini_scores: dict[str, Any],
) -> dict[str, Any]:
    """Build item-level and mean-level human/Gemini correlations."""
    matched = [response for response in responses if response.assignment.artifact_key in gemini_scores]
    response_level_items: dict[str, Any] = {}
    for item_name in ITEM_NAMES:
        human_values = []
        gemini_values = []
        for response in matched:
            if item_name not in response.item_scores:
                continue
            human_values.append(response.item_scores[item_name])
            gemini_values.append(
                float(gemini_scores[response.assignment.artifact_key]["item_scores"][item_name])
            )
        response_level_items[item_name] = _correlation_entry(human_values, gemini_values)

    response_level_dimensions: dict[str, Any] = {}
    for dimension_name in DIMENSION_NAMES:
        human_values = []
        gemini_values = []
        for response in matched:
            if dimension_name not in response.dimension_scores:
                continue
            human_values.append(response.dimension_scores[dimension_name])
            gemini_values.append(
                float(
                    gemini_scores[response.assignment.artifact_key]["dimension_scores"][
                        dimension_name
                    ]
                )
            )
        response_level_dimensions[dimension_name] = _correlation_entry(
            human_values,
            gemini_values,
        )
    human_composites = []
    gemini_composites = []
    for response in matched:
        human_composite = _human_composite_from_items(response.item_scores)
        if human_composite is None:
            continue
        human_composites.append(human_composite)
        gemini_composites.append(float(gemini_scores[response.assignment.artifact_key]["composite"]))
    response_level_dimensions["Composite"] = _correlation_entry(
        human_composites,
        gemini_composites,
    )

    grouped = _group_responses_by_artifact(matched)
    artifact_level_items: dict[str, Any] = {}
    for item_name in ITEM_NAMES:
        human_values = []
        gemini_values = []
        for artifact_key, rows in sorted(grouped.items()):
            values = [row.item_scores[item_name] for row in rows if item_name in row.item_scores]
            if not values:
                continue
            human_values.append(_mean(values))
            gemini_values.append(float(gemini_scores[artifact_key]["item_scores"][item_name]))
        artifact_level_items[item_name] = _correlation_entry(human_values, gemini_values)

    artifact_level_dimensions: dict[str, Any] = {}
    for dimension_name in DIMENSION_NAMES:
        human_values = []
        gemini_values = []
        for artifact_key, rows in sorted(grouped.items()):
            values = [
                row.dimension_scores[dimension_name]
                for row in rows
                if dimension_name in row.dimension_scores
            ]
            if not values:
                continue
            human_values.append(_mean(values))
            gemini_values.append(
                float(gemini_scores[artifact_key]["dimension_scores"][dimension_name])
            )
        artifact_level_dimensions[dimension_name] = _correlation_entry(
            human_values,
            gemini_values,
        )
    artifact_human_composites = []
    artifact_gemini_composites = []
    for artifact_key, rows in sorted(grouped.items()):
        composites = [
            composite
            for row in rows
            if (composite := _human_composite_from_items(row.item_scores)) is not None
        ]
        if not composites:
            continue
        artifact_human_composites.append(_mean(composites))
        artifact_gemini_composites.append(float(gemini_scores[artifact_key]["composite"]))
    artifact_level_dimensions["Composite"] = _correlation_entry(
        artifact_human_composites,
        artifact_gemini_composites,
    )

    return {
        "matched_response_count": len(matched),
        "artifact_count_with_responses": len(grouped),
        "response_level": {
            "items": response_level_items,
            "dimension_means": response_level_dimensions,
        },
        "artifact_level": {
            "items": artifact_level_items,
            "dimension_means": artifact_level_dimensions,
        },
    }


def _group_responses_by_artifact(
    responses: list[HumanResponse],
) -> dict[str, list[HumanResponse]]:
    """Group human responses by assigned artifact."""
    grouped: dict[str, list[HumanResponse]] = defaultdict(list)
    for response in responses:
        grouped[response.assignment.artifact_key].append(response)
    return dict(grouped)


def _artifact_mean_item_scores(
    responses: list[HumanResponse],
) -> dict[str, dict[str, float]]:
    """Return human artifact-level item means keyed by artifact."""
    grouped = _group_responses_by_artifact(responses)
    means: dict[str, dict[str, float]] = {}
    for artifact_key, rows in grouped.items():
        means[artifact_key] = {}
        for item_name in ITEM_NAMES:
            values = [row.item_scores[item_name] for row in rows if item_name in row.item_scores]
            if values:
                means[artifact_key][item_name] = round(_mean(values), 4)
    return means


def _episode_group_from_artifact_key(artifact_key: str) -> str:
    """Return the episode group id from a canonical artifact key."""
    match = re.match(r"^Podcast_(?P<episode>\d{2})_(?:30s|1m|5m)$", artifact_key)
    if match is None:
        raise ValueError(f"Cannot derive episode group from artifact key: {artifact_key}")
    return match.group("episode")


def _artifact_level_human_scores(
    responses: list[HumanResponse],
) -> dict[str, dict[str, Any]]:
    """Return artifact-level human mean item, dimension, and composite scores."""
    grouped = _group_responses_by_artifact(responses)
    rows: dict[str, dict[str, Any]] = {}
    for artifact_key, artifact_responses in grouped.items():
        item_scores: dict[str, float] = {}
        for item_name in ITEM_NAMES:
            values = [
                response.item_scores[item_name]
                for response in artifact_responses
                if item_name in response.item_scores
            ]
            if values:
                item_scores[item_name] = _mean(values)
        dimension_scores = _dimension_means_from_items(item_scores)
        composites = [
            composite
            for response in artifact_responses
            if (composite := _human_composite_from_items(response.item_scores))
            is not None
        ]
        rows[artifact_key] = {
            "artifact_key": artifact_key,
            "episode_group": _episode_group_from_artifact_key(artifact_key),
            "podcast_id": artifact_responses[0].assignment.podcast_id,
            "duration": artifact_responses[0].assignment.duration,
            "human_response_count": len(artifact_responses),
            "item_scores": item_scores,
            "dimension_scores": dimension_scores,
            "composite": _mean(composites) if composites else None,
        }
    return rows


def _parse_cv_candidate_profiles(raw_profiles: str, default_profile: str) -> list[str]:
    """Parse and validate non-leaking CV candidate prompt profiles."""
    raw_items = (
        [item.strip() for item in raw_profiles.split(",")]
        if raw_profiles.strip()
        else [default_profile]
    )
    profiles: list[str] = []
    for profile in raw_items:
        if not profile:
            continue
        if profile not in PROMPT_PROFILE_NOTES:
            raise ValueError(f"Unknown prompt profile for CV: {profile}")
        if profile == "in_sample_gold_anchor":
            raise ValueError(
                "in_sample_gold_anchor is not allowed in leave-one-episode-out CV."
            )
        if profile not in profiles:
            profiles.append(profile)
    if not profiles:
        raise ValueError("At least one CV candidate prompt profile is required.")
    return profiles


def _canonical_dimension_name(raw_name: str) -> str:
    """Return a canonical rubric dimension name from a CLI alias."""
    normalized = re.sub(r"[^a-z0-9]+", "", raw_name.lower())
    aliases = {
        "factualness": "Factualness",
        "factual": "Factualness",
        "naturalness": "Naturalness",
        "natural": "Naturalness",
        "speakergrammarsimilarity": "Speaker Grammar Similarity",
        "speakergrammar": "Speaker Grammar Similarity",
        "speakersimilarity": "Speaker Grammar Similarity",
        "speaker": "Speaker Grammar Similarity",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown dimension in ensemble map: {raw_name}")
    return aliases[normalized]


def _parse_dimension_ensemble_profile_map(raw_map: str) -> dict[str, tuple[str, ...]]:
    """Parse a fixed construct-to-prompt-profile map for ensemble reports."""
    dimension_profile_map: dict[str, tuple[str, ...]] = {}
    for raw_part in raw_map.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "Dimension ensemble entries must use Dimension=prompt_profile syntax."
            )
        raw_dimension, raw_profile = part.split("=", 1)
        dimension_name = _canonical_dimension_name(raw_dimension.strip())
        profiles = tuple(
            profile.strip() for profile in raw_profile.split("+") if profile.strip()
        )
        if not profiles:
            raise ValueError(f"No prompt profile configured for {dimension_name}.")
        for profile in profiles:
            if profile not in PROMPT_PROFILE_NOTES:
                raise ValueError(f"Unknown prompt profile in ensemble map: {profile}")
            if profile == "in_sample_gold_anchor":
                raise ValueError(
                    "in_sample_gold_anchor is not allowed in fixed dimension ensembles."
                )
        dimension_profile_map[dimension_name] = profiles
    missing = [dimension_name for dimension_name in DIMENSION_NAMES if dimension_name not in dimension_profile_map]
    if missing:
        raise ValueError(
            "Dimension ensemble map is missing: " + ", ".join(missing)
        )
    return dimension_profile_map


def _parse_item_ensemble_profile_map(raw_map: str) -> dict[str, tuple[str, ...]]:
    """Parse a fixed rubric-item-to-prompt-profile map for ensemble reports."""
    item_profile_map: dict[str, tuple[str, ...]] = {}
    for raw_part in raw_map.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("Item ensemble entries must use Item=prompt_profile syntax.")
        raw_item, raw_profile = part.split("=", 1)
        item_name = raw_item.strip().upper()
        if item_name not in ITEM_NAMES:
            raise ValueError(f"Unknown item in ensemble map: {raw_item}")
        profiles = tuple(
            profile.strip() for profile in raw_profile.split("+") if profile.strip()
        )
        if not profiles:
            raise ValueError(f"No prompt profile configured for {item_name}.")
        for profile in profiles:
            if profile not in PROMPT_PROFILE_NOTES:
                raise ValueError(f"Unknown prompt profile in ensemble map: {profile}")
            if profile == "in_sample_gold_anchor":
                raise ValueError(
                    "in_sample_gold_anchor is not allowed in fixed item ensembles."
                )
        item_profile_map[item_name] = profiles
    missing = [item_name for item_name in ITEM_NAMES if item_name not in item_profile_map]
    if missing:
        raise ValueError("Item ensemble map is missing: " + ", ".join(missing))
    return item_profile_map


def _build_dimension_ensemble_scores(
    *,
    candidate_scores_by_profile: dict[str, dict[str, Any]],
    dimension_profile_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Build fixed construct-separated judge scores from profile-specific scores."""
    if not candidate_scores_by_profile:
        raise ValueError("At least one candidate score set is required.")
    artifact_keys = sorted(
        set.intersection(
            *[
                set(candidate_scores)
                for candidate_scores in candidate_scores_by_profile.values()
            ]
        )
    )
    ensemble_scores: dict[str, Any] = {}
    for artifact_key in artifact_keys:
        item_scores: dict[str, float] = {}
        component_trace_paths: dict[str, list[str | None]] = {}
        exemplar_entry = next(iter(candidate_scores_by_profile.values()))[artifact_key]
        for dimension_name, item_names in DIMENSION_ITEMS.items():
            profiles = dimension_profile_map[dimension_name]
            source_entries = [
                candidate_scores_by_profile[profile][artifact_key]
                for profile in profiles
            ]
            component_trace_paths[dimension_name] = [
                source_entry.get("trace_path") for source_entry in source_entries
            ]
            for item_name in item_names:
                item_scores[item_name] = _mean(
                    [
                        float(source_entry["item_scores"][item_name])
                        for source_entry in source_entries
                    ]
                )
        dimension_scores = _dimension_means_from_items(item_scores)
        # IEEE citation: This construct-separated ensemble is an internal
        # adaptation of MTMM-style construct separation, not a validated MTMM
        # estimator; it keeps each rubric trait's scoring method explicit [4].
        ensemble_scores[artifact_key] = {
            **exemplar_entry,
            "prompt_profile": "dimension_ensemble",
            "dimension_profile_map": dict(dimension_profile_map),
            "component_trace_paths": component_trace_paths,
            "item_scores": item_scores,
            "dimension_scores": dimension_scores,
            "composite": _mean(
                [dimension_scores[dimension_name] for dimension_name in DIMENSION_NAMES]
            ),
        }
    return ensemble_scores


def _build_item_ensemble_scores(
    *,
    candidate_scores_by_profile: dict[str, dict[str, Any]],
    item_profile_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Build fixed facet-level judge scores from item-specific profile scores."""
    if not candidate_scores_by_profile:
        raise ValueError("At least one candidate score set is required.")
    artifact_keys = sorted(
        set.intersection(
            *[
                set(candidate_scores)
                for candidate_scores in candidate_scores_by_profile.values()
            ]
        )
    )
    ensemble_scores: dict[str, Any] = {}
    for artifact_key in artifact_keys:
        item_scores: dict[str, float] = {}
        component_trace_paths: dict[str, list[str | None]] = {}
        exemplar_entry = next(iter(candidate_scores_by_profile.values()))[artifact_key]
        for item_name in ITEM_NAMES:
            profiles = item_profile_map[item_name]
            source_entries = [
                candidate_scores_by_profile[profile][artifact_key]
                for profile in profiles
            ]
            component_trace_paths[item_name] = [
                source_entry.get("trace_path") for source_entry in source_entries
            ]
            item_scores[item_name] = _mean(
                [
                    float(source_entry["item_scores"][item_name])
                    for source_entry in source_entries
                ]
            )
        dimension_scores = _dimension_means_from_items(item_scores)
        # IEEE citation: This facet-level ensemble is an exploratory internal
        # extension of MTMM-style construct/method separation to rubric items [4].
        ensemble_scores[artifact_key] = {
            **exemplar_entry,
            "prompt_profile": "item_ensemble",
            "item_profile_map": dict(item_profile_map),
            "component_trace_paths": component_trace_paths,
            "item_scores": item_scores,
            "dimension_scores": dimension_scores,
            "composite": _mean(
                [dimension_scores[dimension_name] for dimension_name in DIMENSION_NAMES]
            ),
        }
    return ensemble_scores


def _score_value_for_target(
    score_entry: dict[str, Any],
    target_name: str,
) -> float:
    """Return one item, dimension, or composite judge score."""
    if target_name == "Composite":
        return float(score_entry["composite"])
    if target_name in DIMENSION_NAMES:
        return float(score_entry["dimension_scores"][target_name])
    return float(score_entry["item_scores"][target_name])


def _human_value_for_target(
    artifact_human_scores: dict[str, Any],
    target_name: str,
) -> float:
    """Return one item, dimension, or composite human mean score."""
    if target_name == "Composite":
        return float(artifact_human_scores["composite"])
    if target_name in DIMENSION_NAMES:
        return float(artifact_human_scores["dimension_scores"][target_name])
    return float(artifact_human_scores["item_scores"][target_name])


def _training_metric_value(
    *,
    train_artifact_keys: list[str],
    artifact_human_scores: dict[str, dict[str, Any]],
    judge_scores: dict[str, Any],
    target_name: str,
    metric_name: str,
) -> float | None:
    """Return a training-fold correlation for one candidate prompt."""
    human_values = [
        _human_value_for_target(artifact_human_scores[artifact_key], target_name)
        for artifact_key in train_artifact_keys
    ]
    judge_values = [
        _score_value_for_target(judge_scores[artifact_key], target_name)
        for artifact_key in train_artifact_keys
    ]
    if metric_name == "pearson":
        return _pearson(human_values, judge_values)
    return _spearman(human_values, judge_values)


def _build_leave_one_episode_out_report(
    *,
    responses: list[HumanResponse],
    candidate_scores_by_profile: dict[str, dict[str, Any]],
    selection_metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build pooled out-of-fold correlations from episode-grouped folds."""
    artifact_human_scores = _artifact_level_human_scores(responses)
    artifact_keys = sorted(artifact_human_scores)
    episode_groups = sorted(
        {artifact_human_scores[artifact_key]["episode_group"] for artifact_key in artifact_keys}
    )
    if len(episode_groups) < 2:
        raise ValueError("Leave-one-episode-out CV requires at least two episodes.")
    for profile, scores in candidate_scores_by_profile.items():
        missing = sorted(set(artifact_keys) - set(scores))
        if missing:
            raise ValueError(
                f"Candidate profile {profile} is missing scores for: "
                + ", ".join(missing)
            )

    folds: list[dict[str, Any]] = []
    out_of_fold_scores: dict[str, dict[str, Any]] = {}
    for episode_group in episode_groups:
        train_artifact_keys = [
            artifact_key
            for artifact_key in artifact_keys
            if artifact_human_scores[artifact_key]["episode_group"] != episode_group
        ]
        held_out_artifact_keys = [
            artifact_key
            for artifact_key in artifact_keys
            if artifact_human_scores[artifact_key]["episode_group"] == episode_group
        ]
        candidates: list[dict[str, Any]] = []
        for profile, judge_scores in candidate_scores_by_profile.items():
            metric_value = _training_metric_value(
                train_artifact_keys=train_artifact_keys,
                artifact_human_scores=artifact_human_scores,
                judge_scores=judge_scores,
                target_name="Composite",
                metric_name=selection_metric,
            )
            pearson_value = _training_metric_value(
                train_artifact_keys=train_artifact_keys,
                artifact_human_scores=artifact_human_scores,
                judge_scores=judge_scores,
                target_name="Composite",
                metric_name="pearson",
            )
            spearman_value = _training_metric_value(
                train_artifact_keys=train_artifact_keys,
                artifact_human_scores=artifact_human_scores,
                judge_scores=judge_scores,
                target_name="Composite",
                metric_name="spearman",
            )
            candidates.append(
                {
                    "profile": profile,
                    "selection_metric": _round_or_none(metric_value),
                    "training_pearson": _round_or_none(pearson_value),
                    "training_spearman": _round_or_none(spearman_value),
                    "_sort_metric": metric_value if metric_value is not None else -2.0,
                    "_sort_pearson": pearson_value if pearson_value is not None else -2.0,
                }
            )
        # IEEE citation: Leave-one-group-out selection keeps held-out groups absent
        # from the training decision, following grouped cross-validation [1].
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["_sort_metric"],
                candidate["_sort_pearson"],
                candidate["profile"],
            ),
        )
        selected_profile = str(selected["profile"])
        for artifact_key in held_out_artifact_keys:
            out_of_fold_scores[artifact_key] = {
                **candidate_scores_by_profile[selected_profile][artifact_key],
                "selected_prompt_profile": selected_profile,
                "held_out_episode_group": episode_group,
            }
        folds.append(
            {
                "held_out_episode_group": episode_group,
                "training_artifact_count": len(train_artifact_keys),
                "held_out_artifact_count": len(held_out_artifact_keys),
                "held_out_artifact_keys": held_out_artifact_keys,
                "candidate_training_results": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if not key.startswith("_")
                    }
                    for candidate in sorted(
                        candidates,
                        key=lambda candidate: (
                            candidate["_sort_metric"],
                            candidate["_sort_pearson"],
                            candidate["profile"],
                        ),
                        reverse=True,
                    )
                ],
                "selected_prompt_profile": selected_profile,
            }
        )

    alignment_rows = []
    for artifact_key in artifact_keys:
        human_row = artifact_human_scores[artifact_key]
        judge_row = out_of_fold_scores[artifact_key]
        alignment_rows.append(
            {
                "artifact_key": artifact_key,
                "episode_group": human_row["episode_group"],
                "podcast_id": human_row["podcast_id"],
                "duration": human_row["duration"],
                "human_response_count": human_row["human_response_count"],
                "selected_prompt_profile": judge_row["selected_prompt_profile"],
                "human_item_scores": human_row["item_scores"],
                "judge_item_scores": judge_row["item_scores"],
                "human_dimension_scores": human_row["dimension_scores"],
                "judge_dimension_scores": judge_row["dimension_scores"],
                "human_composite": human_row["composite"],
                "judge_composite": judge_row["composite"],
            }
        )

    def build_target_entries(target_names: list[str]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for offset, target_name in enumerate(target_names):
            human_values = [
                _human_value_for_target(artifact_human_scores[artifact_key], target_name)
                for artifact_key in artifact_keys
            ]
            judge_values = [
                _score_value_for_target(out_of_fold_scores[artifact_key], target_name)
                for artifact_key in artifact_keys
            ]
            groups = [
                str(artifact_human_scores[artifact_key]["episode_group"])
                for artifact_key in artifact_keys
            ]
            entries[target_name] = _correlation_entry_with_uncertainty(
                human_values,
                judge_values,
                groups=groups,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + offset,
            )
        return entries

    episode_artifact_counts = {
        episode_group: sum(
            1
            for artifact_key in artifact_keys
            if artifact_human_scores[artifact_key]["episode_group"] == episode_group
        )
        for episode_group in episode_groups
    }
    response_count = sum(
        int(artifact_human_scores[artifact_key]["human_response_count"])
        for artifact_key in artifact_keys
    )
    return {
        "schema_version": CV_REPORT_SCHEMA_VERSION,
        "method": "leave_one_episode_out_by_episode",
        "selection_metric": selection_metric,
        "selection_target": "Composite",
        "candidate_prompt_profiles": sorted(candidate_scores_by_profile),
        "response_count": response_count,
        "artifact_count": len(artifact_keys),
        "episode_count": len(episode_groups),
        "episode_artifact_counts": episode_artifact_counts,
        "folds": folds,
        "pooled_out_of_fold": {
            "artifact_level": {
                "items": build_target_entries(list(ITEM_NAMES)),
                "dimension_means": build_target_entries(
                    [*list(DIMENSION_NAMES), "Composite"]
                ),
            }
        },
        "out_of_fold_alignment": alignment_rows,
        "methodology_notes": [
            "The unit of evaluation is the unique episode-variation artifact. "
            "Multiple human response rows for the same artifact are averaged before "
            "correlation because the judge emits one score per artifact.",
            "Each fold holds out one episode group and selects one prompt profile "
            "using only the remaining episode groups' artifact-level Composite "
            f"{selection_metric} correlation.",
            "Final correlations are computed once from pooled out-of-fold artifact "
            "predictions. Per-fold correlations are intentionally not reported "
            "because held-out folds contain too few artifacts.",
            "Confidence intervals are approximate and expected to be wide for this "
            f"small n={len(artifact_human_scores)} artifact-level sample, especially "
            "for the episode-cluster bootstrap with only five episode groups.",
        ],
    }


def _write_leave_one_episode_out_alignment_csv(
    path: Path,
    alignment_rows: list[dict[str, Any]],
) -> None:
    """Write one artifact-level out-of-fold alignment CSV."""
    fieldnames = [
        "artifact_key",
        "episode_group",
        "podcast_id",
        "duration",
        "human_response_count",
        "selected_prompt_profile",
        *[f"human_mean_{item_name}" for item_name in ITEM_NAMES],
        *[f"judge_oof_{item_name}" for item_name in ITEM_NAMES],
        *[f"human_mean_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        *[f"judge_oof_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        "human_mean_composite",
        "judge_oof_composite",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for alignment in alignment_rows:
            row: dict[str, Any] = {
                "artifact_key": alignment["artifact_key"],
                "episode_group": alignment["episode_group"],
                "podcast_id": alignment["podcast_id"],
                "duration": alignment["duration"],
                "human_response_count": alignment["human_response_count"],
                "selected_prompt_profile": alignment["selected_prompt_profile"],
                "human_mean_composite": alignment["human_composite"],
                "judge_oof_composite": alignment["judge_composite"],
            }
            for item_name in ITEM_NAMES:
                row[f"human_mean_{item_name}"] = alignment["human_item_scores"].get(
                    item_name
                )
                row[f"judge_oof_{item_name}"] = alignment["judge_item_scores"].get(
                    item_name
                )
            for dimension_name in DIMENSION_NAMES:
                row[f"human_mean_{dimension_name}"] = alignment[
                    "human_dimension_scores"
                ].get(dimension_name)
                row[f"judge_oof_{dimension_name}"] = alignment[
                    "judge_dimension_scores"
                ].get(dimension_name)
            writer.writerow(row)


def _build_artifact_level_uncertainty_report(
    responses: list[HumanResponse],
    judge_scores: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build artifact-level item/dimension correlations with uncertainty."""
    grouped = _group_responses_by_artifact(responses)
    artifact_keys = sorted(
        artifact_key for artifact_key in grouped if artifact_key in judge_scores
    )

    def human_value(artifact_key: str, target_name: str) -> float:
        rows = grouped[artifact_key]
        if target_name == "Composite":
            composites = [
                composite
                for response in rows
                if (composite := _human_composite_from_items(response.item_scores))
                is not None
            ]
            return _mean(composites)
        if target_name in DIMENSION_NAMES:
            values = [
                response.dimension_scores[target_name]
                for response in rows
                if target_name in response.dimension_scores
            ]
            return _mean(values)
        values = [
            response.item_scores[target_name]
            for response in rows
            if target_name in response.item_scores
        ]
        return _mean(values)

    def build_entries(target_names: list[str]) -> dict[str, Any]:
        entries: dict[str, Any] = {}
        for offset, target_name in enumerate(target_names):
            human_values = [human_value(artifact_key, target_name) for artifact_key in artifact_keys]
            judge_values = [
                _score_value_for_target(judge_scores[artifact_key], target_name)
                for artifact_key in artifact_keys
            ]
            groups = [
                _episode_group_from_artifact_key(artifact_key)
                for artifact_key in artifact_keys
            ]
            entries[target_name] = _correlation_entry_with_uncertainty(
                human_values,
                judge_values,
                groups=groups,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + offset,
            )
        return entries

    return {
        "artifact_count": len(artifact_keys),
        "items": build_entries(list(ITEM_NAMES)),
        "dimension_means": build_entries([*list(DIMENSION_NAMES), "Composite"]),
    }


def _calibration_fingerprint(
    calibration_item_scores_by_artifact: dict[str, dict[str, float]] | None,
) -> str:
    """Fingerprint the in-sample anchor payload used by leaked calibration runs."""
    if not calibration_item_scores_by_artifact:
        return ""
    payload = json.dumps(
        calibration_item_scores_by_artifact,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def _in_sample_gold_anchor_note(
    *,
    artifact_key: str,
    item_scores: dict[str, float],
) -> str:
    """Build the direct in-sample human-score anchor note for one artifact."""
    dimension_scores = _dimension_means_from_items(item_scores)
    item_json = json.dumps(
        {item_name: item_scores[item_name] for item_name in ITEM_NAMES},
        indent=2,
        sort_keys=True,
    )
    dimension_json = json.dumps(
        {dimension_name: round(dimension_scores[dimension_name], 4) for dimension_name in DIMENSION_NAMES},
        indent=2,
        sort_keys=True,
    )
    return (
        f"\n\nCurrent artifact key: {artifact_key}\n"
        "In-sample human artifact-level item mean anchors for this exact artifact. "
        "For this calibration experiment, copy these values exactly into `item_scores`:\n"
        f"{item_json}\n"
        "The corresponding dimension means, derived from the item means, are:\n"
        f"{dimension_json}"
    )


def _write_response_alignment_csv(
    path: Path,
    responses: list[HumanResponse],
    judge_scores: dict[str, Any],
) -> None:
    """Write one row per human response with paired judge scores."""
    fieldnames = [
        "participant_id",
        "participant_email",
        "artifact_key",
        "podcast_id",
        "duration",
        *[f"human_{item_name}" for item_name in ITEM_NAMES],
        *[f"judge_{item_name}" for item_name in ITEM_NAMES],
        *[f"human_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        *[f"judge_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        "human_composite",
        "judge_composite",
        "comment",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            artifact_key = response.assignment.artifact_key
            judge = judge_scores.get(artifact_key)
            if judge is None:
                continue
            row: dict[str, Any] = {
                "participant_id": response.participant_id,
                "participant_email": response.participant_email,
                "artifact_key": artifact_key,
                "podcast_id": response.assignment.podcast_id,
                "duration": response.assignment.duration,
                "human_composite": _human_composite_from_items(response.item_scores),
                "judge_composite": judge["composite"],
                "comment": response.comment,
            }
            for item_name in ITEM_NAMES:
                row[f"human_{item_name}"] = response.item_scores.get(item_name)
                row[f"judge_{item_name}"] = judge["item_scores"].get(item_name)
            for dimension_name in DIMENSION_NAMES:
                row[f"human_{dimension_name}"] = response.dimension_scores.get(
                    dimension_name
                )
                row[f"judge_{dimension_name}"] = judge["dimension_scores"].get(
                    dimension_name
                )
            writer.writerow(row)


def _write_artifact_alignment_csv(
    path: Path,
    responses: list[HumanResponse],
    judge_scores: dict[str, Any],
) -> None:
    """Write one row per artifact with mean human scores and judge scores."""
    fieldnames = [
        "artifact_key",
        "podcast_id",
        "duration",
        "human_response_count",
        *[f"human_mean_{item_name}" for item_name in ITEM_NAMES],
        *[f"judge_{item_name}" for item_name in ITEM_NAMES],
        *[f"human_mean_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        *[f"judge_{dimension_name}" for dimension_name in DIMENSION_NAMES],
        "human_mean_composite",
        "judge_composite",
    ]
    grouped = _group_responses_by_artifact(responses)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact_key, rows in sorted(grouped.items()):
            judge = judge_scores.get(artifact_key)
            if judge is None:
                continue
            row: dict[str, Any] = {
                "artifact_key": artifact_key,
                "podcast_id": rows[0].assignment.podcast_id,
                "duration": rows[0].assignment.duration,
                "human_response_count": len(rows),
                "judge_composite": judge["composite"],
            }
            composites = [
                composite
                for response in rows
                if (composite := _human_composite_from_items(response.item_scores))
                is not None
            ]
            row["human_mean_composite"] = _mean(composites) if composites else None
            for item_name in ITEM_NAMES:
                values = [
                    response.item_scores[item_name]
                    for response in rows
                    if item_name in response.item_scores
                ]
                row[f"human_mean_{item_name}"] = _mean(values) if values else None
                row[f"judge_{item_name}"] = judge["item_scores"].get(item_name)
            for dimension_name in DIMENSION_NAMES:
                values = [
                    response.dimension_scores[dimension_name]
                    for response in rows
                    if dimension_name in response.dimension_scores
                ]
                row[f"human_mean_{dimension_name}"] = _mean(values) if values else None
                row[f"judge_{dimension_name}"] = judge["dimension_scores"].get(
                    dimension_name
                )
            writer.writerow(row)


def main() -> None:
    """Run the participant-audio judge alignment workflow."""
    args = _parse_args()
    load_repo_dotenv(REPO_ROOT)
    args.model = _resolve_model(judge_provider=args.judge_provider, raw_model=args.model)
    args.deepseek_base_url = (
        args.deepseek_base_url.strip()
        or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    )
    args.vllm_base_url = _resolve_vllm_base_url(args.vllm_base_url)
    args.vllm_api_key = _resolve_vllm_api_key(args.vllm_api_key)
    if args.judge_provider in {"deepseek", "vllm"} and args.audio_mode != "none":
        raise RuntimeError(
            f"{args.judge_provider} is text-only in this workflow; rerun with --audio-mode none."
        )
    if args.prompt_profile == "in_sample_gold_anchor" and not args.allow_label_leakage:
        raise RuntimeError(
            "in_sample_gold_anchor injects human score anchors and is not a judge "
            "reliability profile. Rerun with --allow-label-leakage only for an "
            "explicit in-sample pipeline check."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _discover_participant_artifacts(
        participant_audio_root=args.participant_audio_root,
        transcript_output_dir=args.transcript_output_dir,
    )
    artifact_manifest_path = args.output_dir / "participant_audio_v3_manifest.json"
    _write_artifact_manifest(artifact_manifest_path, artifacts)
    if args.prepare_only:
        print(f"Prepared {len(artifacts)} transcript artifact(s).")
        print(f"Manifest: {artifact_manifest_path}")
        return

    assignments = _load_assignments(args.assignments_csv)
    responses = _load_human_responses(args.responses_csv, assignments)
    required_artifact_keys = {response.assignment.artifact_key for response in responses}
    missing_artifacts = sorted(required_artifact_keys - set(artifacts))
    if missing_artifacts:
        raise RuntimeError(
            "Missing participant-audio artifacts for responded assignments: "
            + ", ".join(missing_artifacts)
        )
    responded_artifacts = {
        artifact_key: artifacts[artifact_key]
        for artifact_key in sorted(required_artifact_keys)
    }
    if args.item_ensemble_profile_map and args.dimension_ensemble_profile_map:
        raise RuntimeError(
            "Use either --item-ensemble-profile-map or "
            "--dimension-ensemble-profile-map, not both."
        )
    if args.item_ensemble_profile_map:
        item_profile_map = _parse_item_ensemble_profile_map(
            args.item_ensemble_profile_map
        )
        candidate_profiles = list(
            dict.fromkeys(
                profile
                for profiles in item_profile_map.values()
                for profile in profiles
            )
        )
        candidate_scores_by_profile: dict[str, dict[str, Any]] = {}
        candidate_score_paths: dict[str, str] = {}
        candidate_trace_dirs: dict[str, str] = {}
        for candidate_profile in candidate_profiles:
            candidate_run_label = _run_label(
                judge_provider=args.judge_provider,
                model=args.model,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                audio_mode=args.audio_mode,
                prompt_profile=candidate_profile,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
            )
            candidate_scores_path = (
                args.output_dir / f"{candidate_run_label}_participant_audio_item_scores.json"
            )
            candidate_trace_dir = args.output_dir / f"{candidate_run_label}_judge_traces"
            candidate_scores_by_profile[candidate_profile] = _load_judge_scores(
                responded_artifacts,
                candidate_scores_path,
                model=args.model,
                judge_provider=args.judge_provider,
                audio_mode=args.audio_mode,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_base_url=args.deepseek_base_url,
                vllm_base_url=args.vllm_base_url,
                vllm_api_key=args.vllm_api_key,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                prompt_profile=candidate_profile,
                request_timeout_seconds=args.request_timeout_seconds,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
                max_judge_attempts=args.max_judge_attempts,
                trace_dir=candidate_trace_dir,
                skip_gemini=args.skip_gemini,
                force_gemini=args.force_gemini,
                calibration_item_scores_by_artifact=None,
            )
            candidate_score_paths[candidate_profile] = str(candidate_scores_path)
            candidate_trace_dirs[candidate_profile] = str(candidate_trace_dir)
        ensemble_scores = _build_item_ensemble_scores(
            candidate_scores_by_profile=candidate_scores_by_profile,
            item_profile_map=item_profile_map,
        )
        ensemble_hash_payload = json.dumps(
            item_profile_map,
            sort_keys=True,
            separators=(",", ":"),
        )
        ensemble_run_label = _slugify(
            "_".join(
                [
                    args.judge_provider,
                    args.model,
                    f"thinking_{args.thinking_mode}"
                    if args.judge_provider in {"deepseek", "vllm"}
                    else "",
                    "item_ensemble",
                    hashlib.sha256(ensemble_hash_payload.encode("utf-8")).hexdigest()[
                        :12
                    ],
                ]
            )
        )
        ensemble_scores_path = (
            args.output_dir / f"{ensemble_run_label}_participant_audio_item_scores.json"
        )
        _json_dump(ensemble_scores_path, ensemble_scores)
        correlation_report = _build_correlation_report(responses, ensemble_scores)
        report = {
            "run_label": ensemble_run_label,
            "judge_provider": args.judge_provider,
            "responses_csv": str(args.responses_csv),
            "assignments_csv": str(args.assignments_csv),
            "participant_audio_root": str(args.participant_audio_root),
            "transcript_output_dir": str(args.transcript_output_dir),
            "artifact_manifest_path": str(artifact_manifest_path),
            "judge_scores_path": str(ensemble_scores_path),
            "model": args.model,
            "requested_max_tokens": args.max_tokens,
            "judge_seed": args.judge_seed,
            "judge_temperature": args.judge_temperature,
            "vllm_base_url": args.vllm_base_url
            if args.judge_provider == "vllm"
            else None,
            "thinking_mode": args.thinking_mode
            if args.judge_provider in {"deepseek", "vllm"}
            else None,
            "gemini_thinking_level": args.gemini_thinking_level
            if args.judge_provider == "gemini"
            else None,
            "audio_mode": args.audio_mode,
            "item_profile_map": item_profile_map,
            "candidate_score_paths": candidate_score_paths,
            "candidate_trace_dirs": candidate_trace_dirs,
            "response_count": len(responses),
            "artifact_count": len(responded_artifacts),
            "methodology_notes": [
                "This fixed facet-level ensemble maps prompt profiles to rubric "
                "items, averages item scores when multiple profiles are mapped, "
                "then recombines scores through the original rubric dimensions.",
                "The approach is an exploratory internal adaptation of construct "
                "and method separation from multitrait/multimethod validation; it "
                "is not an independently validated MTMM estimator.",
                "Because the item map was developed during exploratory prompt work "
                f"on this same small n={len(responded_artifacts)} artifact set, "
                "treat this as an exploratory candidate requiring prospective "
                "confirmation.",
            ],
            "correlation": correlation_report,
            "artifact_level_uncertainty": _build_artifact_level_uncertainty_report(
                responses,
                ensemble_scores,
                bootstrap_samples=args.cv_bootstrap_samples,
                bootstrap_seed=args.cv_bootstrap_seed,
            ),
        }
        ensemble_report_path = (
            args.output_dir / f"{ensemble_run_label}_human_item_correlation_report.json"
        )
        _json_dump(ensemble_report_path, report)
        _write_response_alignment_csv(
            args.output_dir / f"{ensemble_run_label}_response_level_alignment.csv",
            responses,
            ensemble_scores,
        )
        _write_artifact_alignment_csv(
            args.output_dir / f"{ensemble_run_label}_artifact_level_alignment.csv",
            responses,
            ensemble_scores,
        )
        print(json.dumps(correlation_report, indent=2, ensure_ascii=False))
        print(f"Item ensemble report: {ensemble_report_path}")
        return

    if args.dimension_ensemble_profile_map:
        dimension_profile_map = _parse_dimension_ensemble_profile_map(
            args.dimension_ensemble_profile_map
        )
        candidate_profiles = list(
            dict.fromkeys(
                profile
                for profiles in dimension_profile_map.values()
                for profile in profiles
            )
        )
        candidate_scores_by_profile: dict[str, dict[str, Any]] = {}
        candidate_score_paths: dict[str, str] = {}
        candidate_trace_dirs: dict[str, str] = {}
        for candidate_profile in candidate_profiles:
            candidate_run_label = _run_label(
                judge_provider=args.judge_provider,
                model=args.model,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                audio_mode=args.audio_mode,
                prompt_profile=candidate_profile,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
            )
            candidate_scores_path = (
                args.output_dir / f"{candidate_run_label}_participant_audio_item_scores.json"
            )
            candidate_trace_dir = args.output_dir / f"{candidate_run_label}_judge_traces"
            candidate_scores_by_profile[candidate_profile] = _load_judge_scores(
                responded_artifacts,
                candidate_scores_path,
                model=args.model,
                judge_provider=args.judge_provider,
                audio_mode=args.audio_mode,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_base_url=args.deepseek_base_url,
                vllm_base_url=args.vllm_base_url,
                vllm_api_key=args.vllm_api_key,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                prompt_profile=candidate_profile,
                request_timeout_seconds=args.request_timeout_seconds,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
                max_judge_attempts=args.max_judge_attempts,
                trace_dir=candidate_trace_dir,
                skip_gemini=args.skip_gemini,
                force_gemini=args.force_gemini,
                calibration_item_scores_by_artifact=None,
            )
            candidate_score_paths[candidate_profile] = str(candidate_scores_path)
            candidate_trace_dirs[candidate_profile] = str(candidate_trace_dir)
        ensemble_scores = _build_dimension_ensemble_scores(
            candidate_scores_by_profile=candidate_scores_by_profile,
            dimension_profile_map=dimension_profile_map,
        )
        ensemble_hash_payload = json.dumps(
            dimension_profile_map,
            sort_keys=True,
            separators=(",", ":"),
        )
        ensemble_run_label = _slugify(
            "_".join(
                [
                    args.judge_provider,
                    args.model,
                    f"thinking_{args.thinking_mode}"
                    if args.judge_provider in {"deepseek", "vllm"}
                    else "",
                    "dimension_ensemble",
                    hashlib.sha256(ensemble_hash_payload.encode("utf-8")).hexdigest()[
                        :12
                    ],
                ]
            )
        )
        ensemble_scores_path = (
            args.output_dir / f"{ensemble_run_label}_participant_audio_item_scores.json"
        )
        _json_dump(ensemble_scores_path, ensemble_scores)
        correlation_report = _build_correlation_report(responses, ensemble_scores)
        report = {
            "run_label": ensemble_run_label,
            "judge_provider": args.judge_provider,
            "responses_csv": str(args.responses_csv),
            "assignments_csv": str(args.assignments_csv),
            "participant_audio_root": str(args.participant_audio_root),
            "transcript_output_dir": str(args.transcript_output_dir),
            "artifact_manifest_path": str(artifact_manifest_path),
            "judge_scores_path": str(ensemble_scores_path),
            "model": args.model,
            "requested_max_tokens": args.max_tokens,
            "judge_seed": args.judge_seed,
            "judge_temperature": args.judge_temperature,
            "vllm_base_url": args.vllm_base_url
            if args.judge_provider == "vllm"
            else None,
            "thinking_mode": args.thinking_mode
            if args.judge_provider in {"deepseek", "vllm"}
            else None,
            "gemini_thinking_level": args.gemini_thinking_level
            if args.judge_provider == "gemini"
            else None,
            "audio_mode": args.audio_mode,
            "dimension_profile_map": dimension_profile_map,
            "candidate_score_paths": candidate_score_paths,
            "candidate_trace_dirs": candidate_trace_dirs,
            "response_count": len(responses),
            "artifact_count": len(responded_artifacts),
            "methodology_notes": [
                "This fixed construct-separated ensemble uses different prompt profiles "
                "for rubric dimensions, averages profile item scores within a dimension "
                "when multiple profiles are mapped, and recombines item scores by the "
                "original rubric dimension structure.",
                "The approach is an internal adaptation of construct-separation logic "
                "from multitrait/multimethod validation, not an independently validated "
                "MTMM estimator.",
                "Because the profile map was developed during exploratory prompt work "
                f"on this same small n={len(responded_artifacts)} artifact set, "
                "treat this as an exploratory candidate requiring confirmation on "
                "new human-rated artifacts.",
            ],
            "correlation": correlation_report,
            "artifact_level_uncertainty": _build_artifact_level_uncertainty_report(
                responses,
                ensemble_scores,
                bootstrap_samples=args.cv_bootstrap_samples,
                bootstrap_seed=args.cv_bootstrap_seed,
            ),
        }
        ensemble_report_path = (
            args.output_dir / f"{ensemble_run_label}_human_item_correlation_report.json"
        )
        _json_dump(ensemble_report_path, report)
        _write_response_alignment_csv(
            args.output_dir / f"{ensemble_run_label}_response_level_alignment.csv",
            responses,
            ensemble_scores,
        )
        _write_artifact_alignment_csv(
            args.output_dir / f"{ensemble_run_label}_artifact_level_alignment.csv",
            responses,
            ensemble_scores,
        )
        print(json.dumps(correlation_report, indent=2, ensure_ascii=False))
        print(f"Dimension ensemble report: {ensemble_report_path}")
        return

    if args.leave_one_episode_out:
        candidate_profiles = _parse_cv_candidate_profiles(
            args.cv_candidate_profiles,
            args.prompt_profile,
        )
        candidate_scores_by_profile: dict[str, dict[str, Any]] = {}
        candidate_score_paths: dict[str, str] = {}
        candidate_trace_dirs: dict[str, str] = {}
        for candidate_profile in candidate_profiles:
            candidate_run_label = _run_label(
                judge_provider=args.judge_provider,
                model=args.model,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                audio_mode=args.audio_mode,
                prompt_profile=candidate_profile,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
            )
            candidate_scores_path = (
                args.output_dir / f"{candidate_run_label}_participant_audio_item_scores.json"
            )
            if args.judge_trace_dir is not None and len(candidate_profiles) == 1:
                candidate_trace_dir = args.judge_trace_dir
            else:
                candidate_trace_dir = args.output_dir / f"{candidate_run_label}_judge_traces"
            candidate_scores_by_profile[candidate_profile] = _load_judge_scores(
                responded_artifacts,
                candidate_scores_path,
                model=args.model,
                judge_provider=args.judge_provider,
                audio_mode=args.audio_mode,
                thinking_mode=args.thinking_mode,
                gemini_thinking_level=args.gemini_thinking_level,
                deepseek_base_url=args.deepseek_base_url,
                vllm_base_url=args.vllm_base_url,
                vllm_api_key=args.vllm_api_key,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
                prompt_profile=candidate_profile,
                request_timeout_seconds=args.request_timeout_seconds,
                max_tokens=args.max_tokens,
                judge_seed=args.judge_seed,
                judge_temperature=args.judge_temperature,
                max_judge_attempts=args.max_judge_attempts,
                trace_dir=candidate_trace_dir,
                skip_gemini=args.skip_gemini,
                force_gemini=args.force_gemini,
                calibration_item_scores_by_artifact=None,
            )
            candidate_score_paths[candidate_profile] = str(candidate_scores_path)
            candidate_trace_dirs[candidate_profile] = str(candidate_trace_dir)

        cv_report = _build_leave_one_episode_out_report(
            responses=responses,
            candidate_scores_by_profile=candidate_scores_by_profile,
            selection_metric=args.cv_selection_metric,
            bootstrap_samples=args.cv_bootstrap_samples,
            bootstrap_seed=args.cv_bootstrap_seed,
        )
        cv_hash_payload = json.dumps(
            {
                "candidate_profiles": candidate_profiles,
                "selection_metric": args.cv_selection_metric,
                "bootstrap_samples": args.cv_bootstrap_samples,
                "bootstrap_seed": args.cv_bootstrap_seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        cv_run_label = _slugify(
            "_".join(
                [
                    args.judge_provider,
                    args.model,
                    f"thinking_{args.thinking_mode}"
                    if args.judge_provider in {"deepseek", "vllm"}
                    else "",
                    "loeo",
                    args.cv_selection_metric,
                    hashlib.sha256(cv_hash_payload.encode("utf-8")).hexdigest()[:12],
                ]
            )
        )
        report = {
            "run_label": cv_run_label,
            "judge_provider": args.judge_provider,
            "responses_csv": str(args.responses_csv),
            "assignments_csv": str(args.assignments_csv),
            "participant_audio_root": str(args.participant_audio_root),
            "transcript_output_dir": str(args.transcript_output_dir),
            "artifact_manifest_path": str(artifact_manifest_path),
            "model": args.model,
            "requested_max_tokens": args.max_tokens,
            "judge_seed": args.judge_seed,
            "judge_temperature": args.judge_temperature,
            "vllm_base_url": args.vllm_base_url
            if args.judge_provider == "vllm"
            else None,
            "thinking_mode": args.thinking_mode
            if args.judge_provider in {"deepseek", "vllm"}
            else None,
            "gemini_thinking_level": args.gemini_thinking_level
            if args.judge_provider == "gemini"
            else None,
            "audio_mode": args.audio_mode,
            "candidate_score_paths": candidate_score_paths,
            "candidate_trace_dirs": candidate_trace_dirs,
            "cross_validation": cv_report,
        }
        cv_report_path = args.output_dir / f"{cv_run_label}_leave_one_episode_out_report.json"
        _json_dump(cv_report_path, report)
        _write_leave_one_episode_out_alignment_csv(
            args.output_dir / f"{cv_run_label}_leave_one_episode_out_alignment.csv",
            cv_report["out_of_fold_alignment"],
        )
        print(
            json.dumps(
                cv_report["pooled_out_of_fold"],
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"Leave-one-episode-out report: {cv_report_path}")
        return

    calibration_item_scores_by_artifact = (
        _artifact_mean_item_scores(responses)
        if args.prompt_profile == "in_sample_gold_anchor"
        else None
    )
    calibration_fingerprint = _calibration_fingerprint(
        calibration_item_scores_by_artifact
    )
    run_label = _run_label(
        judge_provider=args.judge_provider,
        model=args.model,
        thinking_mode=args.thinking_mode,
        gemini_thinking_level=args.gemini_thinking_level,
        deepseek_reasoning_effort=args.deepseek_reasoning_effort,
        audio_mode=args.audio_mode,
        prompt_profile=args.prompt_profile,
        max_tokens=args.max_tokens,
        judge_seed=args.judge_seed,
        judge_temperature=args.judge_temperature,
    )
    judge_scores_path = args.output_dir / f"{run_label}_participant_audio_item_scores.json"
    judge_trace_dir = args.judge_trace_dir or (args.output_dir / f"{run_label}_judge_traces")
    judge_scores = _load_judge_scores(
        responded_artifacts,
        judge_scores_path,
        model=args.model,
        judge_provider=args.judge_provider,
        audio_mode=args.audio_mode,
        thinking_mode=args.thinking_mode,
        gemini_thinking_level=args.gemini_thinking_level,
        deepseek_base_url=args.deepseek_base_url,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        deepseek_reasoning_effort=args.deepseek_reasoning_effort,
        prompt_profile=args.prompt_profile,
        request_timeout_seconds=args.request_timeout_seconds,
        max_tokens=args.max_tokens,
        judge_seed=args.judge_seed,
        judge_temperature=args.judge_temperature,
        max_judge_attempts=args.max_judge_attempts,
        trace_dir=judge_trace_dir,
        skip_gemini=args.skip_gemini,
        force_gemini=args.force_gemini,
        calibration_item_scores_by_artifact=calibration_item_scores_by_artifact,
    )
    correlation_report = _build_correlation_report(responses, judge_scores)
    report = {
        "run_label": run_label,
        "judge_provider": args.judge_provider,
        "responses_csv": str(args.responses_csv),
        "assignments_csv": str(args.assignments_csv),
        "participant_audio_root": str(args.participant_audio_root),
        "transcript_output_dir": str(args.transcript_output_dir),
        "artifact_manifest_path": str(artifact_manifest_path),
        "judge_scores_path": str(judge_scores_path),
        "judge_trace_dir": str(judge_trace_dir),
        "model": args.model,
        "requested_max_tokens": args.max_tokens,
        "judge_seed": args.judge_seed,
        "judge_temperature": args.judge_temperature,
        "vllm_base_url": args.vllm_base_url if args.judge_provider == "vllm" else None,
        "thinking_mode": args.thinking_mode if args.judge_provider in {"deepseek", "vllm"} else None,
        "gemini_thinking_level": args.gemini_thinking_level
        if args.judge_provider == "gemini"
        else None,
        "audio_mode": args.audio_mode,
        "prompt_profile": args.prompt_profile,
        "calibration_fingerprint": calibration_fingerprint
        if args.prompt_profile == "in_sample_gold_anchor"
        else None,
        "response_count": len(responses),
        "artifact_count": len(responded_artifacts),
        "methodology_notes": [
            "Artifact-level correlations are the primary alignment statistics because "
            "one judge score is reused for every human response assigned to the "
            "same podcast-duration artifact.",
            "Response-level correlations are descriptive only and are not independent "
            "when multiple respondents evaluated the same artifact.",
            (
                "This in_sample_gold_anchor run injects artifact-level human means "
                "into the prompt and should be treated only as an in-sample upper "
                "bound / pipeline check, not as independent judge reliability."
                if args.prompt_profile == "in_sample_gold_anchor"
                else "The workflow reports observed correlations for the specified "
                "judge inputs; it does not transform scores to force high correlation."
            ),
        ],
        "correlation": correlation_report,
    }
    correlation_report_path = args.output_dir / f"{run_label}_human_item_correlation_report.json"
    _json_dump(correlation_report_path, report)
    _write_response_alignment_csv(
        args.output_dir / f"{run_label}_response_level_alignment.csv",
        responses,
        judge_scores,
    )
    _write_artifact_alignment_csv(
        args.output_dir / f"{run_label}_artifact_level_alignment.csv",
        responses,
        judge_scores,
    )
    print(json.dumps(correlation_report, indent=2, ensure_ascii=False))
    print(f"Correlation report: {correlation_report_path}")


if __name__ == "__main__":
    main()
