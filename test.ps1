$source = "C:\Users\chris\card-framework\artifacts\sporc_human_eval_stage2_sweeps\interjector_leading_silence_trim_ep2_5m_local_20260325_151713\runs\02_spirit_adrift_s_nate_garrett_on_the_love_that_saved_his_life\5m"
$summary = Join-Path $source "summary.xml"
$voiceManifest = Join-Path $source "audio_stage\voice_clone\manifest.json"
$root = "C:\Users\chris\card-framework\artifacts\interjector_tuning_ep2_5m_20260325"

$variants = @(
  @{ name = "echo20_preserve5"; echo = 20; preserve = 5; back = 120; guard = 180 },
  @{ name = "echo10_preserve5"; echo = 10; preserve = 5; back = 120; guard = 180 },
  @{ name = "echo0_preserve5";  echo = 0;  preserve = 5; back = 120; guard = 180 },
  @{ name = "echo0_preserve0";  echo = 0;  preserve = 0; back = 120; guard = 180 }
)

foreach ($v in $variants) {
  $out = Join-Path $root $v.name
  docker compose --profile gpu run --rm setup-and-run-gpu `
    --no-menu `
    --output-root $out `
    --override pipeline.start_stage=stage-4 `
    --override pipeline.final_summary_path=$summary `
    --override pipeline.voice_clone_manifest_path=$voiceManifest `
    --override selection.llm=deepseek_chat `
    --override selection.stage_llm.summarizer=deepseek_chat `
    --override selection.stage_llm.critic=deepseek_chat `
    --override selection.stage_llm.interjector=deepseek_chat `
    --override profiles.llm.deepseek_chat.model=deepseek-chat `
    --override audio.device=cuda `
    --override audio.voice_clone.device=cuda `
    --override audio.interjector.enabled=true `
    --override audio.interjector.echo_alignment_offset_ms=$($v.echo) `
    --override audio.interjector.leading_silence_trim.preserve_ms=$($v.preserve) `
    --override audio.interjector.backchannel_reaction_latency_ms=$($v.back) `
    --override audio.interjector.turn_end_guard_ms=$($v.guard) `
    --override logging.print_to_terminal=true `
    --override logging.summarizer_critic_print_to_terminal=true
}
