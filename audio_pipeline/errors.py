"""Error taxonomy for the audio pipeline."""


class AudioPipelineError(RuntimeError):
    """Base class for audio-pipeline failures."""


class RetryableAudioStageError(AudioPipelineError):
    """A transient failure that may succeed on retry."""


class NonRetryableAudioStageError(AudioPipelineError):
    """A terminal failure that should not be retried."""


class DependencyMissingError(NonRetryableAudioStageError):
    """Required external binary or Python package is unavailable."""


class ArtifactWriteError(NonRetryableAudioStageError):
    """Output artifact could not be safely persisted."""
