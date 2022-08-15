from typing import NamedTuple, List


class KeypointTrackerConfig(NamedTuple):
  """Keypoint tracker specific configuration."""

  keypoint_confidence_threshold: float = 0.3
  """The minimum keypoint confidence threshold.

    A keypoint is only compared in the OKS calculation if both the new detected
    keypoint and the
    corresponding track keypoint have confidences above this threshold
  """

  keypoint_falloff: List[float] = [
      0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
      0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
  ]
  """Per-keypoint falloff in OKS calculation."""

  min_number_of_keypoints: int = 4
  """The minimum number of keypoints that are necessary for computing OKS.

    If the number of confident keypoints (between a pose and track) are under
    this value, an OKS of 0.0
    will be given.
  """


class TrackerConfig(NamedTuple):
  """Shared config for pose trackers."""

  keypoint_tracker_params: KeypointTrackerConfig = KeypointTrackerConfig()
  """Keypoint tracker params."""

  max_tracks: int = 12
  """The maximum number of tracks that an internal tracker will maintain."""

  max_age: int = 1000 * 1000
  """ Maximum track lifetime.

    The maximum duration of time (in milliseconds) that a track can exist
    without being
    linked with a new detection before it is removed. Set this value large if
    you would
    like to recover people that are not detected for long stretches of time (at
    the cost
    of potential false re-identifications).
  """

  min_similarity: float = 0.4
  """New poses will only be linked with tracks if the similarity score exceeds this threshold."""
