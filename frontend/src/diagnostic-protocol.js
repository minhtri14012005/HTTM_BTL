// Clock projection uses the backend's frozen protocol and capture-time boundaries.
export function phaseAt(protocol, elapsed) {
  if (!protocol || elapsed < 0 || elapsed >= protocol.duration_seconds)
    return null;
  const stage = protocol.stages.find(
    (s) => elapsed >= s.start && elapsed < s.end,
  );
  if (!stage) return null;
  return {
    ...stage,
    transition: elapsed < stage.start + stage.transition_seconds,
    remaining_seconds: Math.max(0, stage.end - elapsed),
    transition_remaining_seconds: Math.max(
      0,
      stage.start + stage.transition_seconds - elapsed,
    ),
  };
}
