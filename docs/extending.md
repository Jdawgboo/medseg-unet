# Extension seams

Keep `pair_case_ids` and `split_case_ids` stable. New dataset adapters should produce the same `{case_id, image, mask}` contract so pairing and leakage checks remain central.

New losses should accept logits and binary masks. New architectures should preserve `forward(images) -> logits` with shape `[batch, 1, height, width]`. Add shape tests and a short synthetic smoke test before proposing a real-data run.
