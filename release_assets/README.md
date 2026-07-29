# Optional large release assets

Keep large artifacts out of Git history. Attach them to a versioned GitHub release or a permanent archive only when required.

Potential release assets:

- compact per-image Stage-1 predictions;
- selected Stage-1 logits required for an exact Stage-2 rerun;
- compact Stage-2 per-image method outputs;
- an asset-level `SHA256SUMS.txt`.

The repository already includes exact manifests, configurations, environments, completion evidence, fitted Stage-2 parameters, and compact aggregate analyses under `artifacts/frozen_runs/`.
