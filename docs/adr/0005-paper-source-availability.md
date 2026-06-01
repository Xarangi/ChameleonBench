# Paper source availability policy

Next Chameleons distinguishes exact public sources, local-manifest sources, and
clean-room regenerated sources.

Exact public sources are pinned in config when available, such as DolusChat and
the public Obfuscated Activations dataset. The paper's final benign synthetic
training set, Apollo/RepE deception corpus, and Synthetic Harmful split were not
found as official packaged Neural Chameleons dataset releases during setup.

Policy:

- Prefer exact public sources when they exist.
- Support local manifests for source files a researcher already has.
- Provide clean-room regeneration for paper-described generated data.
- Allow approximate public substitutes only when the report labels them
  approximate.
- Keep all raw safety text and activations in the Controlled Raw Cache, not git.
