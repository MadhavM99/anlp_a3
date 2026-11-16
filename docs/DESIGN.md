# Mamba–DeltaNet Hybrid: Design and Hypothesis

## What the data says about the cipher
Measured on the first 20k training documents, after stripping the inserted punctuation (`!'+,-./:;<=>?[]_{}`):

- Stripped ciphertext and plaintext have the same length in 100% of documents. So decryption is a per-position mapping from 63 cipher symbols to 27 plaintext symbols.
- It is not a fixed substitution. Knowing only the cipher character, the plaintext character is right 17% of the time (most likely plaintext character per cipher symbol).
- Position 0 is an exact substitution: the cipher character alone gives the plaintext character 100% of the time. That falls to 78%, 67% and 39% at positions 1–3, and to about 22% by position 7. The key is built up from what came before.
- Earlier ciphertext adds nothing: adding any single earlier cipher character (1–6 back) leaves the rate at 17%.
- Earlier plaintext does help: the rate rises to 27% with p(i−1), 21% with p(i−2) and 17% with p(i−3) added. With p(i−1), p(i−2) and p(i−3) together it reaches 48%.

So decrypting position i needs two things:
1. **A running key state**, built mainly from the last few plaintext characters, which the model has to infer itself.
2. **A lookup** from (cipher character, key state) to the plaintext character.

These depend on each other: the state needs earlier plaintext, and earlier plaintext needs the lookup.

## Arrangement
6 layers, interleaved, Mamba first:

`Mamba → DeltaNet → Mamba → DeltaNet → Mamba → DeltaNet`

Every layer is pre-norm with a residual connection. Each mixer is implemented as described in its paper, with no attention anywhere. The input is the same for all four models (4 characters of left context plus a 64-character patch), they produce one 27-way prediction per position, and the hybrid is matched to the other models' parameter count to within about 8%.

## Why this arrangement should work
- **Mamba builds the running state.** Its selective state space model, with a step size and B/C that depend on the input, plus a short causal convolution, can choose what recent history to keep or forget. That matches a key that depends mostly on the last 2–3 characters and fades after that.
- **DeltaNet does the lookup.** The delta rule, S ← S + β(v − S k)kᵀ, stores key→value associations and overwrites the stale ones. Each position's (cipher character, state) pair can be read as a key and the plaintext character as the value. When the key state changes, the error-correcting update replaces the old association instead of piling up interference, which a plain linear-attention state would do.
- **Interleaving repeats the state → lookup → state cycle.** Decoding is recursive: each plaintext guess changes the state for the next position. Alternating the two mixers gives three rounds of refinement, with each Mamba layer folding the latest plaintext estimate back into the running state. Putting all Mamba layers first would allow only one round, so mistakes in the state could never be corrected after the lookup.
- **Mamba goes first** because position 0 is a pure substitution and every later position needs a state before any lookup makes sense.

## Predictions to check against the results
1. The hybrid beats both pure Mamba and pure DeltaNet in character accuracy, at matched parameters and the same 10k-step budget.
2. Pure DeltaNet does worse than pure Mamba, because building the state with only the delta rule is weaker than a selective SSM.
3. The Transformer is competitive. With 68-token windows its quadratic cost doesn't matter, and attention can reach the previous characters directly. The hybrid's advantage should show up as better use of the same budget, not necessarily a higher final score.
4. All models are weakest in the first few positions of each patch, where the 4-character left context gives only partial key state. The hybrid's gap over pure Mamba and DeltaNet should be largest in the later positions of a patch, where the repeated state → lookup cycle has room to work.

If the hybrid fails to beat the better of pure Mamba and pure DeltaNet, the hypothesis that the two do complementary jobs (state versus lookup) is rejected.
