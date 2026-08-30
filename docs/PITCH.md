# 5-Minute Pitch Script — Return-Risk Scorer

Written to be read aloud at a natural pace. Timed section headers sum to
5:00; the script itself is 776 words, which lands at 4:51–5:10 depending on
delivery pace (160–150 wpm) — read it aloud once before recording and trim
or expand a clause per section rather than reading faster to force 5:00.
Every number in it is pulled from `docs/EVALUATION.md` — practice against
that document, not against memory, if a number needs re-checking before
recording.

---

### [0:00–0:30] The problem

E-commerce merchants lose margin to three different patterns hiding inside
one word: "returns." A customer who wears something once and returns it as
new. A customer who orders five variants intending to keep one. And actual
fraud — empty boxes, stolen goods, tracking manipulation. Most fraud tooling
collapses all three into one binary flag. That's the wrong tool: those three
patterns deserve three different responses, and treating them the same
either punishes honest heavy returners or lets real fraud through disguised
as a policy violation. This project builds a model that tells the three
apart, and a decision layer that routes each one to the right response
instead of a uniform one.

### [0:30–1:45] Lead with the finding, not the score

Here's the number you'd expect me to open with: a LightGBM model on every
feature this dataset ships scores 0.9988 macro-F1 telling these four classes
apart. I'm not opening with it, because it isn't a real result, and finding
that out is the actual headline.

Four hand-written if/else rules — zero training, two threshold checks — score
0.9188 macro-F1 on their own. That's not a coincidence. This dataset is
synthetic, and its generator drew each class's features from bounded ranges
that barely overlap: fraud returns happen in 1 to 5 days, wardrobing in 25 to
55. A model trained on this isn't learning what return abuse looks like —
it's reverse-engineering the rules that generated the data. Reporting 0.998
as an achievement would be the easiest claim in the room to take apart. So
instead of hiding that, this project runs on it: every number from here on
is reported next to the finding that explains it.

### [1:45–2:45] What the response actually is

Two tracks, both labeled honestly. `full` — every feature — is the real
model, reported with this finding attached and never presented as more than
it is. `testbed` deliberately removes the features doing the box-separating,
because I also needed to show the *decision layer* actually working — and on
`full`, it can't: with predictions this confident, a cost policy has nothing
near a boundary to act on. Sweep the block-versus-approve tradeoff from
one extreme to the other on `full` and you get byte-identical decisions
every time. That flat line isn't a bug I fixed — it's the second piece of
evidence for the same finding, and it's in the report as exactly that.

### [2:45–3:45] The decision layer, where it's actually alive

On `testbed`, the interesting axis isn't block-versus-approve — fraud turns
out to be the easiest class to separate even there. It's approve versus soft
friction: how aggressively to fee or flag a customer who might just be a
heavy, honest returner. Sweep that axis and legitimate customers frictioned
moves from under 3% to nearly 25%, abusers caught moves from 85% to
virtually 100% — an order of magnitude of real movement, on the tradeoff a
merchant would actually argue over. That curve, not a single threshold I
picked for them, is the deliverable: the model scores, the merchant chooses
the posture.

### [3:45–4:30] I checked the obvious follow-ups

Is the confusion just class imbalance? I tested SMOTE — it scored *below*
the class-weighted baseline and made Wardrobing worse, so it's documented
and discarded, not silently dropped. Is the model concentrating false
friction on one customer segment? I audited it by order value — there's a
real 2.4x gap, running toward protecting high-value customers, and I'm
naming it rather than only showing the numbers that look clean. And SHAP
confirms the mechanism: on `full`, each class's top driver is the same
near-disjoint feature the leakage finding already flagged; on `testbed`,
the confused classes' top drivers genuinely overlap — a mechanistic account
of *why* they confuse, not just a matrix saying that they do.

### [4:30–5:00] Close

This is a synthetic dataset, and I'm not claiming these numbers transfer to
real merchant data — that caveat is stated up front in the docs, not
discovered by whoever asks first. What I am claiming is the methodology: an
honest leakage screen that catches what a single-feature check would miss, a
cost-calibrated decision layer that adapts to a merchant's actual risk
posture instead of picking one for them, and a model that never executes an
action on its own — it scores, it recommends, a human decides. The finding
that got in the way of a clean 0.998 headline is the best evidence this
process works the way it's supposed to.

---

## Anticipated questions (not part of the timed script)

**"Why not just use the hard-block axis, it's simpler?"** Because it's
measured to be almost inert on this data — 0.00% to 0.23% movement across
the whole posture range — and reporting only that axis would have satisfied
the letter of a cost-sensitivity analysis while measuring the wrong thing.

**"Isn't `testbed` just p-hacking a good result?"** No — it scores *worse*
(0.8967 vs. 0.9988) and is explicitly never called a model. It exists for
exactly one stated reason: to give the decision layer something non-flat to
demonstrate on. The ablation ladder it comes from degrades smoothly with no
natural cut point, which is exactly the argument for why no rung on it can
be a headline result.

**"What would you do with real merchant data?"** Re-run the same Day 1 gate
first — customer-ID viability, timestamp usability, and the leakage sweep at
the corrected tree depth — before trusting any number past it. On real data,
I'd expect the box-separation to disappear and the actual hard problem (do
these classes even separate at all) to show up for the first time.
