## What this changes

<!-- One paragraph. What moved, and why. -->

## The eval set

A case is data in this repository, so a behaviour change and its case travel in
the same pull request. A reviewer then sees the claim next to the evidence.

- [ ] This changes behaviour, and the cases in `agent/evals/cases/` that describe
      it are edited in this pull request.
- [ ] Or: this changes no behaviour the golden set describes.

### If a safety case changed

**A safety case change needs a written reason here.** A weakened refusal must
never pass as a routine test update, so say what boundary moved and why, below.
There is no quarantine or skip list: a case that has started flaking is a case to
fix, not one to park.

<!--
Which case, what it asserted before, what it asserts now, and why the boundary
is where it now is. Delete this section if no safety case changed.
-->

## Checks

- [ ] `pytest` passes in `agent/`, and `npm test` in `gateway/`.
- [ ] The eval gate passes, or the pull request says which case fails and why it
      is right for it to.
