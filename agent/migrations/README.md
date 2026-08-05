# Writing a migration

Numbered SQL files, applied in order, each recorded in `schema_migration`.
Write the next number; never edit a file that has been applied.

## A release may add. It may never drop or rename.

Add a table, add a column, add an index, add a constraint that the rows already
satisfy. Do not drop a column, do not rename one, do not narrow a type, and do
not add a `not null` column without a default.

The reason is the order the pipeline runs in. A merge to `main` applies the
migrations **before** it deploys either service, so for a few seconds — and for
much longer if the deployment fails — the old code is running against the new
schema. Old code only keeps working if everything it reads is still there under
the name it knows. That is what makes a failed deployment safe: the system is
left healthy, on the previous image, against a schema it can still read.

Two services write to this database (ADR 0003), and they are deployed one after
the other, so there is never an instant when both are running the new code.
Neither service may assume it is alone.

**A removal waits for a later release**, once nothing reads the column: one
release stops reading it, and a release after that drops it.

## A migration is never reversed

There is no downgrade path — the files are plain SQL and the runner has no
`down`. Reversing a migration that has accepted writes loses those writes.

**A rollback is a redeployment of the previous image**, and it leaves the
schema alone. That works precisely because the migration only added.

A mistake is corrected by writing the next file.

## What is not in here

The checkpointer's tables — `checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations` — belong to LangGraph, which
creates them itself at startup. No file in this directory may name one.
