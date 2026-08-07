"""The golden dataset and the eval gate.

`dataset` and `gate` import nothing but the standard library, because the judge
runs in an environment of its own: ragas 0.4 needs a `langchain-community` that
the agent's own `langchain` 1.x cannot sit beside, so the two halves of a run are
two processes rather than one dependency set that resolves to neither.
"""
