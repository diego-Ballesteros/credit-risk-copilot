"""The copilot: four tools with strict contracts, and the graph that orchestrates them.

Four modules, layered so each is testable without the one above it.

`state` declares what the graph carries and depends on nothing else in the package, which is
what lets the tools import the citation record without importing the graph. `tools` owns the
contracts and the execution, and it is the only place the model artefact and the index are
touched; it can be exercised end to end with hand-built doubles and no network. `prompts`
holds text and the reason each clause is in it. `graph` wires the nodes and is the only
module that talks to a language model.

The separation that matters most is inside `tools`: **the language model proposes a call and
the code validates and executes it.** Nothing here lets a proposal reach the production
artefact without passing a Pydantic contract first, and the schemas the model sees do not
contain the applicant's attributes at all - those are bound from the state, so a tool cannot
be run on numbers a language model produced.
"""
