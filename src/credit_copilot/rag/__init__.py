"""Retrieval over the normative corpus: documents, chunking, embeddings and the vector store.

The four modules are deliberately layered so that each can be tested without the one below it.
`documents` turns corpus files into structural units and depends on nothing; `chunking` turns
units into indexable fragments and depends only on `documents`, which is what lets the whole
strategy be tested without downloading a model; `embeddings` owns the model; and `vectorstore`
owns persistence. Nothing here calls a language model - the copilot's tools and graph are a
separate concern and a later step.
"""
