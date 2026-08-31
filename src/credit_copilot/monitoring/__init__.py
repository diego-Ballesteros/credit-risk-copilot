"""Monitoring: what the served system looks like once it is answering requests.

`drift` compares the distribution the model was fitted on against the distribution it is
being asked about. Nothing here retrains, adjusts or corrects anything: it measures and
reports, in the same spirit as `data/validator.py`, and what to do about a finding is a
decision for a person.
"""
