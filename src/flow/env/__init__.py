"""FLOW execution environment: persistent kernel + Docker runtime + NotebookEnvironment.

Maps to Finch's "pre-built Docker container with a persistent Jupyter kernel". The
agent's code runs **only** here (never on the host), and notebook state persists
across cell edits because a single long-lived interpreter process serves the whole
trajectory.
"""
