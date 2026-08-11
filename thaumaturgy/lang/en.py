"""English strings."""

# Guards, shown by a service when it refuses and by a page that checks first.
NO_MODEL = "Load a model on the Model page first."
CHAT_BUSY = "Wait for the current reply to finish."
SPAN_BUSY = "Wait for the current span to finish."
CHAT_TOO_LONG = ("The most recent messages already fill the context window on "
                 "their own. Shorten or delete one, or start a new chat.")

# Chat composer, which points at the Model page while nothing is loaded.
CHAT_PLACEHOLDER = "Message…  (Ctrl+Enter to send)"
CHAT_PLACEHOLDER_NO_MODEL = "Load model from the Model screen to start."

# Compaction: asking to condense a chat, and explaining what that does.
COMPACT_ASK = ("This chat has outgrown the context window ({used:,} of "
               "{total:,} tokens). Summarize the earliest {folded} messages to "
               "carry on?")
COMPACT_ASK_DETAIL = ("Your transcript is untouched — this only changes what "
                      "the model is sent. The chat is unavailable while it runs.")
COMPACT_FORCE_ASK = ("Compact this chat now, before it needs it? The oldest "
                     "messages are summarized and the recap is sent in their "
                     "place.")
COMPACT_REDO_ASK = ("Write the recap again over the same {covers} messages, "
                    "replacing the current one? Use this after changing the "
                    "recap budget or the instructions in compaction.yaml.")
COMPACT_RUNNING = "Summarizing the earlier messages…"
COMPACT_PASS = "Summarizing part {step} of {total}…"
COMPACT_DONE = "Folded {folded} messages into a recap."
COMPACT_STILL_TOO_LONG = ("Still too long after compacting. Shorten the message "
                          "or start a new chat.")
COMPACT_NOT_NEEDED = "This chat already fits; nothing to do."
COMPACT_DIVIDER = "Context compacted — {covers} earlier messages summarized"
NO_RECAP = "This chat hasn't been compacted, so there is no recap yet."

RECAP_BUDGET_HELP = (
    "How long a recap may run when a chat is compacted. No model reports how "
    "long a summary it will write, so this is a ceiling to tune rather than a "
    "target it will hit. It is capped at 15% of the context window, since the "
    "recap is resent with every later turn: beyond that it crowds out the "
    "recent messages it was written to make room for."
)

# Settings page help.
COMPACTION_HELP = (
    "When a chat outgrows the model's context window, its oldest turns are "
    "folded into a recap that the model is sent in their place. Nothing is "
    "removed from the chat itself. The divider marks where the recap takes "
    "over and can be opened to read it. How the recap is written is set in "
    "compaction.yaml in the data directory."
)
COMPACTION_STRATEGY_HELP = (
    "A model asked to condense 40,000 tokens writes about as much as one asked "
    "to condense 4,000, so a single pass over a long chat loses most of the "
    "detail. Several passes summarize it a span at a time and keep the parts in "
    "order, spending the whole recap budget instead of a fraction of it. Each "
    "pass is a generation of its own, so a long chat can take minutes, and the "
    "result reads as sections rather than continuous prose."
)

LOG_HELP = (
    "Off by default. When set, llama-server's output is mirrored to "
    "llama-server.log and each editing attempt is appended to editing.jsonl. "
    "The in-app llama.cpp panel only keeps the last few hundred lines, so a "
    "log directory is what lets you look at a load or a timing after the fact."
)
