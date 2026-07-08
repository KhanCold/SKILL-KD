[RULE 001] Roman numeral normalization
content: When a question asks for a number and the source shows a Roman numeral, do return the equivalent Arabic numeral without surrounding labels.
why: The failing rollouts returned the Roman numeral or its label from the document, while the expected answer normalized it to the Arabic numeral alone.

[RULE 002] Labeled value scope
content: When a question asks for a labeled field value or a specific component within one, do return only the requested complete component or value directly associated with that label, preserving visible symbols such as currency signs, joining currency symbols to their numbers when they are separated only by form layout spacing, and not including other components or neighboring separately labeled fields unless the question asks for them.
why: Earlier failures either appended neighboring fields or omitted visible symbols, while this failing rollout correctly found the amount but inserted a space between the currency symbol and number; the succeeding rollout returned the currency amount with the symbol joined to the number.

[RULE 003] Low-legibility character verification
content: When the requested answer appears in low-legibility handwriting or a skewed annotation, do recheck each character by its visible strokes and distinguish visually similar digits or letters before answering.
why: The failing rollouts returned the correctly scoped label value but misread the low-legibility exhibit number as 20, while the succeeding rollout inspected the annotation and returned 70.

[RULE 004] Complete type phrases
content: When a question asks what kind or type something is and the source shows a multi-word type phrase, do return the complete phrase including the head noun rather than only a modifier.
why: The failing rollout identified the visible phrase but answered only its modifier, while the succeeding rollout returned the full type phrase.
