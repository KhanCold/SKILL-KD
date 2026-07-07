[RULE 001] Roman numeral normalization
content: When a question asks for a number or number abbreviation and the source shows a Roman numeral, do answer with the equivalent Arabic numeral only.
why: The failing rollouts returned the visible Roman numeral or its full label, while the expected answer normalized it to the Arabic numeral alone.

[RULE 002] Straight quote transcription
content: When transcribing quoted text from a document for an answer, do use straight double quotation marks instead of curly or single quotation marks.
why: The failing rollouts used single or curly quotation marks around the phrase, while the succeeding target expected straight double quotation marks.

[RULE 003] Ambiguous organization spelling
content: When answering with an organization name that appears as a visually ambiguous near-miss of a well-known name, do correct the ambiguous letters to the recognized spelling while preserving the document's visible word breaks.
why: The failing rollout transcribed the ambiguous misspelling, while a retry over-normalized to the merged brand form; the expected answer corrected the letter but kept the document's spacing.

[RULE 004] Advertised provider brand only
content: When a question asks which provider is advertised and the document shows a distinctive brand beside generic service descriptors or location text, do answer with only the distinctive brand name.
why: The failing rollouts included the generic service descriptor and location after the brand, while the expected answer used only the advertised brand.

[RULE 005] Currency symbol preservation
content: When answering with a monetary amount and the document shows a currency symbol with that amount, do include the currency symbol in the answer.
why: The failing rollout returned only the numeric amount, while the succeeding rollout preserved the visible dollar sign expected by the benchmark.
