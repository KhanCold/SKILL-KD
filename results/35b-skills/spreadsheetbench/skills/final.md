[RULE 001] No-data placeholder deletion
content: When deleting rows based on blank or placeholder criteria in spreadsheet columns, do inspect the actual column values and treat trimmed no-data display placeholders such as dashes and formatted zero/currency strings as matching the deletion criteria when both required cells lack actual data.
why: The failing rollout only treated the explicitly listed blank and zero/currency tokens as deletable, while the successful behavior would also remove rows whose relevant cells used no-data placeholders like dashes so rows with no actual data did not remain.

[RULE 002] Formula-value materialization
content: When saving spreadsheet changes that affect or rely on formula-generated values in cells that must be read as final outputs, do materialize the intended displayed numeric results as literal values by coercing numeric text or recomputing from the correct source or pre-change data rather than relying on formulas to recalculate.
why: The earlier failing rollout did not produce reliable formula-derived totals, and the current failing rollouts either left formulas with unusable cached values or recomputed them after changing precedent cells, while the successful behavior would write the intended displayed values from the correct source state.

[RULE 003] Column-major whole-number extraction
content: When filtering whole-number values from a rectangular spreadsheet range into a bounded output range with a fixed maximum count per row, do scan the source by columns from top to bottom, exclude non-integers including decimal text, and wrap accepted values across each output row up to the stated width.
why: The failing rollout produced no output workbook, while the successful rollout filtered only integer-valued entries, scanned the source column by column, and filled the destination with no more than the requested count per row.

[RULE 004] Marker-preceding row deletion
content: When asked to remove all rows above the first occurrence of a marker in a spreadsheet column, do find the first exact trimmed match in that column, delete only the rows before it, and save the workbook to the requested output path.
why: The failing rollout produced no workbook, while the successful rollout located the first marker occurrence in the relevant column, deleted every preceding row, and saved the modified workbook.

[RULE 005] Delimiter-key lookup labels
content: When filling an answer range by looking up multiple delimiter-separated keys from each source cell against a mapping table, do split and trim nonempty keys, map each key, de-duplicate mapped labels, sort them in the requested order, join multiple labels with the requested separator, and save the workbook.
why: The failing rollout produced no output workbook, while the successful rollout split semicolon-delimited cell values, ignored trailing empty parts, mapped each key through the lookup table, returned unique groups in ascending order, and saved the requested workbook.

[RULE 006] Date-table fallback labels
content: When filling cells from source dates using an auxiliary table of special-date labels, do normalize spreadsheet date and datetime values to date-only keys, write the mapped label for matched dates, write the requested day-of-week abbreviation for unmatched dates, and save the workbook.
why: The failing rollout produced no output workbook, while the successful rollout normalized the main and lookup date values, wrote the public-holiday type for matched dates, wrote weekday abbreviations for nonmatches, and saved the workbook.

[RULE 007] Cross-sheet key row deletion
content: When deleting rows from multiple spreadsheet sheets based on matching identifier and numeric-value fields, do normalize identifiers and currency/number values into composite keys, delete only the paired matching row count for each key from every involved sheet in descending row order, and save the workbook.
why: The failing rollout produced no output workbook, while the successful rollout matched rows across the two sheets using normalized reference and amount pairs, handled duplicate keys by deleting only paired counts, deleted rows bottom-up, and saved the workbook.

[RULE 008] Grouped-status conditional labels
content: When filling labels based on whether each identifier appears with exclusive or combined normalized status categories and whether a companion field is populated, do aggregate statuses by identifier first, apply combined and exclusive group-level rules before row-level populated-field fallbacks, and save the workbook.
why: The failing rollout produced no output workbook, while the successful rollout grouped rows by identifier to detect exclusive versus combined statuses, then used the current row's status and date presence only for the remaining fallback labels.

[RULE 009] Predefined crosstab lookup fill
content: When filling a predefined cross-tab answer range from source rows containing row keys, column keys, and values, do use the existing answer row labels and column headers as lookup keys, write only inside the requested range, and fill missing key pairs with the requested blank or placeholder.
why: The failing rollout derived row and column order from the source data and misaligned a value, while the successful rollout used the pre-defined row labels and headers in the answer sheet to look up each key pair.

[RULE 010] Header-located column move
content: When moving a spreadsheet column whose source position may vary into a requested target position, do find the first exact trimmed header match, preserve the source column values and formatting, delete the original column before inserting it at the target position when needed, and save the workbook.
why: The failing rollout produced no workbook, while the successful rollout searched for the changing header location, moved the entire matching column to the requested position while preserving formatting, and saved the output workbook.

[RULE 011] Sanitized numeric cell coercion
content: When sanitizing spreadsheet cell values by removing nonnumeric characters while allowing numeric punctuation, do treat grouping separators as removable formatting, preserve decimal separators for parsing, and write valid cleaned results as numeric cell values rather than text.
why: The failing rollouts kept a cleaned string containing a grouping comma, while the expected workbook removed formatting commas during parsing and stored the cleaned amount as a numeric value.
