[RULE 001] Document field extraction
content: When asked to find a specific field value in a structured document, scan for the exact field label and extract the text immediately adjacent to it, preserving the original casing and spacing.
why: The trajectory showed that locating the labeled field 'Sponsorship Program' in the agreement table and reading the adjacent value directly yielded the correct answer.

[RULE 002] Precise answer extraction from document subjects
content: When extracting an answer from a document, provide only the specific entities directly requested, omitting supplementary context like related product names, usage details, descriptive labels (e.g., 'Page' when asked for a page number), leading articles (e.g., 'the', 'a', 'an'), units of measurement or symbols when already specified in the question (e.g., '(kg)', '%'), values from adjacent/separate fields (e.g., city/state/zip when only the street address field is asked for), or related entities from combined fields (e.g., company name when only the job title is asked for).
why: The trajectory showed that including the unit 'Kg.' when the question already specified '(kg)' resulted in an ANLS score of 0.0 compared to the gold answer '147', including the leading article 'The' before 'basal diet' resulted in a lower ANLS score (0.71) compared to the gold answer 'basal diet', extracting the full combined field value 'President, The Great Western Sugar Company' instead of just the requested title 'President' resulted in an ANLS score of 0.0, and adding '%' to '53' when the question already asked for '%' resulted in an ANLS score of 0.667 compared to the gold answer '53'.

[RULE 003] Number formatting in extraction
content: When extracting a number from a document where it appears with leading punctuation (like a hyphen or dash), remove the punctuation and provide only the numeric value.
why: The trajectory showed that including the hyphen in '-659' caused a mismatch with the gold answer '659', even though the correct abstract number was identified.

[RULE 004] Roman numeral to Arabic numeral conversion
content: When extracting a diagram or figure number expressed as a Roman numeral, convert it to the corresponding Arabic numeral.
why: The trajectory showed that answering 'I' for 'Diagram I' caused a mismatch with the gold answer '1', indicating Roman numerals should be converted to Arabic numerals.

[RULE 005] Handwritten digit disambiguation
content: When extracting handwritten numbers, carefully examine ambiguous digits (such as 2 and 6) by checking for loop closures and tail shapes to avoid misrecognition.
why: The trajectory showed that misreading a handwritten '6' as '2' resulted in the incorrect answer '4-5-2' instead of the gold '4-5-6'.

[RULE 006] Quotation mark normalization
content: When extracting quoted terms or expressions from a document, normalize quotation marks to standard straight double quotes (") rather than preserving typographic curly quotes, as the expected answer format typically uses straight quotes.
why: The trajectory showed that preserving curly/smart quotes ("Redefining") instead of converting to straight quotes ("Redefining") resulted in a slightly lower ANLS score (0.93) compared to the gold answer which used straight quotes.

[RULE 007] Printed text letter disambiguation
content: When extracting text from a document, carefully verify each letter in proper nouns and company names, paying attention to easily confused letter pairs (such as 'a' and 'o') to avoid misrecognition.
why: The trajectory showed that misreading 'Astra' as 'Astro' in a company name resulted in an ANLS score of 0.917 instead of a perfect match with the gold answer 'Astra Zeneca'.

[RULE 008] Brand name extraction from logos
content: When extracting a brand name from a logo or advertisement where the text appears with visual spacing but represents a single compound brand name, extract it as one word without spaces.
why: The trajectory showed that extracting 'Fair way' as two words from the FAIR WAY RENT-A-CAR logo resulted in an ANLS score of 0.875 compared to the gold answer 'fairway' (one word), indicating that brand names in logos should be extracted as single compound words.

[RULE 009] Currency symbol inclusion in monetary extraction
content: When extracting a monetary amount from a document where a currency symbol (like $) is present adjacent to the number, include the currency symbol directly attached to the number without spaces in the answer unless the question explicitly specifies the currency unit.
why: The trajectory showed that extracting "$ 50.00" with a space between the dollar sign and the number resulted in a lower ANLS score (0.857) compared to the gold answer "$50.00", indicating that currency symbols should be preserved and directly attached to the numeric value without intervening spaces.
