from langchain_core.prompts import PromptTemplate

from typing import Final


TERM_EXTRACTION: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "input",
        "term_defs",
        "domain_knowledge",
    ],
    template="""
You are a specialized system for identifying entities (mentions of objects, people,
concepts, etc.) in text that match specific logical term definitions.
Your task is to analyze the input text and identify text spans that correspond to defined terms.

# Input Text
{input}

# Terms Definitions
{term_defs}

# Domain Context
The following provides domain-specific context to help you understand the text:
{domain_knowledge}

# Instructions

1. Carefully read the input text and term definitions.
2. For each term definition, identify ALL text spans in the input that could represent that term.
3. For each identified entity:
   - Extract the exact text span from the input
   - Assign it to the appropriate term
   - Provide a confidence score (0.0-1.0) indicating how well it matches the term definition
   - Include a brief explanation of why this text span matches the term definition
4. If a definition has multiple possible text spans, include all of them as separate terms.
5. If no text span matches a term definition, do not include it in the results.

# Output Format

Return a structured JSON output with the following format:

```json
{{
  "terms": [
    {{
      "text": "exact text from input",
      "term_name": "name of the matching term",
      "confidence": 0.9,
      "explanation": "This text represents the term because..."
    }},
    ...
  ]
}}
```

# Example

## Fallacy Detection Example
Input: "Sometimes flu vaccines don't work; therefore vaccines are useless"
Terms Definitions:
- claim: The initial statement that serves as a premise
- implication: The conclusion drawn from the claim

Output:
```json
{{
  "terms": [
    {{
      "text": "Sometimes flu vaccines don't work",
      "term_name": "claim",
      "confidence": 0.95,
      "explanation": "This is the premise statement that presents a limited factual assertion about flu vaccines sometimes being ineffective."
    }},
    {{
      "text": "vaccines are useless",
      "term_name": "implication",
      "confidence": 0.9,
      "explanation": "This is the conclusion being drawn from the premise, making a broad claim about all vaccines being useless. The 'therefore' signals this is the implication."
    }}
  ]
}}
```

Now, analyze the provided input text and identify all entities that match the term definitions.
""",
)


PREDICATE_EXTRACTION: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "input",
        "terms",
        "predicate_defs",
        "domain_knowledge",
    ],
    template="""
You are a specialized system for identifying predicates in text that match specific predefined predicate definitions.
Your task is to analyze the input text and determine which predicates apply to the terms previously identified.

# Input Text
{input}

# Identified Terms
The following terms have been extracted from the input text:
{terms}

# Predicate Definitions
{predicate_defs}

# Domain Context
The following provides domain-specific context to help you understand the text:
{domain_knowledge}

# Instructions

1. Carefully read the input text, identified terms, and predicate definitions.
2. For each predicate definition, determine if it applies to any of the identified terms.
3. A predicate can apply to multiple terms, and multiple predicate can apply to the same term.
4. For each predicate application:
   - Identify the predicate name from the predicate definitions
   - List the term variables that are arguments to this predicate
   - Provide a confidence score (0.0-1.0) indicating how strongly the predicate applies
   - Include a brief explanation of why this predicate applies to these terms
5. Be precise about which terms are arguments to which predicates.
6. In the "arguments" field, use ONLY the term names (e.g., "p", "a", "e", "i"), not term text.
   Never list the same variable name more than once in a single predicate's arguments.
7. Mind which variables the predicates are referring to. In case the predicate applies
   to multiple terms, be careful when referring to 'Identified Terms' - if
   necessary introduce new term variables for another entities.
8. On the argument list, a variable of the same name may appear at most once.
9. If a predicate doesn't apply to any terms, do not include it in the results.

# Output Format

Return a structured JSON output with the following format:

```json
{{
  "predicates": [
    {{
      "name": "predicate_name",
      "arguments": ["t1", "t2", ...], // ONLY term names, never the full text
      "confidence": 0.9,
      "explanation": "This predicate applies because..."
    }},
    ...
  ]
}}
```
""",
)


SD_DIRECT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "task_domain",
        "task_name",
        "domain_knowledge",
        "predicate",
        "input",
        "term_defs",
        "predicate_defs",
    ],
    template="""
You are a specialized system for evaluating identifying relevant terms and predicates
in the text and evaluating task predicate.

# About The Task

We are solving the following task:

Domain: {task_domain}
Task name: {task_name}
Domain knowledge:
{domain_knowledge}

# Input Text
{input}

# Task Predicate

You need analyze the Input Text, identify relevant terms and predicates and then
evaluate the following first-order-logic expression and finally return true/false result:

{predicate}

# Terms Definitions

{term_defs}

## Instructions for Term Extraction

1. Carefully read the input text and term definitions.
2. For each term definition, identify ALL text spans in the input that could represent that term.
3. For each identified entity:
   - Extract the exact text span from the input
   - Assign it to the appropriate term
   - Provide a confidence score (0.0-1.0) indicating how well it matches the term definition
   - Include a brief explanation of why this text span matches the term definition
4. If a definition has multiple possible text spans, include all of them as separate terms.
5. If no text span matches a term definition, do not include it in the results.

# Predicate Definitions

{predicate_defs}

## Instructions for Predicate Extraction

1. Carefully read the input text, identified terms, and predicate definitions.
2. For each predicate definition, determine if it applies to any of the identified terms.
3. A predicate can apply to multiple terms, and multiple predicate can apply to the same term.
4. For each predicate application:
   - Identify the predicate name from the predicate definitions
   - List the term variables that are arguments to this predicate
   - Provide a confidence score (0.0-1.0) indicating how strongly the predicate applies
   - Include a brief explanation of why this predicate applies to these terms
5. Be precise about which terms are arguments to which predicates.
6. In the "arguments" field, use ONLY the term names (e.g., "p", "a", "e", "i"), not term text.
   Never list the same variable name more than once in a single predicate's arguments.
7. Mind which variables the predicates are referring to. In case the predicate applies
   to multiple terms, be careful when referring to 'Identified Terms' - if
   necessary introduce new term variables for another entities.
8. On the argument list, a variable of the same name may appear at most once.
9. If a predicate doesn't apply to any terms, do not include it in the results.

# Output Format

Return a structured JSON output with the following format:

```json
{{
    "is_satisfied": true/false, // result of evaluation of task predicate
    "terms": [ // list all identified predicates - all that you identified before evaluating the task predicate
        {{
            "text": "...relevant text...", // relevant text
            "term_name": "i", // name of a term
            "confidence": 0.87, // how confident you are about the term
            "explanation": "..."
        }}
    ],
    "predicates": [ // list all identified predicates - all that you identified before evaluating the task predicate
        {{
            "name": "predicate_name",
            "arguments": ["t1", "t2", ...], // ONLY term names, never the full text
            "confidence": 0.9,
            "explanation": "This predicate applies because..."
        }}
    ],
}}
```
""",
)
