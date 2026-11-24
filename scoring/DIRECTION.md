# Research Direction

## Research Problem

Design AI recommendations that maximise human-AI team accuracy when:
- humans keep control;
- humans follow advice only when unsure;
- how often they follow it depends on how often the AI agreed with them when
  they were sure.

The difficulty is that being trusted and being useful pull in different
directions, and the AI cannot observe where a given human feels sure.

## Direction

Human-centred assistance. The AI should agree with humans where they are
confident, correct them where they are not, and decide which applies to each
case from the case alone.

## Approach Scope

Valid directions include:
- better models of where humans feel sure;
- better models of what they would decide;
- better outcome models for the cases where they defer;
- better routing or combination of these;
- recommendation rules that optimise the team objective directly.

## Out of Bounds

- Changing the team-decision model.
- Using human judgements or confidences at test time.
- Writing predictions in any format other than the documented one.
