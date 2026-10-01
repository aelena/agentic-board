# Board vs single model

Board `startup` vs one call of the same model, 20 ideas, judged blind twice each (order swapped) by `openai/gpt-4.1`. Model under test: `openai/gpt-4o`.

**Board wins 13, single wins 0, ties 0, split decisions 7.**

## Mean scores (1-10)

| criterion | board | single |
|---|---|---|
| specificity | 8.88 | 7.83 |
| coverage | 9.65 | 8.75 |
| actionability | 9.07 | 8.03 |
| honesty | 9.78 | 8.93 |
| overall | 9.6 | 8.38 |

| | board | single |
|---|---|---|
| mean seconds | 28.43 | 15.29 |
| mean output tokens (approx.) | 3327.65 | 2224.8 |
| LLM calls | 12 agent calls | 1 |

## Per idea

| idea | verdict | judgement 1 | judgement 2 |
|---|---|---|---|
| Local legal document comparison | **board** | board (10 vs 8) | board (10 vs 9) |
| Battery health analytics for delivery fleets | **board** | board (10 vs 8) | board (10 vs 9) |
| Voice-recorded family memoir app | **board** | board (10 vs 8) | board (10 vs 9) |
| Port crane KPI platform | **board** | board (9 vs 7) | board (10 vs 9) |
| Adaptive maths tutor for 10 to 14 year olds | **split** | board (10 vs 8) | single (9 vs 10) |
| EU AI Act compliance copilot for SMEs | **split** | board (10 vs 8) | single (8 vs 9) |
| Operating system for container farms | **board** | board (10 vs 8) | board (10 vs 10) |
| Autonomous tax filing for Spanish freelancers | **board** | board (10 vs 7) | board (10 vs 9) |
| Frost prediction for vineyards | **board** | board (10 vs 8) | board (10 vs 8) |
| Zero-retention meeting notes | **board** | board (10 vs 8) | board (9 vs 9) |
| Agentic boardroom for decisions | **board** | board (10 vs 7) | board (10 vs 8) |
| Sneaker authentication by phone camera | **split** | board (10 vs 8) | single (8 vs 10) |
| Hospital bed flow prediction | **split** | board (10 vs 9) | single (8 vs 9) |
| Contract negotiation assistant for creators | **split** | board (10 vs 7) | single (9 vs 10) |
| Maintenance manual assistant for factories | **board** | board (10 vs 8) | board (10 vs 8) |
| Micro-pension for gig workers | **board** | board (10 vs 8) | board (9 vs 8) |
| Kitchen display with demand forecasting | **split** | board (10 vs 8) | single (8 vs 9) |
| Licence governance for AI-generated code | **board** | board (10 vs 8) | board (10 vs 8) |
| Companion robot rental for care homes | **board** | board (10 vs 8) | board (10 vs 9) |
| Developer relations analytics | **split** | board (9 vs 7) | single (8 vs 9) |
