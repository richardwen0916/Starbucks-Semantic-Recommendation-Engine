# Starbucks Semantic Product Recommendation Engine

An AI-powered search engine for the Starbucks menu. It takes natural-language requests such as *"I want a low-calorie iced coffee under $5"* or *"Do you have a decaf option for vegans?"* and returns the most relevant drinks, ranked.

**Result:** Average **NDCG@5 = 0.8560** on the training queries.

## Business Problem

The Starbucks menu has hundreds of variations in temperature, caffeine, and dietary needs. Customers describe what they want in their own words, but keyword search (SQL `LIKE`, string matching) can't capture what they mean:

- "decaf" means **0 caffeine**
- "frozen" means **blended**
- "pick-me-up" means **high caffeine / espresso**

The goal is to understand the constraints in a natural-language query and rank products by semantic relevance, which improves conversion and the overall search experience.

## Architecture: Three-Stage Hybrid Retrieval

```
User query ──► [1] LLM Intent Extraction ──► JSON constraints
                    (Gemini 2.5 Flash)            │
                                                  ▼
products.csv ──────────────────────────► [2] Deterministic Filtering (Pandas)
                                                  │  valid candidates
                                                  ▼
                                         [3] Semantic Ranking
                                             (text-embedding-004 + cosine)
                                                  │
                                                  ▼
                                           Top-K ranked products
```

### Stage 1: Intent Understanding (LLM Extraction)
- **Model:** `gemini-2.5-flash`, prompted to act as a "Virtual Barista"
- **Output:** a structured JSON object enforced with `response_mime_type: "application/json"` and a strict response schema:

| Field | Type | Values |
|---|---|---|
| `category` | array | `brewed`, `cold_brew`, `espresso`, `frappuccino`, `refresher`, `tea` |
| `temperature` | string | `hot`, `iced`, `blended` |
| `caffeine_level` | string | `none`, `low`, `medium`, `high` |
| `max_calories`, `max_sugar` | integer | — |
| `max_price` | number | — |
| `dairy_free`, `vegan` | boolean | — |

- **Prompt rules** encode business logic. For example:
  - Only fields mentioned in the query are returned.
  - Generic "coffee" expands to `["brewed", "cold_brew", "espresso"]`.
  - "latte" or "cappuccino" maps to `espresso`.
  - "frozen" maps to `blended`, and "decaf" maps to `caffeine_level: none`.

### Stage 2: Deterministic Filtering (Hard Constraints)
Pandas boolean filters apply the extracted constraints to `products.csv`. This guarantees the business rules are met: a vegan user never sees dairy products, no matter how semantically similar they are. Caffeine in mg is bucketed into levels: none ≤ 15, low ≤ 70, medium < 170, high ≥ 170. If the filters return no products, the constraints are relaxed.

### Stage 3: Semantic Ranking (Relevance)
- **Embeddings:** `models/text-embedding-004`, applied to a combined field: `Product: <name>; Category: <category>; Desc: <description>`
- **Similarity:** cosine similarity between the query embedding and each candidate's embedding
- This handles synonyms and vague requests that keyword search misses. For example, "pick-me-up" lands close to espresso and high-caffeine drinks.

## Why These Choices

- **Gemini Flash:** low latency and cost, which a real-time search feature needs.
- **Forced JSON output:** reliability matters more than creativity here. JSON output removes parsing errors for the downstream Pandas filters.
- **Hybrid design:** hard filters guarantee correctness on the user's constraints, and embeddings handle relevance. Neither approach does both on its own.

## Evaluation

- **Metric:** NDCG@5 (Normalized Discounted Cumulative Gain). It rewards putting the right products **at the top** of the list, not just somewhere in it.
- **Result:** **0.8560** average NDCG on `queries_train.csv`. In practice, the most relevant products usually appear at the top of the results, which reduces search friction.

## Repository Structure

```
.
├── Starbucks_Project.ipynb   # Full pipeline: extraction, filtering, ranking, evaluation
└── README.md
```

## Data

The datasets are **not public** and are not included in this repository. The notebook expects these files in the working directory:

- `products.csv`: product catalog with `product_id`, `name`, `category`, `temperature`, `caffeine_mg`, `calories`, `sugar_g`, dairy/vegan flags, `description`, and `price`
- `queries_train.csv`: labeled queries with ranked `relevant_products`
- `queries_test.csv`: unlabeled test queries

## Getting Started

```bash
pip install pandas numpy scikit-learn google-generativeai pydantic
export GOOGLE_API_KEY="your-gemini-api-key"
jupyter notebook Starbucks_Project.ipynb
```

The notebook reads the API key from the `GOOGLE_API_KEY` environment variable. Free-tier quotas may slow down evaluation; the code includes retry and backoff for rate limits.
