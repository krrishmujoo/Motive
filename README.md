Motive

Motive is an intent-aware e-commerce recommendation system that shows why each item was recommended. It is built on the anonymized Retailrocket dataset and includes a Python recommendation backend, a FastAPI service, and a React frontend.

Most recommenders return a ranked list and stop there. Motive keeps the evidence behind each result: which item from the shopper's history led to it, how similar the two items are, whether shoppers viewed them in the same sessions, how popular the item is, where the learned model ranked it, and whether a natural-language request moved it. When a request includes something the data cannot verify, such as a brand or a price limit, Motive says so instead of pretending it was applied.

What Motive Does
Recommends products using two separate retrieval signals: catalog-property similarity and behavioral co-visitation.
Reranks candidates with a trained gradient-boosted tree model.
Handles new, low-history, and established shoppers differently.
Uses Anthropic's Claude to turn requests like "show me something different" into structured preferences. Claude never picks the items.
Applies supported preferences as a bounded adjustment on top of the learned ranking.
Flags constraints it cannot verify, such as brand, price, and use case.
Generates explanations from recorded evidence, not generic templates.
Ships with 37 automated tests and a 158-check product regression suite.
An Important Note on the Data

Retailrocket products are anonymized. Items are numeric IDs, and most catalog properties are hashed values. The dataset does not reliably tell you what a product is called, who makes it, what it costs, or what it is for.

Because of this, Motive does not show product names, brands, prices, specifications, images, or retailer links. The colored marks on recommendation cards are generated from item IDs and are not product images. If a shopper asks for "Sony headphones under $300," Motive can understand the request, but it will not claim that any result is a Sony product, costs under $300, or is a pair of headphones.

This constraint shaped most of the design. The explanations and the handling of unsupported requests exist because honest output was a requirement, not an afterthought.

Dataset
	Approximate size
Interaction rows	2.75 million
Event types	view, add-to-cart, transaction
Visitors	1.4 million
Products in interaction data	235,000
Products with catalog metadata	400,000+
Time span	May to September 2015

The data is very sparse. Around 90% of September users have no history in the earlier training period, which makes them cold-start users.

Architecture
                      Natural-language request (optional)
                                   │
                                   ▼
                        Claude intent parser
                     (structured UserIntent only)
                                   │
Shopper ID ──► User routing ───────┼─────────────────────────────┐
                 │                 │                             │
     new user    │    known user   │                             │
        ▼        ▼                 │                             │
  Weighted    Candidate generation │                             │
  popularity  (content + co-visit) │                             │
        │        │                 │                             │
        │        ▼                 │                             │
        │   Tree reranker          │                             │
        │        │                 ▼                             │
        │        └──► Intent blend (85% learned / 15% intent)    │
        │                          │                             │
        └──────────────────────────┴──► Evidence + explanations ◄┘
                                          │
                                          ▼
                                FastAPI ──► React UI
User Routing

Shoppers are routed by how many items they have interacted with before:

Segment	Historical items	Strategy
New	0	Weighted catalog popularity
Low-history	1 to 2	Personalized retrieval and reranking
Established	3 or more	Personalized retrieval and reranking
Cold Start

New shoppers have no behavioral evidence, so they receive popularity-based recommendations. Popularity is weighted by interaction type: a view counts as 1, an add-to-cart as 3, and a transaction as 5.

A new shopper's request is still parsed, but it does not currently change the popularity ordering. The UI states this directly so the shopper is not misled into thinking their request changed the results.

Candidate Generation

For known shoppers, Motive builds a pool of roughly 200 candidates from two sources.

Content similarity. Each product's catalog metadata is converted into tokens that preserve which property each value came from. These form a sparse item-feature matrix of about 417,053 products, 161,379 features, and 22.5 million non-zero entries. Rows are normalized, so similarity between two items is a cosine-style score. For example, item 460429's closest neighbor is item 100656, with a similarity of about 0.92.

Co-visitation. Items viewed by the same shopper within a single session (using a 30-minute inactivity boundary) are linked. The May to July build produced about 1.19 million sessions, of which 183,446 contained more than one item. These yielded 1.41 million directed item pairs covering 96,460 items.

The two signals were kept separate on purpose. Across 100 sampled items, their top-50 neighbor lists shared a mean of about 1.92 items (median 1), with a mean Jaccard overlap of about 0.031. They capture different relationships, so combining them gives the reranker more to work with.

The final configuration uses up to 20 of a shopper's strongest history items, pulls 50 content neighbors and 50 co-visitation neighbors for each, removes items the shopper has already seen, and records exactly which history item produced each candidate. Popularity is used as a ranking feature but not as an extra candidate source for known shoppers.

Reranker

Each candidate gets a feature vector built from its retrieval evidence, including content score, maximum content similarity, co-visitation score, the number of history items that support it through each signal, popularity (raw and log-scaled), the shopper's history size, and an established-user flag.

A logistic regression reranker was tested first. The final model is scikit-learn's HistGradientBoostingClassifier:

HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=200,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42,
)
Natural-Language Intent

Claude converts a free-text request into a structured UserIntent object. It does not see the catalog and does not choose, add, or remove recommendations. Claude never picks the items.

Four ranking preferences are supported and can affect ranking:

familiar: closer to what the shopper has already viewed
exploratory: further from the shopper's usual items
popular: favors widely engaged items
niche: favors less common items

Other fields can be parsed but not verified against anonymized data: brand, minimum and maximum price, use case, priority features, and features to avoid. These are returned as unverifiable constraints and shown in the UI.

Blending Intent With the Learned Ranking

The learned ranking stays in charge. Supported intent is applied afterward as a limited adjustment:

final score = 0.85 × normalized reranker score + 0.15 × normalized intent adjustment

An earlier version had a bug here. Raw tree probabilities were very small, while intent scores sat near a 0 to 1 range. Even with an 85/15 blend on paper, intent effectively replaced the model's ranking. I found this while testing explanations, because rank movements looked far larger than the weights should allow. The fix was to normalize both scores independently before blending.

After the fix, intent nudges results rather than overriding them. In one test, item 323403 moved from rank 5 to rank 4, item 186360 moved from 4 to 5, and the top three learned results stayed in the top three.

The 85/15 split is a hand-picked product choice. It was not learned or tuned offline.

Explainability

Early versions produced explanations like "This item is similar to products from your history." That told the shopper almost nothing, so candidate generation was changed to keep exact provenance through every stage.

Each recommendation can now carry:

the exact history item that retrieved it, and that item's interaction weight
content similarity and its weighted contribution
the co-visitation source item and score
support counts for both signals
the popularity signal
base score, base rank, final score, and final rank
the intent adjustment
any unverifiable constraints from the request

Explanations are generated from this evidence. A real example:

Item 323403 is most similar to item 72028 from your history with a catalog-property similarity of 0.70. It also has a session-based relationship with item 72028 with a co-visitation score of 2.00. Its popularity signal was 114. After applying the user's supported preferences, it moved from rank 5 to rank 4.

A similarity of 0.70 is an internal score over catalog properties. It does not mean the products are "70% similar" in any real-world sense.

Evaluation

Each result below applies to a specific group of users or test cases. Please read the population notes before comparing numbers across tables.

Candidate Retrieval

Population: a prototype sample of validation users. Metric: the share of users whose candidate pool contained at least one item they interacted with in the future period.

Retrieval setup	Overall	Established	Low-history
Basic content	~8%	~9%	~7%
Deeper content	~10%	~12%	~8%
Content + co-visitation	~13%	~15%	~11%

Adding co-visitation gave the largest improvement, which is why it stayed in the final system. Even so, most users' candidate pools contain no future positive item. Retrieval is the main bottleneck in the system.

Reranking

Population: only the 54 validation users whose candidate pool already contained at least one future positive item. These numbers measure how well the reranker orders candidates once a correct answer is available. They are not full-system metrics.

Metric	Baseline	Tree reranker
Precision@10	0.0481	0.0685
Recall@10	0.2350	0.2988
HitRate@10	0.3704	0.4444
MRR	0.1674	0.2116
NDCG	0.1658	0.2025
ROC-AUC		0.7785

The HitRate@10 of 0.4444 means that for 44% of these 54 users, a future positive appeared in the top 10. It does not mean Motive finds a relevant item for 44% of all users.

Intent Parsing

Population: a frozen holdout of 26 natural-language requests, compared against a deterministic rule-based parser.

Parser	Correct	Accuracy
Rule-based baseline	12 / 26	46.15%
Claude (Anthropic API)	25 / 26	96.15%

All outputs passed schema validation. The one miss was "Keep choices pretty conventional," which was expected to map to a popularity preference, but Claude returned no popularity preference. With only 26 cases, this result shows the parser works well on the kinds of requests tested. It is not a general accuracy estimate.

Frontend

The frontend is built with React and Vite and has four sections: Discover, System, Intent, and Trust.

Discover is the main view. A shopper ID and an optional request produce ranked recommendation cards. The page shows the shopper's segment, how the request was interpreted, and any constraints that could not be verified. Each card shows its blended score and a grounded explanation, and an evidence drawer shows the source history item, session support, popularity signal, intent adjustment, and base-to-final rank movement. There are also recent directions, intent shortcuts, and a comparison of the top results.

API
Method	Endpoint	Description
GET	/health	Service health check
GET	/recommend/{user_id}	Recommendations for a shopper without a request
POST	/recommend/intelligent	Recommendations with a natural-language request

Interactive Swagger docs are available at /docs on the local backend when it is running.

The Anthropic API key is read on the server from the ANTHROPIC_API_KEY environment variable. It is never sent to the frontend or included in API responses.

Installation
Prerequisites
Python 3 with venv
Node.js and npm
Git LFS
An Anthropic API key
Setup

Clone the repository and pull the runtime artifacts, which are stored with Git LFS:

git lfs install
git clone <repository-url>
cd AI-Recomender-project
git lfs pull

Create a virtual environment and install the backend dependencies:

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Set your Anthropic API key:

export ANTHROPIC_API_KEY="your-key-here"     # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key-here"

The Anthropic API key is read from the ANTHROPIC_API_KEY environment variable. Local .env files are ignored by Git so secrets are not committed.

Install the frontend dependencies:

cd frontend
npm install
cd ..
Running Locally

Start the backend from the project root:

python -m uvicorn api.main:app --reload

In a second terminal, start the frontend:

cd frontend
npm run dev

Vite prints the local URL to open in your browser. Motive currently runs locally only. There is no public deployment.

The runtime artifacts pulled through Git LFS are enough to run the application. The raw Retailrocket CSV files are not included in Git. You only need them if you want to rebuild the artifacts or retrain the reranker using the scripts in scripts/.

Testing

Motive has two layers of automated checks.

Unit and integration tests:

pytest -q

Current result: 37 passed.

Product regression suite:

python -m scripts.regression_suite

Current result: 158 / 158 checks passed.

The regression suite checks product behavior end to end. It covers all three user segments, ranking with and without intent, familiar and popular requests, exploratory and niche requests, unsupported brand, price, and use-case constraints (alone and combined), result counts, duplicate prevention, provenance, co-visitation evidence, rank movement, limits on how far intent can move an item, grounded explanations, and deterministic output.

Performance Notes

Candidate evaluation was very slow at first. Established shoppers could trigger similarity calculations against the full catalog for each of their history items. The fixes were:

normalizing the sparse item matrix once instead of per query
using dictionary lookups for item indices
using np.argpartition to find top neighbors without fully sorting
caching and pre-warming content neighbors

One evaluation run dropped from over 14 minutes to about 5 seconds after these changes and cache warm-up. This was measured in my local development environment and is not a formal benchmark.

Project Structure
Motive/
├── api/
│   └── main.py                  # FastAPI app and endpoints
├── artifacts/                   # Runtime artifacts (Git LFS)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── services/
│   └── package.json
├── notebooks/                   # Exploration and analysis
├── scripts/                     # Artifact building, training, evaluation, regression suite
├── src/
│   ├── candidates.py            # Candidate generation with provenance
│   ├── content.py               # Content similarity
│   ├── covisitation.py          # Session-based item relationships
│   ├── evidence.py              # Evidence collection
│   ├── explanations.py          # Grounded explanations
│   ├── features.py              # Reranker features
│   ├── intent.py                # UserIntent schema and parsing
│   ├── intent_ranking.py        # Intent blending
│   ├── popularity.py            # Weighted popularity
│   ├── recommender.py           # Main recommendation pipeline
│   ├── reranker.py              # Tree reranker
│   ├── routing.py               # User segmentation
│   ├── artifacts.py             # Artifact loading
│   └── llm/                     # Anthropic client code
├── tests/
├── pytest.ini
├── .gitignore
└── .gitattributes

Runtime artifacts include content neighbors, item IDs and index mappings, the normalized item matrix, weighted popularity, user history counts, user-item strengths, co-visitation neighbors, and the final reranker. The training dataset (reranker_training_data.pkl), the unused logistic reranker, raw CSVs, the virtual environment, node_modules, and the frontend build output are excluded from Git.

Limitations

These are known limits of the current system and its evaluation.

Anonymized products. Product names, brands, prices, specifications, and links are not available, so Motive cannot verify requests that depend on them.
Metadata timing. Content features come from the available metadata snapshot, so the content evaluation is not strictly time-separated. The evaluation should not be read as having research-grade temporal purity.
Conditional reranker metrics. The reranker results apply only to the 54 users whose candidate pool contained a future positive. They do not describe the whole system.
Retrieval is the bottleneck. Most validation users' candidate pools contained no future positive item, so reranking cannot help them.
Hand-set intent weight. The 85/15 blend was chosen manually and has not been learned or calibrated.
Cold-start intent. Requests from new shoppers are parsed but do not change their popularity-based results.
Small intent holdout. The parser evaluation uses 26 cases.
Final benchmark pending. The full system has not yet been re-evaluated on the untouched September period since the latest ranking, provenance, and intent changes.
Scope. Motive does not provide live cross-site recommendations, live prices, semantic product search, or integration with a real retailer.
Future Work
Improve candidate retrieval, since it limits everything downstream.
Try sequential recommendation models that use the order of a shopper's actions.
Learn the intent weighting instead of setting it by hand.
Use richer, verified catalog metadata so constraints like brand and price can actually be checked.
Personalize cold-start results, including letting intent change them.
Run the final integrated system on the untouched September benchmark.
Tech Stack
Area	Technologies
Backend and ML	Python, NumPy, Pandas, SciPy, scikit-learn, FastAPI, Pydantic, Anthropic API
Frontend	React, Vite
Testing and tooling	Pytest, Git LFS
Current Status

Motive runs end to end locally. At the time of writing:

37 / 37 pytest tests pass
158 / 158 regression checks pass
the frontend production build succeeds
live familiar + popular and exploratory + niche requests have been tested through the full stack
the evidence UI has been checked manually

The next major milestone is the final evaluation on the untouched September data.