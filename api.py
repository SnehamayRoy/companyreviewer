from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(
    title="Company Sentiment API",
    description="Returns aspect ratings and summaries computed from Reddit analysis",
    version="1.0"
)

# Load the equal-weighted aspect ratings once on startup
ratings_df = pd.read_csv("company_aspect_ratings_equal_weight.csv")

# Pivot to easy lookup: {company -> {aspect -> rating}}
ratings_pivot = ratings_df.pivot(index="company", columns="aspect", values="rating_1_to_5")


def summarize_company(company):
    """Return summary string & overall score."""
    if company not in ratings_pivot.index:
        raise KeyError

    row = ratings_pivot.loc[company].dropna()

    # Overall rating = mean across aspects
    overall = row.mean()

    # Find strongest and weakest aspects
    best_aspect = row.idxmax().replace("_", " ")
    best_score = row.max()

    worst_aspect = row.idxmin().replace("_", " ")
    worst_score = row.min()

    # Tone label
    if overall >= 3.0:
        tone = "generally positive"
    elif overall >= 2.5:
        tone = "mixed / neutral"
    else:
        tone = "generally negative"

    summary = (
        f"{company} shows strongest perception in **{best_aspect}** ({best_score:.2f}/5), "
        f"while **{worst_aspect}** is the weakest area ({worst_score:.2f}/5). "
        f"Overall employee sentiment is {tone}."
    )

    return overall, summary


@app.get("/company/{company_name}")
def get_company(company_name: str):
    company_name = company_name.strip()

    try:
        overall, summary = summarize_company(company_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Company not found")

    aspects = ratings_pivot.loc[company_name].dropna().to_dict()

    return {
        "company": company_name,
        "overall_rating": round(overall, 2),
        "summary": summary,
        "aspects": aspects
    }
