from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
    HttpOptions,
)

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")

# If using Vertex AI, set these env vars:
USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() in ("true", "1")

if USE_VERTEXAI:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set when using Vertex AI")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )
else:
    client = genai.Client(api_key=API_KEY)

app = FastAPI()

class ProductRequest(BaseModel):
    productName: str
    country: str

def fetch_from_gemini(product_name: str, country: str, use_search: bool = False):
    prompt = f"""
    Search Google for a real product matching "{product_name}" available for consumers in {country}.
    Based on the search results, generate a single JSON object with the exact structure below.
    Your entire response MUST be only this JSON object and nothing else.

    {{
        "productName": "string",
        "brand": "string",
        "description": "string (a detailed paragraph)",
        "category": "string",
        "price": "number",
        "currency": "string (e.g., USD, EUR, INR, based on the country)",
        "specifications": [{{ "key": "string", "value": "string" }}],
        "barcode": "string (a plausible 12 or 13-digit barcode)",
        "isEdible": "boolean",
        "healthBenefits": ["string"],
        "ingredients": ["string"],
        "manufacturedIn": "string (country name)",
        "availableStores": ["string (names of stores in the specified country)"],
        "visualDescription": "string"
    }}
    """

    if use_search:
        config = GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())],
            temperature=0.5,
        )
    else:
        config = GenerateContentConfig(
            temperature=0.5,
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
    except Exception as e:
        # maybe include which mode failed
        raise HTTPException(status_code=500, detail=f"Gemini API call failed (use_search={use_search}): {str(e)}")

    raw_text = response.text
    if not raw_text:
        raise HTTPException(status_code=500, detail="Empty response from Gemini API")

    # extract JSON
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise HTTPException(status_code=500, detail="No valid JSON found in response")

    product_data = json.loads(raw_text[start:end+1])

    # get grounding metadata if available
    sources = []
    try:
        grounding = response.candidates[0].grounding_metadata.grounding_chunks
        for chunk in grounding:
            if hasattr(chunk, "web") and chunk.web.uri and chunk.web.title:
                sources.append({"uri": chunk.web.uri, "title": chunk.web.title})
    except Exception:
        sources = []

    if "visualDescription" in product_data:
        del product_data["visualDescription"]

    product_data["sources"] = sources

    return product_data

@app.post("/fetch-product-details")
async def fetch_product_details_post(request: ProductRequest, useSearch: bool = Query(False)):
    return fetch_from_gemini(request.productName, request.country, use_search=useSearch)

@app.get("/fetch-product-details")
async def fetch_product_details_get(
    productName: str = Query(...),
    country: str = Query(...),
    useSearch: bool = Query(False),
):
    return fetch_from_gemini(productName, country, use_search=useSearch)
