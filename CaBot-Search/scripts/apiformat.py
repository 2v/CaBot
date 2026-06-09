"""
apiformat.py -- reproduce the CaBot /api/search JSON response byte-for-byte.

These are faithful Python ports of the formatters and the result envelope from the
production Node/Express controller. They are intentionally literal (including a few
JavaScript quirks, e.g. the double period in a citation and "undefined." initials
from empty name parts) so the JSON emitted here matches the API output exactly.

Serialize the result of build_response() with:
    json.dumps(resp, indent=2, ensure_ascii=False)
(ensure_ascii=False matches JS JSON.stringify, which does not escape non-ASCII.)
"""

import json
import re


def format_authors_nejm(authors):
    """Port of formatAuthorsNejm: 'Last F.M.' style, first 6, '..., et al' if >6."""
    if not authors:
        return "Unknown authors"
    try:
        formatted = []
        for name in authors[:6]:
            if name:
                parts = name.strip().split(" ")
                if len(parts) >= 2:
                    last = parts[-1]
                    first_parts = parts[:-1]
                    # JS: p[0]?.toUpperCase() + '.'  -> empty part yields "undefined."
                    initials = "".join(
                        (p[0].upper() if p else "undefined") + "." for p in first_parts)
                    formatted.append(f"{last} {initials}")
                else:
                    formatted.append(name)
        if len(authors) > 6:
            if len(formatted) >= 3:
                return ", ".join(formatted[:3]) + ", et al"
            return ", ".join(formatted) + ", et al"
        return ", ".join(formatted)
    except Exception:
        return "Unknown authors"


def format_nejm_citation(title, authors, journal, year, biblio, doi):
    """Port of formatNejmCitation (authors. title. journal year;vol(iss):pp. doi:...)."""
    try:
        authors_str = format_authors_nejm(authors)
        clean_title = re.sub(r"\.$", "", (title or "").strip())
        citation = f"{authors_str}. {clean_title}. {journal}"
        b = biblio
        if isinstance(b, str):
            try:
                b = json.loads(b)
            except Exception:
                b = None
        if b:
            volume = b.get("volume") or ""
            issue = b.get("issue") or ""
            first_page = b.get("first_page") or ""
            last_page = b.get("last_page") or ""
            if volume:
                citation += f" {year};{volume}"
                if issue:
                    citation += f"({issue})"
                if first_page:
                    if last_page and last_page != first_page:
                        citation += f":{first_page}-{last_page}"
                    else:
                        citation += f":{first_page}"
            else:
                citation += f" {year}"
        else:
            citation += f" {year}"
        citation += "."
        if doi and doi != "null" and doi != "undefined":
            clean_doi = doi.replace("https://doi.org/", "")
            citation += f" doi:{clean_doi}"
        return citation
    except Exception as e:
        return f"Error formatting citation: {e}"


def format_oa_locations(oa_locations):
    """Port of formatOaLocations: numbered '<host> - <version> (<url>)' lines."""
    if not oa_locations:
        return "[No open access locations available]"
    try:
        locs = json.loads(oa_locations) if isinstance(oa_locations, str) else oa_locations
        if not locs or len(locs) == 0:
            return "[No open access locations available]"
        lines = []
        for i, loc in enumerate(locs):
            host_type = loc.get("host_type") or "Unknown"
            url = loc.get("landing_page_url") or "No URL"
            version = loc.get("version") or "Unknown version"
            lines.append(f"      {i + 1}. {host_type} - {version} ({url})")
        return "\n".join(lines)
    except Exception:
        return "[Error parsing open access locations]"


def _as_obj(v):
    """oaLocations/biblio are returned as parsed JSON objects in the API response."""
    if v is None or v == "":
        return None
    return json.loads(v) if isinstance(v, str) else v


def _iso_datetime(pub_date):
    """publicationDate matches node-pg's DATE serialization on a UTC server:
    'YYYY-MM-DD' -> 'YYYY-MM-DDT00:00:00.000Z'."""
    if not pub_date:
        return None
    d = pub_date if isinstance(pub_date, str) else pub_date.isoformat()
    return f"{d}T00:00:00.000Z"


def build_result(row):
    """Build one result object (exact key order of the API) from a row dict with keys:
    id, title, score, year, journal, has_abstract, abstract, oa_locations,
    cited_by_count, authors, doi, biblio, publication_date, is_pubmed_indexed,
    is_open_access, article_type."""
    authors = list(row["authors"]) if row.get("authors") else []
    citation_count = row.get("cited_by_count") or 0
    return {
        "id": row["id"],
        "title": row["title"],
        "score": float(row["score"]),
        "year": row["year"],
        "journal": row["journal"],
        "hasAbstract": row["has_abstract"],
        "abstract": row["abstract"],
        "oaLocations": _as_obj(row.get("oa_locations")),
        "citationCount": citation_count,
        "authors": format_authors_nejm(authors),
        "nejmCitation": format_nejm_citation(
            row["title"], authors, row["journal"], row["year"],
            row.get("biblio"), row.get("doi")),
        "formattedOaLocations": format_oa_locations(row.get("oa_locations")),
        "doi": row.get("doi"),
        "biblio": _as_obj(row.get("biblio")),
        "publicationDate": _iso_datetime(row.get("publication_date")),
        "isPubmedIndexed": row["is_pubmed_indexed"],
        "isOpenAccess": row["is_open_access"],
        "articleType": row["article_type"],
    }


def build_response(query, rows, user_api_key_present=True):
    """Build the full /api/search response envelope."""
    results = [build_result(r) for r in rows]
    return {
        "userApiKeyPresent": user_api_key_present,
        "query": query,
        "totalResults": len(results),
        "results": results,
    }
