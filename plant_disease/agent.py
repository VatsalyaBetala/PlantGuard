"""Agentic triage for PlantGuard predictions.

Two layers:

1. ``triage_prediction`` — deterministic local triage. Always runs first.
   Produces facts and safety defaults; never calls the network.

2. ``enhance_triage_with_openai`` — OpenAI tool-using agent loop.
   Given the local triage as ground truth, the LLM may call tools
   (``lookup_disease``, ``get_top3``, ``get_history``, ``request_recapture``,
   ``finalize_triage``) until it terminates with a structured plan.

The loop emits a real ``agent_trace`` of tool calls and observations rather
than a fabricated chain of thought. The final tool arguments are validated
against a pydantic schema; one repair attempt is made before falling back.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


logger = logging.getLogger("plant_disease.agent")


HIGH_RISK_TERMS = (
    "blight",
    "rot",
    "wilt",
    "virus",
    "curl",
    "mosaic",
    "black measles",
)
MEDIUM_RISK_TERMS = (
    "spot",
    "rust",
    "mold",
    "mite",
    "scab",
    "esca",
    "leaf",
)


# ---------------------------------------------------------------------------
# Local deterministic triage (unchanged behavior)
# ---------------------------------------------------------------------------

def _clean_label(value: str | None) -> str:
    if not value:
        return "Unknown"
    return value.replace("_", " ").strip() or "Unknown"


def _is_unknown(value: str | None) -> bool:
    return _clean_label(value).lower() in {"unknown", "no leaf detected"}


def _is_healthy(disease: str | None) -> bool:
    return "healthy" in _clean_label(disease).lower()


def _risk_level(disease: str | None) -> str:
    disease_text = _clean_label(disease).lower()
    if _is_unknown(disease):
        return "unknown"
    if _is_healthy(disease):
        return "low"
    if any(term in disease_text for term in HIGH_RISK_TERMS):
        return "high"
    if any(term in disease_text for term in MEDIUM_RISK_TERMS):
        return "medium"
    return "medium"


def _confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "not_reported"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.50:
        return "medium"
    return "low"


def triage_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic triage plan for one persisted prediction."""
    plant = _clean_label(prediction.get("plant"))
    disease = _clean_label(prediction.get("disease"))
    plant_conf = prediction.get("plant_conf")
    disease_conf = prediction.get("disease_conf")
    has_heatmap = bool(prediction.get("heatmap"))

    blockers: list[str] = []
    reliability_notes: list[str] = []

    if _is_unknown(plant):
        blockers.append("Plant could not be identified from this image.")
    if _is_unknown(disease):
        blockers.append("Disease could not be identified from this image.")
    if not has_heatmap:
        reliability_notes.append("No Grad-CAM heatmap is available for this prediction.")

    plant_band = _confidence_band(plant_conf)
    disease_band = _confidence_band(disease_conf)
    if plant_band == "low":
        reliability_notes.append("Plant classifier confidence is below the reliable threshold.")
    elif plant_band == "not_reported":
        reliability_notes.append("Plant classifier confidence was not stored for this run.")

    if disease_band == "low":
        reliability_notes.append("Disease classifier confidence is below the reliable threshold.")
    elif disease_band == "not_reported":
        reliability_notes.append("Disease classifier confidence was not stored for this run.")

    risk_level = _risk_level(disease)
    if blockers:
        diagnosis_quality = "needs_more_evidence"
        recommended_action = "recapture"
    elif disease_band == "low" or plant_band == "low":
        diagnosis_quality = "low_confidence"
        recommended_action = "verify"
    elif not has_heatmap:
        diagnosis_quality = "usable_with_caution"
        recommended_action = "verify"
    else:
        diagnosis_quality = "usable"
        recommended_action = "act" if risk_level in {"high", "medium"} else "monitor"

    next_questions = _next_questions(plant, disease, risk_level, diagnosis_quality)
    capture_guidance = _capture_guidance(diagnosis_quality, has_heatmap)
    immediate_steps = _immediate_steps(disease, risk_level, diagnosis_quality)

    top3 = _parse_top3(prediction.get("top3_json"))

    return {
        "agent": "local_triage_agent",
        "uses_openai": False,
        "filename": prediction.get("filename"),
        "plant": plant,
        "disease": disease,
        "risk_level": risk_level,
        "diagnosis_quality": diagnosis_quality,
        "recommended_action": recommended_action,
        "confidence": {
            "plant": plant_conf,
            "disease": disease_conf,
            "plant_band": plant_band,
            "disease_band": disease_band,
        },
        "signals": {
            "has_heatmap": has_heatmap,
            "source": prediction.get("source"),
            "backend": prediction.get("backend"),
            "top3": top3,
        },
        "blockers": blockers,
        "reliability_notes": reliability_notes,
        "next_questions": next_questions,
        "capture_guidance": capture_guidance,
        "immediate_steps": immediate_steps,
        "follow_up_days": _follow_up_days(risk_level, diagnosis_quality),
        "summary": _summary(plant, disease, risk_level, diagnosis_quality),
    }


# ---------------------------------------------------------------------------
# Disease knowledge base — consulted by the LLM via the lookup_disease tool
# ---------------------------------------------------------------------------

DISEASE_KB: dict[str, dict[str, Any]] = {
    "early blight": {
        "pathogen": "Alternaria solani (fungal)",
        "hosts": ["Tomato", "Potato"],
        "lookalikes": ["late blight", "septoria leaf spot"],
        "key_signs": [
            "concentric dark rings on older lower leaves",
            "yellow halo around lesions",
        ],
        "spreads_via": ["splashing water", "warm humid weather"],
        "severity": "medium-high",
    },
    "late blight": {
        "pathogen": "Phytophthora infestans (oomycete)",
        "hosts": ["Tomato", "Potato"],
        "lookalikes": ["early blight", "septoria"],
        "key_signs": [
            "water-soaked grey-green lesions on leaf edges",
            "white fuzzy growth on leaf underside in humid conditions",
        ],
        "spreads_via": ["wind-borne spores", "cool wet weather"],
        "severity": "high",
    },
    "powdery mildew": {
        "pathogen": "Various Erysiphales fungi",
        "hosts": ["Squash", "Cucumber", "Grape", "Pepper"],
        "lookalikes": ["downy mildew (underside, not topside)"],
        "key_signs": ["white powdery patches on upper leaf surface"],
        "spreads_via": ["wind", "warm dry days with humid nights"],
        "severity": "medium",
    },
    "leaf mold": {
        "pathogen": "Passalora fulva",
        "hosts": ["Tomato"],
        "lookalikes": ["early blight", "septoria"],
        "key_signs": [
            "yellow patches on upper leaf",
            "olive-green velvety mold on underside",
        ],
        "spreads_via": ["high humidity in greenhouses"],
        "severity": "medium",
    },
    "septoria leaf spot": {
        "pathogen": "Septoria lycopersici",
        "hosts": ["Tomato"],
        "lookalikes": ["early blight", "bacterial spot"],
        "key_signs": [
            "many small circular spots with dark borders and grey centers",
        ],
        "spreads_via": ["splashing water", "infected plant debris"],
        "severity": "medium",
    },
    "spider mites": {
        "pathogen": "Tetranychus urticae (mite)",
        "hosts": ["Tomato", "Bean", "Strawberry"],
        "lookalikes": ["nutrient deficiency stippling"],
        "key_signs": [
            "fine stippling on upper leaf",
            "webbing on underside",
        ],
        "spreads_via": ["dry dusty conditions", "wind"],
        "severity": "medium",
    },
    "black rot": {
        "pathogen": "Guignardia bidwellii",
        "hosts": ["Grape"],
        "lookalikes": ["esca", "anthracnose"],
        "key_signs": [
            "circular tan lesions with dark border on leaves",
            "shrivelled black mummified berries",
        ],
        "spreads_via": ["overwintering mummies", "warm wet spring"],
        "severity": "high",
    },
    "esca": {
        "pathogen": "Wood-rotting fungi complex",
        "hosts": ["Grape"],
        "lookalikes": ["black measles"],
        "key_signs": [
            "tiger-stripe leaf pattern",
            "trunk wood necrosis",
        ],
        "spreads_via": ["pruning wounds"],
        "severity": "high",
    },
    "mosaic virus": {
        "pathogen": "Various mosaic viruses (e.g. ToMV, TMV, CMV)",
        "hosts": ["Tomato", "Pepper", "Squash"],
        "lookalikes": ["nutrient deficiency", "herbicide damage"],
        "key_signs": [
            "mottled light/dark green pattern",
            "distorted new growth",
        ],
        "spreads_via": ["aphids", "contaminated tools/hands"],
        "severity": "high",
    },
    "yellow leaf curl virus": {
        "pathogen": "Tomato yellow leaf curl virus (TYLCV)",
        "hosts": ["Tomato"],
        "lookalikes": ["nutrient deficiency"],
        "key_signs": [
            "upward leaf curling",
            "yellow leaf margins",
            "stunted growth",
        ],
        "spreads_via": ["whiteflies (Bemisia tabaci)"],
        "severity": "high",
    },
    "bacterial spot": {
        "pathogen": "Xanthomonas spp.",
        "hosts": ["Tomato", "Pepper"],
        "lookalikes": ["septoria", "early blight"],
        "key_signs": [
            "small water-soaked spots that turn brown/black",
            "shot-hole appearance after lesions dry",
        ],
        "spreads_via": ["splashing water", "contaminated seed"],
        "severity": "medium-high",
    },
    "scab": {
        "pathogen": "Venturia inaequalis (apple) / Cladosporium spp.",
        "hosts": ["Apple", "Peach"],
        "lookalikes": ["sooty blotch"],
        "key_signs": [
            "olive-green velvety spots on leaves",
            "rough corky lesions on fruit",
        ],
        "spreads_via": ["spring rain", "infected leaf litter"],
        "severity": "medium",
    },
}


def _kb_match(query: str) -> tuple[str | None, dict[str, Any] | None]:
    """Best-effort fuzzy lookup: substring on cleaned label."""
    if not query:
        return None, None
    needle = query.replace("_", " ").strip().lower()
    if not needle:
        return None, None
    if needle in DISEASE_KB:
        return needle, DISEASE_KB[needle]
    for key, entry in DISEASE_KB.items():
        if key in needle or needle in key:
            return key, entry
    return None, None


# ---------------------------------------------------------------------------
# Tool implementations (sync, fast, no network)
# ---------------------------------------------------------------------------

def _tool_lookup_disease(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("name", ""))
    key, entry = _kb_match(name)
    if entry is None:
        return {
            "found": False,
            "query": name,
            "available_keys": sorted(DISEASE_KB.keys()),
        }
    return {"found": True, "query": name, "matched_key": key, **entry}


def _tool_get_top3(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    triage = ctx["local_triage"]
    top3 = triage.get("signals", {}).get("top3", {})
    plant_top = top3.get("plant", []) or []
    disease_top = top3.get("disease", []) or []
    spread = None
    if len(disease_top) >= 2:
        try:
            spread = float(disease_top[0].get("confidence", 0)) - float(
                disease_top[1].get("confidence", 0)
            )
        except (TypeError, ValueError, AttributeError):
            spread = None
    return {
        "plant_top3": plant_top,
        "disease_top3": disease_top,
        "disease_top1_top2_spread": spread,
        "is_ambiguous": (spread is not None and spread < 0.15),
    }


def _tool_get_history(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    plant = str(args.get("plant", ctx["local_triage"].get("plant", ""))).lower()
    limit = int(args.get("limit", 5))
    limit = max(1, min(limit, 20))

    history_fn = ctx.get("history_fn")
    if history_fn is None:
        return {"available": False, "reason": "history_fn not provided"}

    try:
        rows = history_fn() or []
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("history_fn failed: %s", exc)
        return {"available": False, "reason": f"history error: {exc}"}

    current_filename = ctx["local_triage"].get("filename")
    filtered = []
    for row in rows:
        if row.get("filename") == current_filename:
            continue
        if plant and plant not in str(row.get("plant", "")).lower():
            continue
        filtered.append(
            {
                "filename": row.get("filename"),
                "plant": row.get("plant"),
                "disease": row.get("disease"),
                "disease_conf": row.get("disease_conf"),
                "uploadDate": row.get("uploadDate"),
            }
        )
        if len(filtered) >= limit:
            break

    return {
        "available": True,
        "plant_filter": plant or None,
        "count": len(filtered),
        "items": filtered,
    }


def _tool_request_recapture(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Terminal tool — agent gives up and asks user for a better photo."""
    ctx["terminal"] = {
        "kind": "recapture",
        "reason": str(args.get("reason", "")),
        "capture_hints": _as_list(args.get("capture_hints")),
        "next_questions": _as_list(args.get("next_questions")),
    }
    return {"acknowledged": True}


def _tool_get_local_outbreaks(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Return active outbreaks within ``radius_km`` of the prediction's location."""
    triage = ctx["local_triage"]
    pred = ctx.get("prediction", {}) or {}
    lat = pred.get("lat")
    lon = pred.get("lon")
    if lat is None or lon is None:
        return {"available": False, "reason": "Prediction has no location."}

    try:
        from plant_disease.db.repo import list_outbreaks_near
    except Exception as exc:
        return {"available": False, "reason": f"db unavailable: {exc}"}

    radius_km = float(args.get("radius_km", 10.0) or 10.0)
    days = int(args.get("days", 14) or 14)
    radius_km = max(1.0, min(radius_km, 100.0))
    days = max(1, min(days, 90))
    rows = list_outbreaks_near(lat, lon, radius_km=radius_km, days=days)

    matches_disease = [
        r for r in rows
        if r["disease"].replace("_", " ").lower() in (triage.get("disease") or "").replace("_", " ").lower()
        or (triage.get("disease") or "").replace("_", " ").lower() in r["disease"].replace("_", " ").lower()
    ]
    return {
        "available": True,
        "radius_km": radius_km,
        "days": days,
        "count": len(rows),
        "outbreaks": rows,
        "same_disease_nearby_count": len(matches_disease),
    }


def _tool_get_pressure_forecast(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Return per-day disease pressure forecast for the prediction's grid cell."""
    pred = ctx.get("prediction", {}) or {}
    lat = pred.get("lat")
    lon = pred.get("lon")
    if lat is None or lon is None:
        return {"available": False, "reason": "Prediction has no location."}

    try:
        from plant_disease.db.repo import list_pressure_for_grid
    except Exception as exc:
        return {"available": False, "reason": f"db unavailable: {exc}"}

    disease = args.get("disease") or ctx["local_triage"].get("disease")
    rows = list_pressure_for_grid(lat, lon, disease=disease)
    if not rows:
        return {
            "available": True,
            "disease": disease,
            "days": [],
            "note": "No cached pressure scores yet for this cell.",
        }
    peak = max(rows, key=lambda r: r.get("score", 0))
    return {
        "available": True,
        "disease": disease,
        "days": rows,
        "peak_day": peak["day"],
        "peak_score": peak["score"],
        "peak_model": peak["model"],
    }


def _tool_finalize_triage(args: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Terminal tool — agent commits to the final structured triage."""
    ctx["terminal"] = {"kind": "finalize", "payload": args}
    return {"acknowledged": True}


TOOL_IMPLEMENTATIONS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "lookup_disease": _tool_lookup_disease,
    "get_top3": _tool_get_top3,
    "get_history": _tool_get_history,
    "get_local_outbreaks": _tool_get_local_outbreaks,
    "get_pressure_forecast": _tool_get_pressure_forecast,
    "request_recapture": _tool_request_recapture,
    "finalize_triage": _tool_finalize_triage,
}


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_disease",
            "description": (
                "Look up a curated disease entry (pathogen, hosts, key signs, lookalikes, "
                "spread vectors, severity). Use this before recommending action so advice "
                "is grounded in the knowledge base, not invented."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Disease name as predicted by the classifier or a candidate diagnosis.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top3",
            "description": (
                "Return the top-3 candidate plants and diseases with their confidences, "
                "and whether top-1 and top-2 are within 0.15 (ambiguous). Use to decide "
                "whether to surface competing diagnoses."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Return recent prior predictions for the same plant. Use to spot trends "
                "(e.g. repeated tomato early blight uploads suggest a spreading outbreak)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant": {
                        "type": "string",
                        "description": "Plant name to filter on. Defaults to the current prediction's plant.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of prior items (1-20). Default 5.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_outbreaks",
            "description": (
                "Return outbreak clusters within a given radius and time window of the "
                "current prediction's location. Use to assess whether a regional "
                "epidemic is active and should escalate the recommended_action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "radius_km": {
                        "type": "number",
                        "description": "Search radius in kilometers (1-100). Default 10.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Look back this many days for outbreak end (1-90). Default 14.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pressure_forecast",
            "description": (
                "Return per-day disease pressure scores (Beaumont/Wallin/Gubler/generic) "
                "for the prediction's grid cell. Helps justify 'spray now' vs 'monitor' "
                "based on forecast weather, not just current symptoms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {
                        "type": "string",
                        "description": "Disease name. Defaults to current prediction's disease.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_recapture",
            "description": (
                "Terminal action when evidence is too weak to give a safe diagnosis. "
                "The agent stops and asks the user for a better photo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the current evidence is insufficient.",
                    },
                    "capture_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete photo instructions for the user.",
                    },
                    "next_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Clarifying questions to ask alongside recapture.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_triage",
            "description": (
                "Terminal action: commit to the final structured triage plan. Call this "
                "exactly once, after gathering enough evidence via the other tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                    "diagnosis_quality": {
                        "type": "string",
                        "enum": [
                            "usable",
                            "usable_with_caution",
                            "low_confidence",
                            "needs_more_evidence",
                        ],
                    },
                    "recommended_action": {
                        "type": "string",
                        "enum": ["act", "monitor", "verify", "recapture"],
                    },
                    "competing_diagnoses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "why_it_matters": {"type": "string"},
                                "what_would_confirm": {"type": "string"},
                            },
                            "required": ["label"],
                        },
                    },
                    "escalation_flags": {"type": "array", "items": {"type": "string"}},
                    "field_mission": {
                        "type": "object",
                        "properties": {
                            "next_10_minutes": {"type": "array", "items": {"type": "string"}},
                            "next_24_hours": {"type": "array", "items": {"type": "string"}},
                            "next_follow_up": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "next_questions": {"type": "array", "items": {"type": "string"}},
                    "capture_guidance": {"type": "array", "items": {"type": "string"}},
                    "immediate_steps": {"type": "array", "items": {"type": "string"}},
                    "follow_up_days": {"type": "integer"},
                },
                "required": [
                    "summary",
                    "risk_level",
                    "diagnosis_quality",
                    "recommended_action",
                ],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Pydantic schema for finalize_triage validation
# ---------------------------------------------------------------------------

class CompetingDiagnosis(BaseModel):
    label: str
    why_it_matters: str = ""
    what_would_confirm: str = ""


class FieldMission(BaseModel):
    next_10_minutes: list[str] = Field(default_factory=list)
    next_24_hours: list[str] = Field(default_factory=list)
    next_follow_up: list[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    summary: str
    risk_level: str
    diagnosis_quality: str
    recommended_action: str
    competing_diagnoses: list[CompetingDiagnosis] = Field(default_factory=list)
    escalation_flags: list[str] = Field(default_factory=list)
    field_mission: FieldMission = Field(default_factory=FieldMission)
    next_questions: list[str] = Field(default_factory=list)
    capture_guidance: list[str] = Field(default_factory=list)
    immediate_steps: list[str] = Field(default_factory=list)
    follow_up_days: int = 7


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are PlantGuard's agronomy triage coordinator. A vision pipeline has "
    "already produced a plant + disease prediction. Your job is to AUDIT that "
    "result, gather supporting evidence via tools, and produce a farmer-safe "
    "next-action plan.\n\n"
    "Rules:\n"
    "- The local_triage facts are ground truth for confidence and labels. "
    "Do not invent confidences.\n"
    "- Before recommending treatment, call lookup_disease for the predicted "
    "disease (and any competing candidate from get_top3) so advice is grounded.\n"
    "- If the prediction has lat/lon, ALSO call get_local_outbreaks AND "
    "get_pressure_forecast. A nearby active outbreak of the same disease, or "
    "a high pressure score within 3 days, should escalate recommended_action "
    "to 'act' (or at minimum 'verify') and add an explicit escalation_flag.\n"
    "- If top-1 and top-2 disease confidences are close, surface both as "
    "competing_diagnoses with what_would_confirm grounded in the KB key_signs.\n"
    "- If evidence is too weak (Unknown labels, very low confidence, no heatmap "
    "AND ambiguous top3), call request_recapture instead of guessing.\n"
    "- Never recommend exact pesticide dosing.\n"
    "- Terminate with exactly one of: finalize_triage OR request_recapture.\n"
    "- Use at most 8 tool calls. Be decisive."
)


def run_agent_loop(
    prediction: dict[str, Any],
    local_triage: dict[str, Any],
    client: Any,
    *,
    model: str = "gpt-4o",
    history_fn: Callable[[], list[dict[str, Any]]] | None = None,
    max_iterations: int = 8,
) -> dict[str, Any]:
    """Run the tool-using agent loop. Returns a result dict that includes a
    real ``agent_trace`` of tool calls/observations and either a finalized
    triage payload or a recapture request.

    The function is exception-tolerant at the *tool* level (a tool error is
    fed back to the model as an observation), but propagates fatal errors
    (network, auth) to the caller.
    """
    ctx: dict[str, Any] = {
        "local_triage": local_triage,
        "history_fn": history_fn,
        "prediction": prediction,
        "terminal": None,
    }

    user_payload = {
        "prediction": {
            "filename": prediction.get("filename"),
            "plant": prediction.get("plant"),
            "disease": prediction.get("disease"),
            "plant_conf": prediction.get("plant_conf"),
            "disease_conf": prediction.get("disease_conf"),
            "has_heatmap": bool(prediction.get("heatmap")),
            "backend": prediction.get("backend"),
            "source": prediction.get("source"),
            "lat": prediction.get("lat"),
            "lon": prediction.get("lon"),
            "capture_ts": prediction.get("capture_ts"),
            "outbreak_id": prediction.get("outbreak_id"),
        },
        "local_triage": local_triage,
    }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Audit this prediction and produce a triage plan. "
                "Use tools to ground your advice. Facts:\n"
                + json.dumps(user_payload, indent=2)
            ),
        },
    ]

    trace: list[dict[str, Any]] = []
    finalize_args: dict[str, Any] | None = None
    repair_used = False

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        # Persist the assistant turn so subsequent tool replies link correctly.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        if not tool_calls:
            # Model declined to call a tool — nudge it once toward a terminal.
            trace.append(
                {
                    "iteration": iteration,
                    "event": "no_tool_call",
                    "assistant_text": (message.content or "")[:500],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You must call either finalize_triage or request_recapture "
                        "to terminate. Tool calls only — no plain text answers."
                    ),
                }
            )
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = TOOL_IMPLEMENTATIONS.get(name)
            if impl is None:
                observation: dict[str, Any] = {
                    "error": f"unknown tool: {name}",
                    "available": sorted(TOOL_IMPLEMENTATIONS.keys()),
                }
            else:
                # finalize_triage gets validated before we accept termination.
                if name == "finalize_triage":
                    try:
                        TriageResponse.model_validate(args)
                    except ValidationError as exc:
                        if not repair_used:
                            repair_used = True
                            observation = {
                                "accepted": False,
                                "validation_errors": exc.errors(),
                                "instructions": (
                                    "The finalize_triage arguments failed schema "
                                    "validation. Fix the listed fields and call "
                                    "finalize_triage again. This is your last repair."
                                ),
                            }
                            trace.append(
                                {
                                    "iteration": iteration,
                                    "event": "validation_failed",
                                    "tool": name,
                                    "errors": [e.get("msg") for e in exc.errors()],
                                }
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "name": name,
                                    "content": json.dumps(observation),
                                }
                            )
                            continue
                        # Repair already used — accept best-effort and bail.
                        finalize_args = args
                        observation = {"accepted": True, "warning": "schema imperfect; accepted after repair"}
                        ctx["terminal"] = {"kind": "finalize", "payload": args}
                    else:
                        finalize_args = args
                        observation = impl(args, ctx)
                else:
                    try:
                        observation = impl(args, ctx)
                    except Exception as exc:  # pragma: no cover - defensive
                        observation = {"error": f"tool {name} raised: {exc}"}

            trace.append(
                {
                    "iteration": iteration,
                    "event": "tool_call",
                    "tool": name,
                    "args": args,
                    "observation_summary": _summarize_obs(name, observation),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(observation, default=str),
                }
            )

        if ctx["terminal"] is not None:
            break

    return {
        "trace": trace,
        "terminal": ctx["terminal"],
        "finalize_args": finalize_args,
        "iterations_used": iteration,
        "max_iterations": max_iterations,
    }


def _summarize_obs(tool: str, obs: dict[str, Any]) -> str:
    if tool == "lookup_disease":
        if obs.get("found"):
            return f"KB hit: {obs.get('matched_key')} (severity={obs.get('severity')})"
        return f"KB miss for '{obs.get('query')}'"
    if tool == "get_top3":
        return f"top3 ambiguous={obs.get('is_ambiguous')} spread={obs.get('disease_top1_top2_spread')}"
    if tool == "get_history":
        if not obs.get("available"):
            return "history unavailable"
        return f"history: {obs.get('count')} prior items for '{obs.get('plant_filter')}'"
    if tool == "get_local_outbreaks":
        if not obs.get("available"):
            return f"outbreaks unavailable ({obs.get('reason')})"
        return (
            f"outbreaks within {obs.get('radius_km')}km / {obs.get('days')}d: "
            f"{obs.get('count')} ({obs.get('same_disease_nearby_count')} same disease)"
        )
    if tool == "get_pressure_forecast":
        if not obs.get("available"):
            return f"pressure unavailable ({obs.get('reason')})"
        if not obs.get("days"):
            return "pressure: no scores cached"
        return f"pressure peak {obs.get('peak_score')} on {obs.get('peak_day')} ({obs.get('peak_model')})"
    if tool == "request_recapture":
        return "terminal: recapture requested"
    if tool == "finalize_triage":
        return "terminal: triage finalized"
    return ""


# ---------------------------------------------------------------------------
# Public entry point — keeps the existing function signature
# ---------------------------------------------------------------------------

def enhance_triage_with_openai(
    prediction: dict[str, Any],
    local_triage: dict[str, Any],
    client: Any,
    *,
    model: str = "gpt-4o",
    history_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run the tool-using agent loop and merge its result with local triage.

    Behavior:
    - On finalize_triage: returned dict has the LLM's structured plan plus
      ``agent_trace`` (real tool log) and ``competing_diagnoses``,
      ``escalation_flags``, ``field_mission`` from the model.
    - On request_recapture: returned dict forces ``recommended_action='recapture'``
      and ``diagnosis_quality='needs_more_evidence'`` with the model's reason
      surfaced in ``summary`` and its hints in ``capture_guidance``.
    - On loop exhaustion without a terminal: falls back to local_triage with a
      diagnostic note.
    """
    loop_result = run_agent_loop(
        prediction,
        local_triage,
        client,
        model=model,
        history_fn=history_fn,
    )

    terminal = loop_result.get("terminal")
    trace = loop_result.get("trace", [])

    if terminal is None:
        merged = dict(local_triage)
        merged["agent"] = "openai_triage_agent"
        merged["uses_openai"] = True
        merged["agent_trace"] = _trace_for_ui(trace)
        merged["competing_diagnoses"] = []
        merged["escalation_flags"] = ["Agent loop exhausted without a terminal decision."]
        merged["field_mission"] = {}
        return merged

    if terminal["kind"] == "recapture":
        merged = dict(local_triage)
        merged["agent"] = "openai_triage_agent"
        merged["uses_openai"] = True
        reason = terminal.get("reason") or "Evidence is insufficient for a safe diagnosis."
        merged["summary"] = reason
        merged["risk_level"] = local_triage.get("risk_level", "unknown")
        merged["diagnosis_quality"] = "needs_more_evidence"
        merged["recommended_action"] = "recapture"
        hints = terminal.get("capture_hints") or []
        if hints:
            merged["capture_guidance"] = list(hints)
        next_qs = terminal.get("next_questions") or []
        if next_qs:
            merged["next_questions"] = list(next_qs)
        merged["follow_up_days"] = 1
        merged["agent_trace"] = _trace_for_ui(trace)
        merged["competing_diagnoses"] = []
        merged["escalation_flags"] = []
        merged["field_mission"] = {}
        return merged

    # Finalize path
    payload = terminal.get("payload", {}) or {}
    merged = dict(local_triage)
    merged["agent"] = "openai_triage_agent"
    merged["uses_openai"] = True

    for key in (
        "summary",
        "risk_level",
        "diagnosis_quality",
        "recommended_action",
        "next_questions",
        "capture_guidance",
        "immediate_steps",
        "follow_up_days",
    ):
        if key in payload:
            merged[key] = payload[key]

    merged["agent_trace"] = _trace_for_ui(trace)
    merged["competing_diagnoses"] = _as_list(payload.get("competing_diagnoses"))
    merged["escalation_flags"] = _as_list(payload.get("escalation_flags"))
    field_mission = payload.get("field_mission")
    merged["field_mission"] = field_mission if isinstance(field_mission, dict) else {}
    return merged


def _trace_for_ui(trace: list[dict[str, Any]]) -> list[str]:
    """Render the structured trace as human-readable bullet lines for the UI."""
    lines: list[str] = []
    for entry in trace:
        if entry.get("event") == "tool_call":
            tool = entry.get("tool")
            summary = entry.get("observation_summary") or ""
            args = entry.get("args") or {}
            arg_bits = []
            for k, v in args.items():
                if isinstance(v, (str, int, float, bool)):
                    arg_bits.append(f"{k}={v}")
            arg_str = ", ".join(arg_bits)
            if arg_str:
                lines.append(f"called {tool}({arg_str}) ->{summary}")
            else:
                lines.append(f"called {tool}() ->{summary}")
        elif entry.get("event") == "validation_failed":
            errs = ", ".join(entry.get("errors") or [])
            lines.append(f"finalize_triage rejected: {errs}; retrying once")
        elif entry.get("event") == "no_tool_call":
            lines.append("model emitted no tool call; nudged toward terminal")
    return lines


# ---------------------------------------------------------------------------
# Deterministic helpers (unchanged behavior, used by triage_prediction)
# ---------------------------------------------------------------------------

def _parse_top3(top3_json: str | None) -> dict[str, Any]:
    if not top3_json:
        return {"plant": [], "disease": []}
    try:
        parsed = json.loads(top3_json)
    except (TypeError, json.JSONDecodeError):
        return {"plant": [], "disease": []}
    if not isinstance(parsed, dict):
        return {"plant": [], "disease": []}
    return {
        "plant": parsed.get("plant", []) if isinstance(parsed.get("plant", []), list) else [],
        "disease": parsed.get("disease", []) if isinstance(parsed.get("disease", []), list) else [],
    }


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _next_questions(
    plant: str,
    disease: str,
    risk_level: str,
    diagnosis_quality: str,
) -> list[str]:
    if diagnosis_quality == "needs_more_evidence":
        return [
            "Can you upload a sharp close-up of a single affected leaf?",
            "Is the affected area on the upper side, underside, or edge of the leaf?",
            "Are nearby leaves showing similar symptoms?",
        ]

    questions = [
        f"How quickly are the symptoms spreading on the {plant.lower()} plant?",
        "Did the plant recently receive heavy rain, overhead watering, or high humidity?",
    ]
    disease_text = disease.lower()
    if risk_level == "high":
        questions.append("Are lower leaves or neighboring plants showing the same pattern?")
    if "mite" in disease_text:
        questions.append("Do you see fine webbing or tiny moving dots on the underside of leaves?")
    if "virus" in disease_text or "curl" in disease_text or "mosaic" in disease_text:
        questions.append("Have you noticed whiteflies, aphids, or distorted new growth?")
    return questions


def _capture_guidance(diagnosis_quality: str, has_heatmap: bool) -> list[str]:
    guidance = [
        "Use natural light and keep one leaf filling most of the frame.",
        "Avoid shadows, blur, wet leaf surfaces, and mixed plants in the same image.",
    ]
    if diagnosis_quality != "usable" or not has_heatmap:
        guidance.extend(
            [
                "Upload one close-up of the affected area and one wider image of the whole leaf.",
                "Include both healthy-looking and affected leaves if symptoms vary across the plant.",
            ]
        )
    return guidance


def _immediate_steps(disease: str, risk_level: str, diagnosis_quality: str) -> list[str]:
    if diagnosis_quality == "needs_more_evidence":
        return [
            "Do not apply treatment based on this result alone.",
            "Recapture the image and rerun analysis before deciding on control measures.",
        ]

    if _is_healthy(disease):
        return [
            "Keep monitoring the plant every few days.",
            "Maintain steady watering, airflow, and leaf cleanliness.",
        ]

    steps = [
        "Isolate or mark the affected plant so symptoms can be checked again.",
        "Remove heavily damaged leaves only if the plant has enough healthy foliage remaining.",
        "Avoid overhead watering until the diagnosis is verified.",
    ]
    if risk_level == "high":
        steps.insert(0, "Inspect nearby plants today for similar symptoms.")
    return steps


def _follow_up_days(risk_level: str, diagnosis_quality: str) -> int:
    if diagnosis_quality == "needs_more_evidence":
        return 1
    if risk_level == "high":
        return 2
    if risk_level == "medium":
        return 4
    return 7


def _summary(plant: str, disease: str, risk_level: str, diagnosis_quality: str) -> str:
    if diagnosis_quality == "needs_more_evidence":
        return "The agent needs a clearer image before trusting this diagnosis."
    if _is_healthy(disease):
        return f"The {plant} sample looks healthy; continue routine monitoring."
    return (
        f"The agent rates this {plant} / {disease} result as {diagnosis_quality} "
        f"with {risk_level} field risk."
    )
