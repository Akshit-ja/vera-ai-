import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
START_TIME = time.time()

SCOPES = {"category", "merchant", "customer", "trigger"}

# In-memory stores
contexts: Dict[str, Dict[str, Dict[str, Any]]] = {
    "category": {},
    "merchant": {},
    "customer": {},
    "trigger": {},
}

conversations: Dict[str, Dict[str, Any]] = {}
handled_triggers: set[str] = set()
suppression_keys_used: set[str] = set()
sent_bodies: set[str] = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_get(dct: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _owner_name(merchant: Dict[str, Any], category: Dict[str, Any]) -> str:
    owner = _safe_get(merchant, ["identity", "owner_first_name"], "")
    if not owner:
        name = _safe_get(merchant, ["identity", "name"], "there")
        return name
    salutation_examples = _safe_get(category, ["voice", "salutation_examples"], [])
    if salutation_examples:
        return salutation_examples[0].replace("{first_name}", owner)
    return owner


def _hi_en_mix(merchant: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> bool:
    cust_lang = _safe_get(customer or {}, ["identity", "language_pref"], "")
    merch_langs = _safe_get(merchant, ["identity", "languages"], [])
    return ("hi" in cust_lang) or ("hi" in merch_langs) or ("hi-en" in cust_lang)


def _pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{int(round(value * 100))}%"


def _format_currency(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"{int(value)}"
    except (ValueError, TypeError):
        return None


def _pick_offer(merchant: Dict[str, Any], category: Dict[str, Any]) -> Optional[str]:
    offers = merchant.get("offers", [])
    for offer in offers:
        if offer.get("status") == "active" and offer.get("title"):
            return offer["title"]
    for offer in category.get("offer_catalog", []):
        if offer.get("title"):
            return offer["title"]
    return None


def _pick_digest_item(category: Dict[str, Any], item_id: str) -> Optional[Dict[str, Any]]:
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def _render_slots(slots: List[Dict[str, Any]]) -> List[str]:
    labels = []
    for slot in slots:
        label = slot.get("label")
        if label:
            labels.append(label)
    return labels


def _detect_auto_reply(text: str) -> bool:
    t = text.lower()
    patterns = [
        "thank you for contacting",
        "our team will respond",
        "we will respond shortly",
        "auto-reply",
        "automated message",
        "outside business hours",
    ]
    return any(p in t for p in patterns)


def _detect_opt_out(text: str) -> bool:
    t = text.lower()
    patterns = ["stop", "unsubscribe", "do not message", "dont message", "not interested"]
    return any(p in t for p in patterns)


def _detect_off_topic(text: str) -> bool:
    t = text.lower()
    patterns = ["gst", "tax", "loan", "bank", "rent agreement"]
    return any(p in t for p in patterns)


def _detect_positive(text: str) -> bool:
    t = text.lower()
    patterns = ["yes", "ok", "okay", "please", "go ahead", "send", "sure"]
    return any(p in t for p in patterns)


def _detect_negative(text: str) -> bool:
    t = text.lower()
    patterns = ["no", "not now", "later", "busy"]
    return any(p in t for p in patterns)


def _cta_sentence(prompt: str, cta: str) -> str:
    return prompt.strip()


def compose(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any],
            customer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kind = trigger.get("kind", "unknown")
    scope = trigger.get("scope", "merchant")
    send_as = "vera" if scope == "merchant" else "merchant_on_behalf"
    suppression_key = trigger.get("suppression_key") or f"trigger:{trigger.get('id')}"

    salutation = _owner_name(merchant, category) if scope == "merchant" else _safe_get(customer or {}, ["identity", "name"], "there")
    intro = f"Hi {salutation}," if scope == "customer" else f"{salutation},"

    offer_title = _pick_offer(merchant, category)
    hi_en = _hi_en_mix(merchant, customer)

    body = ""
    cta = "open_ended"
    rationale = ""
    template_params: List[str] = []

    if kind == "research_digest":
        item_id = _safe_get(trigger, ["payload", "top_item_id"], "")
        item = _pick_digest_item(category, item_id) if item_id else None
        title = _safe_get(item or {}, ["title"], "New research update")
        source = _safe_get(item or {}, ["source"], "")
        trial_n = _safe_get(item or {}, ["trial_n"], None)
        patient_segment = _safe_get(item or {}, ["patient_segment"], "")
        trial_str = f"{trial_n}-patient" if trial_n else ""
        segment_str = f" for {patient_segment.replace('_', ' ')}" if patient_segment else ""
        body = f"{intro} {title}. {trial_str}{segment_str}. Want me to pull the abstract and draft a patient-facing WhatsApp?"
        if source:
            body = f"{body} — {source}"
        cta = "open_ended"
        rationale = "Research digest trigger; highlight one item with source and offer to draft a patient message."
        template_params = [salutation, title, source or ""]

    elif kind == "regulation_change":
        item_id = _safe_get(trigger, ["payload", "top_item_id"], "")
        item = _pick_digest_item(category, item_id) if item_id else None
        title = _safe_get(item or {}, ["title"], "Regulatory update")
        deadline = _safe_get(trigger, ["payload", "deadline_iso"], "")
        deadline_str = f"effective {deadline.split('T')[0]}" if deadline else "effective soon"
        body = f"{intro} Heads-up: {title} ({deadline_str}). Want a 3-point compliance checklist?"
        source = _safe_get(item or {}, ["source"], "")
        if source:
            body = f"{body} — {source}"
        cta = "binary_yes_no"
        rationale = "Compliance trigger; highlight deadline and offer a checklist."
        template_params = [salutation, title, deadline_str]

    elif kind == "recall_due":
        slots = _safe_get(trigger, ["payload", "available_slots"], [])
        slot_labels = _render_slots(slots)
        due_date = _safe_get(trigger, ["payload", "due_date"], "")
        last_service = _safe_get(trigger, ["payload", "last_service_date"], "")
        offer = offer_title or "cleaning"
        prefix = "Aapke liye" if hi_en else "For you"
        if slot_labels:
            body = (
                f"{intro} {prefix} recall due since {last_service} (due {due_date}). "
                f"Slots ready: {slot_labels[0]} or {slot_labels[1] if len(slot_labels) > 1 else slot_labels[0]}. "
                f"{offer}. Reply 1 for first, 2 for second, or share a preferred time."
            )
            cta = "multi_choice_slot"
        else:
            body = f"{intro} {prefix} recall due since {last_service} (due {due_date}). Want me to hold a slot this week?"
            cta = "binary_yes_no"
        rationale = "Customer recall trigger; use due date and provide specific slot choices."
        template_params = [salutation, due_date, offer]

    elif kind == "perf_dip":
        metric = _safe_get(trigger, ["payload", "metric"], "metric")
        delta_pct = _safe_get(trigger, ["payload", "delta_pct"], 0)
        window = _safe_get(trigger, ["payload", "window"], "7d")
        vs_baseline = _safe_get(trigger, ["payload", "vs_baseline"], None)
        baseline_str = f" (baseline {vs_baseline})" if vs_baseline is not None else ""
        action = "draft 2 Google posts" if "stale_posts" in " ".join(merchant.get("signals", [])) else "refresh your top offer"
        offer = offer_title or "your top service"
        body = (
            f"{intro} {metric} down {_pct(delta_pct)} over {window}{baseline_str}. "
            f"I can {action} around {offer}. Want me to do that?"
        )
        cta = "binary_yes_no"
        rationale = "Performance dip trigger; connect metric drop to a specific corrective action."
        template_params = [salutation, metric, _pct(delta_pct)]

    elif kind == "perf_spike":
        metric = _safe_get(trigger, ["payload", "metric"], "metric")
        delta_pct = _safe_get(trigger, ["payload", "delta_pct"], 0)
        window = _safe_get(trigger, ["payload", "window"], "7d")
        driver = _safe_get(trigger, ["payload", "likely_driver"], "")
        driver_str = f" Likely driver: {driver}." if driver else ""
        body = f"{intro} {metric} up {_pct(delta_pct)} over {window}.{driver_str} Want me to double down with a fresh post today?"
        cta = "binary_yes_no"
        rationale = "Performance spike trigger; reinforce the likely driver and suggest doubling down."
        template_params = [salutation, metric, _pct(delta_pct)]

    elif kind == "renewal_due":
        days = _safe_get(trigger, ["payload", "days_remaining"], None)
        plan = _safe_get(trigger, ["payload", "plan"], "")
        amount = _safe_get(trigger, ["payload", "renewal_amount"], None)
        amount_str = f" (renewal {amount})" if amount else ""
        body = f"{intro} your {plan} plan renews in {days} days{amount_str}. Want me to send the renewal link?"
        cta = "binary_yes_no"
        rationale = "Renewal due trigger; clear timeline and one-step CTA."
        template_params = [salutation, str(days), plan]

    elif kind == "festival_upcoming":
        festival = _safe_get(trigger, ["payload", "festival"], "festival")
        date = _safe_get(trigger, ["payload", "date"], "")
        offer = offer_title or "a festive offer"
        body = f"{intro} {festival} is on {date}. Want me to draft a {festival} post using {offer}?"
        cta = "binary_yes_no"
        rationale = "Festival trigger; propose a timely post using their offer."
        template_params = [salutation, festival, date]

    elif kind == "wedding_package_followup":
        wedding_date = _safe_get(trigger, ["payload", "wedding_date"], "")
        next_step = _safe_get(trigger, ["payload", "next_step_window_open"], "")
        body = f"{intro} your wedding is on {wedding_date}. Want to lock the {next_step.replace('_', ' ')} slot this week?"
        cta = "binary_yes_no"
        rationale = "Bridal followup trigger; anchor to wedding date and next-step window."
        template_params = [salutation, wedding_date]

    elif kind == "curious_ask_due":
        body = f"{intro} quick one — which service is getting the most asks this week?"
        cta = "open_ended"
        rationale = "Curious ask trigger; light-touch engagement with a single question."
        template_params = [salutation]

    elif kind == "winback_eligible":
        days = _safe_get(trigger, ["payload", "days_since_expiry"], None)
        lapsed = _safe_get(trigger, ["payload", "lapsed_customers_added_since_expiry"], None)
        offer = offer_title or "a winback offer"
        body = f"{intro} it's been {days} days since expiry; {lapsed} lapsed customers added since then. Want me to send {offer} to a winback list?"
        cta = "binary_yes_no"
        rationale = "Winback trigger; cite days and lapsed count, propose a targeted winback send."
        template_params = [salutation, str(days), str(lapsed)]

    elif kind == "ipl_match_today":
        match = _safe_get(trigger, ["payload", "match"], "match")
        match_time = _safe_get(trigger, ["payload", "match_time_iso"], "")
        offer = offer_title or "a match-night offer"
        body = f"{intro} {match} tonight ({match_time.split('T')[0]}). Want me to push a match-night post using {offer} for the 6pm window?"
        cta = "binary_yes_no"
        rationale = "IPL trigger; anchor to match timing and propose a targeted post."
        template_params = [salutation, match, offer]

    elif kind == "review_theme_emerged":
        theme = _safe_get(trigger, ["payload", "theme"], "")
        occurrences = _safe_get(trigger, ["payload", "occurrences_30d"], None)
        quote = _safe_get(trigger, ["payload", "common_quote"], "")
        quote_snip = f"\"{quote}\"" if quote else ""
        body = f"{intro} {occurrences} reviews this month mention {theme}. {quote_snip} Want me to draft a public response + a quick fix note?"
        cta = "binary_yes_no"
        rationale = "Review theme trigger; show evidence and offer response + fix note."
        template_params = [salutation, theme, str(occurrences)]

    elif kind == "milestone_reached":
        metric = _safe_get(trigger, ["payload", "metric"], "metric")
        value_now = _safe_get(trigger, ["payload", "value_now"], None)
        milestone = _safe_get(trigger, ["payload", "milestone_value"], None)
        gap = None
        if isinstance(value_now, (int, float)) and isinstance(milestone, (int, float)):
            gap = milestone - value_now
        gap_str = str(gap) if gap is not None else "a few"
        body = f"{intro} you are at {value_now} {metric}s — just {gap_str} away from {milestone}. Want a quick review-ask message to hit it?"
        cta = "binary_yes_no"
        rationale = "Milestone trigger; highlight the gap and offer a simple review-ask."
        template_params = [salutation, str(value_now), str(milestone)]

    elif kind == "active_planning_intent":
        intent = _safe_get(trigger, ["payload", "intent_topic"], "plan")
        offer = offer_title or "your best-selling item"
        body = (
            f"{intro} I can draft a 3-point plan for {intent.replace('_', ' ')} using {offer} as the anchor. "
            "What minimum headcount should I assume?"
        )
        cta = "open_ended"
        rationale = "Active planning intent; propose a draft and request one missing detail."
        template_params = [salutation, intent, offer]

    elif kind == "seasonal_perf_dip":
        metric = _safe_get(trigger, ["payload", "metric"], "metric")
        delta_pct = _safe_get(trigger, ["payload", "delta_pct"], 0)
        season_note = _safe_get(trigger, ["payload", "season_note"], "")
        note = f"Expected seasonal dip ({season_note})." if season_note else "Expected seasonal dip."
        body = f"{intro} {metric} down {_pct(delta_pct)} this week. {note} Want me to run a light promo to smooth it?"
        cta = "binary_yes_no"
        rationale = "Seasonal dip trigger; acknowledge seasonality and offer a light intervention."
        template_params = [salutation, metric, _pct(delta_pct)]

    elif kind == "customer_lapsed_hard":
        days = _safe_get(trigger, ["payload", "days_since_last_visit"], None)
        focus = _safe_get(trigger, ["payload", "previous_focus"], "fitness")
        body = f"{intro} it's been {days} days since your last visit. Want to restart with a {focus.replace('_', ' ')} check-in this week?"
        cta = "binary_yes_no"
        rationale = "Customer winback trigger; cite gap and suggest a focused re-entry."
        template_params = [salutation, str(days), focus]

    elif kind == "trial_followup":
        slots = _safe_get(trigger, ["payload", "next_session_options"], [])
        slot_labels = _render_slots(slots)
        if slot_labels:
            body = f"{intro} thanks for the trial! Next slot: {slot_labels[0]}. Want me to book it?"
            cta = "binary_yes_no"
        else:
            body = f"{intro} thanks for the trial! Want to schedule the next session?"
            cta = "binary_yes_no"
        rationale = "Trial followup trigger; offer a specific next session."
        template_params = [salutation, slot_labels[0] if slot_labels else ""]

    elif kind == "supply_alert":
        molecule = _safe_get(trigger, ["payload", "molecule"], "")
        batches = _safe_get(trigger, ["payload", "affected_batches"], [])
        batch_str = ", ".join(batches[:2])
        body = f"{intro} supply alert: {molecule} recall for batches {batch_str}. Want a staff checklist and customer notice draft?"
        cta = "binary_yes_no"
        rationale = "Supply alert trigger; list affected molecule and batches and offer checklist."
        template_params = [salutation, molecule, batch_str]

    elif kind == "chronic_refill_due":
        molecules = _safe_get(trigger, ["payload", "molecule_list"], [])
        runs_out = _safe_get(trigger, ["payload", "stock_runs_out_iso"], "")
        delivery = _safe_get(trigger, ["payload", "delivery_address_saved"], False)
        med_list = ", ".join(molecules[:3])
        delivery_str = "We can deliver to your saved address." if delivery else ""
        body = f"{intro} refill due for {med_list} before {runs_out.split('T')[0]}. {delivery_str} Want me to arrange a refill?"
        cta = "binary_yes_no"
        rationale = "Chronic refill trigger; list molecules and stock-out date, offer refill."
        template_params = [salutation, med_list, runs_out]

    elif kind == "category_seasonal":
        trends = _safe_get(trigger, ["payload", "trends"], [])
        top_trend = trends[0] if trends else "seasonal demand shift"
        body = f"{intro} summer shift: {top_trend}. Want a quick shelf plan for the top 3 movers?"
        cta = "binary_yes_no"
        rationale = "Seasonal category trigger; cite the top trend and offer a shelf plan."
        template_params = [salutation, top_trend]

    elif kind == "gbp_unverified":
        path = _safe_get(trigger, ["payload", "verification_path"], "verification")
        uplift = _safe_get(trigger, ["payload", "estimated_uplift_pct"], None)
        uplift_str = _pct(uplift) if uplift is not None else ""
        body = f"{intro} your GBP is unverified. Verification via {path} can lift calls by {uplift_str}. Want me to start it?"
        cta = "binary_yes_no"
        rationale = "GBP unverified trigger; clear benefit and action."
        template_params = [salutation, path, uplift_str]

    elif kind == "cde_opportunity":
        digest_id = _safe_get(trigger, ["payload", "digest_item_id"], "")
        item = _pick_digest_item(category, digest_id) if digest_id else None
        title = _safe_get(item or {}, ["title"], "CDE session")
        date = _safe_get(item or {}, ["date"], "")
        credits = _safe_get(trigger, ["payload", "credits"], None)
        body = f"{intro} CDE opportunity: {title} on {date} ({credits} credits). Want me to register you?"
        cta = "binary_yes_no"
        rationale = "CDE trigger; specify title, date, credits and offer registration."
        template_params = [salutation, title, date]

    elif kind == "competitor_opened":
        name = _safe_get(trigger, ["payload", "competitor_name"], "a competitor")
        distance = _safe_get(trigger, ["payload", "distance_km"], None)
        their_offer = _safe_get(trigger, ["payload", "their_offer"], "")
        offer = offer_title or "your top offer"
        body = f"{intro} {name} opened {distance} km away offering {their_offer}. Want me to counter with a post featuring {offer}?"
        cta = "binary_yes_no"
        rationale = "Competitor trigger; name, distance, offer; propose a counter-post."
        template_params = [salutation, name, offer]

    elif kind == "dormant_with_vera":
        days = _safe_get(trigger, ["payload", "days_since_last_merchant_message"], None)
        body = f"{intro} it has been {days} days since we last spoke. Want a quick 2-line update on your profile health?"
        cta = "binary_yes_no"
        rationale = "Dormancy trigger; low-friction re-entry question."
        template_params = [salutation, str(days)]

    else:
        body = f"{intro} I have a quick update for you. Want me to share it?"
        cta = "binary_yes_no"
        rationale = "Fallback; keep CTA simple and avoid inventing details."
        template_params = [salutation]

    body = body.strip()
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
        "template_name": f"{'merchant' if scope == 'customer' else 'vera'}_{kind}_v1",
        "template_params": template_params,
    }


class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
async def healthz():
    counts = {scope: len(store) for scope, store in contexts.items()}
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vera",
        "team_members": ["Builder"],
        "model": "deterministic-rules",
        "approach": "rule-based composer with trigger-driven templates",
        "contact_email": "you@example.com",
        "version": "0.1.0",
        "submitted_at": _now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: ContextPush):
    if body.scope not in SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"unknown scope: {body.scope}",
        })

    store = contexts[body.scope]
    current = store.get(body.context_id)
    if current and current.get("version", -1) >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False,
            "reason": "stale_version",
            "current_version": current.get("version"),
        })

    store[body.context_id] = {
        "version": body.version,
        "payload": body.payload,
        "delivered_at": body.delivered_at,
        "stored_at": _now_iso(),
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": _now_iso(),
    }


def _get_context(scope: str, context_id: str) -> Optional[Dict[str, Any]]:
    record = contexts.get(scope, {}).get(context_id)
    return record.get("payload") if record else None


def _trigger_score(trigger: Dict[str, Any], now: Optional[datetime]) -> int:
    urgency = int(trigger.get("urgency", 1))
    score = urgency * 100

    kind = trigger.get("kind", "")
    if kind in {"active_planning_intent", "supply_alert"}:
        score += 80
    if kind in {"regulation_change", "perf_dip", "renewal_due"}:
        score += 40
    if kind in {"recall_due", "chronic_refill_due"}:
        score += 30

    expires_at = _parse_iso(trigger.get("expires_at"))
    if expires_at and now:
        days_left = max(0, (expires_at - now).days)
        score += max(0, 10 - days_left)

    return score


def _eligible_trigger(trigger: Dict[str, Any], now: Optional[datetime]) -> bool:
    if trigger.get("id") in handled_triggers:
        return False

    suppression_key = trigger.get("suppression_key")
    if suppression_key and suppression_key in suppression_keys_used:
        return False

    expires_at = _parse_iso(trigger.get("expires_at"))
    if expires_at and now and expires_at < now:
        return False

    merchant_id = trigger.get("merchant_id")
    if not merchant_id or not _get_context("merchant", merchant_id):
        return False

    if trigger.get("scope") == "customer":
        customer_id = trigger.get("customer_id")
        customer = _get_context("customer", customer_id) if customer_id else None
        if not customer:
            return False
        consent = customer.get("consent", {}).get("scope", [])
        if not consent:
            return False

    return True


def _select_trigger(trigger_ids: List[str], now: Optional[datetime]) -> Optional[Dict[str, Any]]:
    candidates = []
    for trigger_id in trigger_ids:
        trigger = _get_context("trigger", trigger_id)
        if not trigger:
            continue
        if not _eligible_trigger(trigger, now):
            continue
        score = _trigger_score(trigger, now)
        candidates.append((score, trigger_id, trigger))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][2]


def _conversation_id(trigger: Dict[str, Any]) -> str:
    return f"conv_{trigger.get('id')}"


def _record_conversation(conversation_id: str, payload: Dict[str, Any]) -> None:
    conversations[conversation_id] = payload


def _track_sent(merchant_id: str, body: str, suppression_key: str) -> None:
    sent_bodies.add(_hash_text(f"{merchant_id}:{body}"))
    if suppression_key:
        suppression_keys_used.add(suppression_key)


@app.post("/v1/tick")
async def tick(body: TickRequest):
    now = _parse_iso(body.now)
    trigger = _select_trigger(body.available_triggers, now)
    if not trigger:
        return {"actions": []}

    merchant = _get_context("merchant", trigger.get("merchant_id"))
    if not merchant:
        return {"actions": []}

    category = _get_context("category", merchant.get("category_slug")) or {}
    customer = _get_context("customer", trigger.get("customer_id")) if trigger.get("customer_id") else None

    message = compose(category, merchant, trigger, customer)
    body_hash = _hash_text(f"{merchant.get('merchant_id')}:{message['body']}")
    if body_hash in sent_bodies:
        return {"actions": []}

    convo_id = _conversation_id(trigger)
    handled_triggers.add(trigger.get("id"))
    _track_sent(merchant.get("merchant_id"), message["body"], message.get("suppression_key"))

    _record_conversation(convo_id, {
        "merchant_id": merchant.get("merchant_id"),
        "customer_id": trigger.get("customer_id"),
        "trigger_id": trigger.get("id"),
        "kind": trigger.get("kind"),
        "last_body": message["body"],
        "auto_reply_hits": 0,
        "slots": _render_slots(_safe_get(trigger, ["payload", "available_slots"], [])),
        "status": "open",
    })

    action = {
        "conversation_id": convo_id,
        "merchant_id": merchant.get("merchant_id"),
        "customer_id": trigger.get("customer_id"),
        "send_as": message["send_as"],
        "trigger_id": trigger.get("id"),
        "template_name": message["template_name"],
        "template_params": message["template_params"],
        "body": message["body"],
        "cta": message["cta"],
        "suppression_key": message["suppression_key"],
        "rationale": message["rationale"],
    }

    return {"actions": [action]}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    convo = conversations.get(body.conversation_id)
    if not convo or convo.get("status") != "open":
        return {"action": "end", "rationale": "Conversation not found or already closed."}

    text = body.message.strip()

    if _detect_opt_out(text):
        convo["status"] = "ended"
        return {"action": "end", "rationale": "Merchant opted out; closing conversation."}

    if _detect_auto_reply(text):
        convo["auto_reply_hits"] += 1
        if convo["auto_reply_hits"] >= 3:
            convo["status"] = "ended"
            return {"action": "end", "rationale": "Auto-reply loop detected; ending conversation."}
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Auto-reply detected; waiting for owner."}

    if _detect_off_topic(text):
        return {
            "action": "send",
            "body": "I am not able to help with that, but I can help with your merchant growth tasks here. Want me to continue with the last update?",
            "cta": "binary_yes_no",
            "rationale": "Off-topic request; redirect to the active conversation.",
        }

    if _detect_negative(text):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked to pause; backing off."}

    if _detect_positive(text):
        kind = convo.get("kind")
        if kind in {"research_digest", "regulation_change", "cde_opportunity"}:
            return {
                "action": "send",
                "body": "Sending it now. Want me to also draft a short post you can share?",
                "cta": "binary_yes_no",
                "rationale": "Merchant accepted; deliver and offer the next best step.",
            }
        if kind in {"recall_due", "trial_followup"} and convo.get("slots"):
            slot = convo["slots"][0]
            return {
                "action": "send",
                "body": f"Got it. I can hold {slot} for you. If you need a different time, reply here.",
                "cta": "none",
                "rationale": "Customer confirmed a slot; acknowledge and keep the channel open.",
            }
        return {
            "action": "send",
            "body": "Great. I will draft it and share shortly. Any specific detail to include?",
            "cta": "open_ended",
            "rationale": "Merchant accepted; proceed and ask for one detail if needed.",
        }

    return {
        "action": "send",
        "body": "Thanks for the reply. Want me to proceed with a draft based on the last update?",
        "cta": "binary_yes_no",
        "rationale": "Keep the thread moving with a simple next step.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
