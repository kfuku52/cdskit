def strict_uniprot_targetp_label(location_text, organism_group):
    txt = location_assertions(location_text)
    organism_group = str(organism_group or "").strip().lower()
    if txt.strip() == "":
        return None, "missing_location_text"

    if any(
        term in str(location_text or "").lower()
        for term in ("unconventional", "nonclassical")
    ):
        return None, "unconventional_secretion"
    has_sp = ("secreted" in txt) or ("signal peptide" in txt)
    has_mtp = "mitochond" in txt
    has_ctp = ("chloroplast" in txt) or ("plastid" in txt)
    has_thylakoid = "thylakoid" in txt
    has_lumen = ("lumen" in txt) or ("lumenal" in txt) or ("luminal" in txt)

    if organism_group != "plant" and (has_ctp or has_thylakoid):
        return None, "nonplant_plastid"

    plastid_signal = organism_group == "plant" and has_ctp
    ltp_signal = organism_group == "plant" and has_thylakoid and has_lumen
    if sum(1 for value in [has_sp, has_mtp, plastid_signal] if value) > 1:
        return None, "ambiguous"
    if ltp_signal and (has_sp or has_mtp):
        return None, "ambiguous"

    if ltp_signal:
        if "membrane" in txt and "thylakoid lumen" not in txt:
            return None, "ltp_membrane_noise"
        return "lTP", ""
    if plastid_signal:
        if has_thylakoid:
            return None, "thylakoid_not_lumen"
        return "cTP", ""
    if has_mtp:
        return "mTP", ""
    if has_sp:
        return "SP", ""
    return "noTP", ""


def targeting_label_from_evidence(text):
    """Read curated experimental targeting evidence; never infer from location.

    SIGNAL/TRANSIT coordinates may be unknown, but the record must cite an
    experiment supporting peptide presence (or absence for noTP).
    """
    import json
    from cdskit.localize_labels import validate_label_evidence

    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("Targeting evidence must be a JSON list.")
    positive = [
        r.get("label")
        for r in records
        if isinstance(r, dict) and r.get("state") == "positive"
    ]
    if len(set(positive)) != 1 or set(positive) - {"SP", "mTP", "cTP", "lTP", "noTP"}:
        raise ValueError(
            "Exactly one experimentally supported targeting class is required."
        )
    validate_label_evidence(text, positive, [], "experimental")
    for record in records:
        if record["label"] != "noTP" and record.get("feature_type") not in (
            "SIGNAL",
            "TRANSIT",
        ):
            raise ValueError(
                "Peptide evidence requires feature_type SIGNAL or TRANSIT."
            )
        if record["label"] == "SP" and record.get("feature_type") != "SIGNAL":
            raise ValueError("SP evidence requires SIGNAL.")
        if (
            record["label"] in ("mTP", "cTP", "lTP")
            and record.get("feature_type") != "TRANSIT"
        ):
            raise ValueError("Organelle peptide evidence requires TRANSIT.")
        start, end = record.get("start"), record.get("end")
        if (
            (start is not None and (type(start) is not int or start < 1))
            or (end is not None and (type(end) is not int or end < 1))
            or (start is not None and end is not None and end < start)
        ):
            raise ValueError(
                "Invalid peptide coordinates (one-based inclusive or null)."
            )
    return positive[0]


def location_assertions(location_text):
    """Conservatively retain positive location clauses, excluding free-text notes.

    This only builds weak localization proxies, not peptide evidence. Negated
    clauses are omitted rather than parsed into an experimental negative.
    """
    import re

    text = str(location_text or "").lower().split("note=", 1)[0]
    clauses = re.split(r"[.;]", text)
    return "; ".join(
        clause
        for clause in clauses
        if not re.search(
            r"\b(not|no|never|without|unconventional|nonclassical)\b", clause
        )
    )
