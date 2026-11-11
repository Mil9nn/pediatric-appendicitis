        if unit.startswith("month"):
            age /= 12
        elif unit.startswith("day"):
            age /= 365

        found["Age"] = round(age, 1)
        found["Sex"] = match.group(3).lower()

    return found


def extract_vitals(text):
    """Extract height, weight, and temperature when present."""
    patterns = {
        "Height": r"Height\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*cm",
        "Weight": r"Weight\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*kg",
        "Body_Temperature": r"(?:Body\s*)?Temp(?:erature)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*([CF])?",
    }

    found = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            value = float(match.group(1))

            if name == "Body_Temperature":
                unit = (match.group(2) or "").upper()
                if unit == "F" or value > 45:
                    value = (value - 32) * 5 / 9

            found[name] = round(value, 1)

    return found


def finding(text, words):
    """Return yes/no when a finding is mentioned with clear negation handling."""
    for match in re.finditer(words, text, flags=re.IGNORECASE):
        before = re.split(
            r"[\n.;]",
            text[max(0, match.start() - 40):match.start()],
        )[-1].lower()

        after = re.split(
            r"[\n.;]",
            text[match.end():match.end() + 25],
        )[0].lower()

        if re.search(r"\b(no|not|without|absent|negative|nil|none)\b", before):
            return "no"

        if re.search(r"\b(absent|not seen|not visible|nil|negative|none)\b", after):
            return "no"

        return "yes"

    return None


def extract_ultrasound(text):
    """Extract supported ultrasound findings from report text."""
    found = {"US_Performed": "yes"}

    # Appendix diameter is stored in millimetres.
    match = (
        re.search(
            r"appendi(?:x|ceal)[^.\n]{0,60}?([0-9]+(?:\.[0-9]+)?)\s*(mm|cm)",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"diameter[^.\n]{0,30}?([0-9]+(?:\.[0-9]+)?)\s*(mm|cm)",
            text,
            flags=re.IGNORECASE,
        )
    )

    if match:
        diameter = float(match.group(1))
        if match.group(2).lower() == "cm":
            diameter *= 10

        found["Appendix_Diameter"] = float(
            min(max(round(diameter, 1), 0.0), 30.0)
        )

    # Determine whether the appendix was visualised.
    if (
        re.search(
            r"appendix[^.\n]*\b(?:not|non)[ -]?(?:visuali[sz]ed|seen|identified|visible)",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:unable to|could not|cannot) (?:visuali[sz]e|identify|see)[^.\n]*appendix",
            text,
            flags=re.IGNORECASE,
        )
    ):
        found["Appendix_on_US"] = "no"

    elif (
        re.search(
            r"appendix[^.\n]*(?:visuali[sz]ed|seen|identified|visible)",
            text,
            flags=re.IGNORECASE,
        )
        or "Appendix_Diameter" in found
    ):
        found["Appendix_on_US"] = "yes"

    fluid = finding(text, r"free\s+(?:\w+\s+){0,2}fluids?")
    if fluid:
        found["Free_Fluids"] = fluid

    tissue = finding(
        text,
        r"fat stranding|periappendiceal (?:fat|inflammat\w*|reaction)|"
        r"surrounding (?:tissue|fat) (?:reaction|inflammat\w*)|"
        r"inflammatory (?:changes|reaction)",
    )
    if tissue:
        found["Surrounding_Tissue_Reaction"] = tissue

    return found