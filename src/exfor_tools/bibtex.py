import re


def bibtex_article_from_x4bibmetadata(meta, citekey=None):
    """
    Convert X4BibMetaData to a BibTeX @article.

    Parsing policy (based on EXFOR pretty reference strings):
      - Use str(meta.reference) as the authoritative source to derive `journal`.
      - If multiple references are present separated by ';', use the first one for
        journal/volume/pages/year and preserve the full string in `note`.
      - Try to parse patterns like:
          "Journal Name 116, 643 (1968)"
          "Journal Name 56, (2), 900 (1997)"
          "Journal Name 62, (4), 044609 (2000)"
      - Fall back gracefully: if parsing fails, emit only `note` and whatever meta already provides.
    """

    def norm(s):
        return re.sub(r"\s+", " ", str(s).strip())

    def tex_escape(s):
        s = norm(s)
        return (
            s.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
            .replace("_", r"\_")
        )

    def lastname(name):
        name = norm(name)
        if not name:
            return "exfor"
        if "," in name:
            base = name.split(",", 1)[0]
        else:
            parts = name.split()
            base = parts[-1] if parts else name
        base = re.sub(r"[^A-Za-z0-9]+", "", base)
        return base or "exfor"

    def first_reference_chunk(ref_str):
        # EXFOR sometimes concatenates multiple refs with semicolons.
        # Use the first for structured BibTeX fields, keep full string in note.
        return norm(ref_str.split(";", 1)[0])

    def parse_pretty_reference(ref_chunk):
        """
        Returns dict with journal, volume, number(issue), pages, year.
        All keys optional if not found.
        """
        out = {}

        # Year: last "(YYYY)" at end
        m = re.search(r"\((?P<year>\d{4})\)\s*$", ref_chunk)
        if not m:
            return out
        out["year"] = m.group("year")

        prefix = ref_chunk[: m.start()].rstrip()

        # Common tails:
        #   "... <vol>, <pages>"
        #   "... <vol>, (<issue>), <pages>"
        tail_re = re.compile(
            r"(?P<journal>.+?)\s+"
            r"(?P<volume>\d+)\s*,\s*"
            r"(?:(?:\((?P<number>[^)]+)\)\s*,\s*)?)"
            r"(?P<pages>[A-Za-z0-9]+(?:\s*[-–]\s*[A-Za-z0-9]+)?)\s*$"
        )
        m2 = tail_re.match(prefix)
        if not m2:
            # If we can’t get vol/pages, still try to produce a journal guess:
            # take everything before the last number-comma segment.
            out["journal"] = prefix
            return out

        out["journal"] = m2.group("journal").rstrip(" ,")
        out["volume"] = m2.group("volume")
        if m2.group("number"):
            out["number"] = norm(m2.group("number"))
        out["pages"] = norm(m2.group("pages")).replace(" ", "")
        return out

    # ---- author/title/year (from meta) ----
    authors = meta.author if isinstance(meta.author, (list, tuple)) else []
    authors = [norm(a) for a in authors if norm(a)]
    author_bib = " and ".join(tex_escape(a) for a in authors) if authors else None

    title = norm(getattr(meta, "title", "")) or None

    year = getattr(meta, "year", None)
    year = None if year in (None, "None", "") else norm(year)

    # ---- reference string ----
    ref_obj = getattr(meta, "reference", None)
    ref_str = None
    if ref_obj not in (None, "None"):
        ref_str = norm(str(ref_obj))

    parsed = {}
    if ref_str:
        parsed = parse_pretty_reference(first_reference_chunk(ref_str))

    # Prefer parsed year if meta.year is missing
    if not year and parsed.get("year"):
        year = parsed["year"]

    journal = parsed.get("journal")
    volume = parsed.get("volume")
    number = parsed.get("number")
    pages = parsed.get("pages")

    # ---- citekey ----
    if citekey is None:
        base = lastname(authors[0]) if authors else "exfor"
        y = year or "nd"
        subent = (
            re.sub(r"[^A-Za-z0-9]+", "", norm(getattr(meta, "subent", "subent")))
            or "subent"
        )
        citekey = f"{base}{y}_{subent}"

    # ---- assemble bibtex ----
    fields = []
    if author_bib:
        fields.append(("author", author_bib))
    if title:
        fields.append(("title", tex_escape(title)))
    if journal:
        fields.append(("journal", tex_escape(journal)))
    if year:
        fields.append(("year", tex_escape(year)))
    if volume:
        fields.append(("volume", tex_escape(volume)))
    if number:
        fields.append(("number", tex_escape(number)))
    if pages:
        fields.append(("pages", tex_escape(pages)))

    # Keep full EXFOR reference string no matter what
    note_parts = []
    if ref_str:
        note_parts.append(f"EXFOR reference: {ref_str}")
    subent = norm(getattr(meta, "subent", ""))
    if subent and subent != "????????":
        note_parts.append(f"EXFOR subent: {subent}")
    if note_parts:
        fields.append(("note", tex_escape("; ".join(note_parts))))

    lines = [f"@article{{{citekey},"]
    for k, v in fields:
        lines.append(f"  {k} = {{{v}}},")
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)
