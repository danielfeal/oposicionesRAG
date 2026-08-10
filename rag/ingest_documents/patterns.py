"""Regex patterns for the structure of Spanish legal documents (Título, Capítulo,
Artículo, Disposición) and for numbering the headings within them.
"""

import re

# Spanish ordinal/cardinal number words used in article and disposition headings.
# Units/tens and hundreds are kept apart so that a hundreds word is never matched as the
# unit that prefixes it (regex alternation is leftmost-first, so a combined list would
# match "cuatro" inside "cuatrocientos"). Inside each group longer words come first for
# the same reason ("ciento" before "cien").
_UNIT = (
    r"(?:[uú]nico|[uú]nica|primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|d[eé]cimo"
    r"|und[eé]cimo|duod[eé]cimo|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez"
    r"|once|doce|trece|catorce|quince|diecis[eé]is|diecisiete|dieciocho|diecinueve"
    r"|veinti\w+|veinte|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa"
    r")"
)
_HUNDRED = (
    r"(?:ciento|cien|doscientos|trescientos|cuatrocientos|quinientos"
    r"|seiscientos|setecientos|ochocientos|novecientos"
    r")"
)
# "treinta", "treinta y dos", "veintiuno"
_TENS = rf"{_UNIT}(?:\s+y\s+{_UNIT})?"
# "132", "ciento treinta y dos", "cuatrocientos uno", "cien", "treinta y dos"
_NUM = rf"(?:\d+|{_HUNDRED}(?:\s+{_TENS})?|{_TENS})"

# Ordinal word stems (gender suffix [oa] applied separately), used for headings that are just
# a bare ordinal on its own line ("Primero.", "Primera.") and for the ordinal suffix of a
# "Disposición <tipo> <ordinal>." heading. First letter matches either case: capitalized when
# the ordinal opens the line ("Primero."), lowercase when it's part of a longer heading it
# doesn't start ("Disposición adicional vigésima primera."). Tens and units are kept apart so
# a compound ordinal like "vigésima quinta" can be matched as one heading instead of just its
# first word.
_ORDINAL_UNIT_STEM = (
    r"(?:[UuÚú]nic|[Pp]rimer|[Ss]egund|[Tt]ercer|[Cc]uart|[Qq]uint|[Ss]ext|[Ss][eé]ptim|[Oo]ctav|[Nn]oven"
    r"|[Uu]nd[eé]cim|[Dd]uod[eé]cim"
    r")"
)
# Tens that can also carry a unit as a separate word ("vigésima quinta", "décimo octava").
# "décim" belongs here rather than with the units so the spaced teens are matched whole.
_ORDINAL_TENS_STEM = (
    r"(?:[Vv]ig[eé]sim|[Tt]rig[eé]sim|[Cc]uadrag[eé]sim|[Qq]uincuag[eé]sim"
    r"|[Ss]exag[eé]sim|[Ss]eptuag[eé]sim|[Oo]ctog[eé]sim|[Nn]onag[eé]sim|[Dd][eé]cim"
    r")"
)
# Teens/twenties written as a single word ("decimotercera", "vigesimoctava"). The unit is
# glued to the tens with an "o" ("decimo-" + "quinta"), except for "octava", which absorbs it
# ("decimoctava"). Must be tried before the plain tens stem, or "decimotercera" matches only
# its "décim" prefix and the heading is truncated to "decimo".
_ORDINAL_COMPOUND_STEM = (
    r"(?:(?:[Dd]ecim|[Vv]igesim)"
    r"(?:o(?:primer|segund|tercer|cuart|quint|sext|s[eé]ptim|noven)|octav)"
    r")"
)
# Ordinal suffixes that mark an article or disposición inserted by a later reform
# ("Artículo 26 quáter.", "Disposición adicional novena bis."). Both the accented and
# unaccented spellings of "quáter" occur in the corpus. Ordered longest-first: regex
# alternation matches the first alternative that succeeds, not the longest one, so
# "terdecies"/"quaterdecies" would otherwise be cut short as "ter"/"quater" (a real
# prefix of them) with "decies" left dangling outside the match.
_ORDINAL_SUFFIX = (
    r"(?:quaterdecies|septendecies"
    r"|duodevicies"
    r"|quindecies|undevicies"
    r"|quinquies|duodecies|terdecies|sexdecies"
    r"|undecies"
    r"|septies"
    r"|qu[aá]ter|sexies|octies|nonies|decies|vicies"
    r"|bis|ter"
    r")"
)
# A single-word compound ("Decimotercera", "Vigesimoctava"), a spaced tens+unit ordinal
# ("Vigésima quinta", "Décimo octava") or a plain single-word one ("Primero", "Duodécima",
# or an exact "Vigésima" alone for a round 20th/30th). The tens/unit separator is `\s*`
# rather than `\s+` because PDF extraction sometimes drops the space ("vigésimaprimera");
# the zero-width case is only reachable when the two words are glued together.
_ORDINAL_WORD = (
    rf"(?:{_ORDINAL_COMPOUND_STEM}[oa]"
    rf"|{_ORDINAL_TENS_STEM}[oa](?:\s*{_ORDINAL_UNIT_STEM}[oa])?"
    rf"|{_ORDINAL_UNIT_STEM}[oa])"
)

# A handful of documents embed a second, self-contained document after the outer decree's own body,
# under one of these known titles. Matched literally since these are specific, known cases - not a
# general pattern. Whitespace between words is relaxed to `\s+` because PDF extraction wraps the
# longer titles over several lines.
EMBEDDED_DOCUMENT_TITLES = (
    "Carta de los Derechos Fundamentales de la Unión Europea",
    "ESTATUTO DE LA OFICINA ESPAÑOLA DE PATENTES Y MARCAS, O.A.",
    "ESTATUTO DEL CONSEJO DE TRANSPARENCIA Y BUEN GOBIERNO",
    "ESTATUTO DEL INSTITUTO DE TURISMO DE ESPAÑA (TURESPAÑA)",
    "ESTRATEGIA DE SEGURIDAD NACIONAL 2021",
    "ESTRATEGIA ESPAÑOLA DE APOYO ACTIVO AL EMPLEO 2025-2028",
    "IV CONVENIO ÚNICO PARA EL PERSONAL LABORAL DE LA ADMINISTRACIÓN GENERAL DEL ESTADO",
    "PLAN GENERAL DE CONTABILIDAD",
    "PLAN GENERAL DE CONTABILIDAD PÚBLICA",
    "REGLAMENTO DE ACTUACIÓN Y FUNCIONAMIENTO DEL SECTOR PÚBLICO POR MEDIOS ELECTRÓNICOS",
    "REGLAMENTO DE BIENES DE LAS ENTIDADES LOCALES",
    "REGLAMENTO DE DISCIPLINA URBANÍSTICA PARA EL DESARROLLO DE LA LEY SOBRE RÉGIMEN DEL SUELO Y ORDENACIÓN URBANA",
    "REGLAMENTO DE EVALUACIÓN Y CERTIFICACIÓN DE LA SEGURIDAD DE LAS TECNOLOGÍAS DE LA INFORMACIÓN",
    "REGLAMENTO DE LA LEY 38/2003, DE 17 DE NOVIEMBRE, GENERAL DE SUBVENCIONES",
    "REGLAMENTO DE LA LEY DE EXPROPIACIÓN FORZOSA",
    "REGLAMENTO DE LA LEY ORGÁNICA 4/2000, DE 11 DE ENERO, SOBRE DERECHOS Y LIBERTADES DE LOS EXTRANJEROS EN ESPAÑA Y SU INTEGRACIÓN SOCIAL",
    "REGLAMENTO DE REGIMEN DISCIPLINARIO DE LOS FUNCIONARIOS DE LA ADMINISTRACION DEL ESTADO",
    "REGLAMENTO DE REUTILIZACIÓN DEL AGUA",
    "REGLAMENTO DE SITUACIONES ADMINISTRATIVAS DE LOS FUNCIONARIOS CIVILES DE LA ADMINISTRACION GENERAL DEL ESTADO",
    "REGLAMENTO DEL REGISTRO CENTRAL DE PERSONAL",
    "REGLAMENTO GENERAL DE DESARROLLO DE LA LEY 58/2003, DE 17 DE DICIEMBRE, GENERAL TRIBUTARIA, EN MATERIA DE REVISIÓN EN VÍA ADMINISTRATIVA",
    "REGLAMENTO GENERAL DE INGRESO DEL PERSONAL AL SERVICIO DE LA ADMINISTRACION GENERAL DEL ESTADO Y DE PROVISION DE PUESTOS DE TRABAJO Y PROMOCION PROFESIONAL DE LOS FUNCIONARIOS CIVILES DE LA ADMINISTRACION GENERAL DEL ESTADO",
    "REGLAMENTO GENERAL DE LA LEY 33/2003, DE 3 DE NOVIEMBRE, DEL PATRIMONIO DE LAS ADMINISTRACIONES PÚBLICAS",
    "REGLAMENTO GENERAL DEL MUTUALISMO ADMINISTRATIVO",
    "REGLAMENTO ORGÁNICO DEL CONSEJO DE ESTADO",
    "REGLAMENTO SOBRE LAS CONDICIONES BÁSICAS PARA EL ACCESO DE LAS PERSONAS CON DISCAPACIDAD A LAS TECNOLOGÍAS, PRODUCTOS Y SERVICIOS RELACIONADOS CON LA SOCIEDAD DE LA INFORMACIÓN Y MEDIOS DE COMUNICACIÓN SOCIAL",
    "TEXTO REFUNDIDO DE LA LEY DE AGUAS",
    "TEXTO REFUNDIDO DE LA LEY DE EMPLEO",
    "TEXTO REFUNDIDO DE LA LEY DE PREVENCIÓN Y CONTROL INTEGRADOS DE LA CONTAMINACIÓN",
    "TEXTO REFUNDIDO DE LA LEY DE PROPIEDAD INTELECTUAL",
    "TEXTO REFUNDIDO DE LA LEY DE PUERTOS DEL ESTADO Y DE LA MARINA MERCANTE",
    "TEXTO REFUNDIDO DE LA LEY DE SOCIEDADES DE CAPITAL",
    "TEXTO REFUNDIDO DE LA LEY DE SUELO Y REHABILITACIÓN URBANA",
    "TEXTO REFUNDIDO DE LA LEY DEL ESTATUTO BÁSICO DEL EMPLEADO PÚBLICO",
    "TEXTO REFUNDIDO DE LA LEY DEL ESTATUTO DE LOS TRABAJADORES",
    "TEXTO REFUNDIDO DE LA LEY GENERAL DE LA SEGURIDAD SOCIAL",
    "TEXTO REFUNDIDO DE LA LEY REGULADORA DE LAS HACIENDAS LOCALES",
    "TEXTO REFUNDIDO DE LA LEY SOBRE INFRACCIONES Y SANCIONES EN EL ORDEN SOCIAL",
    "TEXTO REFUNDIDO DE LA LEY SOBRE SEGURIDAD SOCIAL DE LOS FUNCIONARIOS CIVILES DEL ESTADO",
    "TEXTO REFUNDIDO DE LEY DE CLASES PASIVAS DEL ESTADO",
)
EMBEDDED_DOCUMENT_TITLE = re.compile(
    r"(?m)^(?:"
    + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in EMBEDDED_DOCUMENT_TITLES)
    + r")\s*$"
)

# Phrases that commonly introduce a cross-reference to another section in Spanish legal prose
# ("... a que se refiere la Disposición adicional cuadragésima octava..."). Used to catch
# heading-shaped text that is actually mid-sentence, not a real section boundary.
CITATION_LEADIN = re.compile(
    r"(a que se refiere|al que se refiere|a la que se refiere|a los que se refiere|a las que se refiere"
    r"|de conformidad con|de acuerdo con|conforme a lo (?:dispuesto|establecido|previsto) en"
    r"|con arreglo a|lo (?:dispuesto|establecido|previsto) en|regulad[oa] en|prevista? en"
    r")\s*[a-z]{0,3}\s*\Z",
    re.IGNORECASE,
)

# Regex patterns matching the structure of Spanish legal documents (Título, Capítulo, Artículo, Disposición).
# Case-sensitive on purpose: real headings are always uppercase (TÍTULO/CAPÍTULO) or title-case
# (Artículo/Disposición).
PATTERNS = {
    # Roman numbering is the norm, but a few documents number their divisions with
    # arabic digits ("CAPÍTULO 1").
    "titulo": re.compile(r"(?m)^\s*T[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)\b.*$"),
    "capitulo": re.compile(r"(?m)^\s*CAP[IÍ]TULO\s+(?:[IVXLCDM]+|\d+)\b.*$"),
    "articulo_disposicion": re.compile(
        rf"(?m)^\s*("
        rf"Art[ií]culo\s+{_NUM}\b\s*(?:{_ORDINAL_SUFFIX})?\.?"
        rf"|Disposici[oó]n\s+(?:adicional|transitoria|final|derogatoria)"
        rf"(?:\s+(?:{_ORDINAL_WORD}|\w+))?(?:\s+{_ORDINAL_SUFFIX})?(?:\s*\[sic\])?\."
        r")"
    ),
    # Older-style documents group unnumbered disposición items under an ALL-CAPS section header.
    "disposicion_categoria": re.compile(
        r"(?m)^\s*DISPOSICI[OÓ]N(?:ES)?\s+(ADICIONAL(?:ES)?|TRANSITORIAS?|FINAL(?:ES)?|DEROGATORIAS?)\s*$"
    ),
    # Bare ordinal heading ("Primero.", "Segunda.").
    "ordinal_bare": re.compile(rf"(?m)^\s*({_ORDINAL_WORD})\.(?=\s+\S)"),
    # Bare numeric heading ("1.", "2."...).
    "numero_bare": re.compile(r"(?m)^\s*(\d{1,3})\.(?=\s+\S)"),
    "texto_consolidado": re.compile(r"(?m)^.*TEXTO CONSOLIDADO.*$"),
    # A handful of documents (e.g. doc215) use "ANEJO" instead of "ANEXO".
    "anexo": re.compile(r"(?m)^\s*ANE(?:XO|JO)(?:\s+[IVXLCDM]+)?\b.*$"),
}

# An article headed by an ordinal word ("Artículo segundo.") rather than a number. Amending
# laws use this form for their own top-level divisions, while the amended text they quote
# always uses numbered articles - so an ordinal article is never quoted content.
ARTICULO_ORDINAL = re.compile(rf"^Art[ií]culo\s+{_ORDINAL_WORD}\b")
