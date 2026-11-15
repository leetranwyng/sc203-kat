"""Normalize VLM entity labels and resolve a real scene viewer without camera fallback."""

import ast
import json
import os
from pathlib import Path
import re
import unicodedata


class ReferenceViewerError(ValueError):
    pass


def clean_label(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = value.translate(str.maketrans({"‘":"'", "’":"'", "“":'"', "”":'"'}))
    while len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"`":
        value = value[1:-1].strip()
    return re.sub(r"\s+", " ", value).casefold()


def canonical(value):
    return re.sub(r"^(?:the|a|an)\s+", "", clean_label(value))


def viewer_key(value):
    # Only matching keys change. Keep the scene label used to index the geometry.
    value = canonical(value).replace('"', '').replace('`', '')
    value = re.sub(r"(?<!\w)'|'(?!\w)", "", value)
    value = re.sub(r"\bwomen\b", "woman", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_objects(text):
    try:
        values = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        values = text.strip().removeprefix("[").removesuffix("]").split(",")
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise ValueError("VLM object list must contain strings")
    result, seen = [], set()
    for value in values:
        value = clean_label(value)
        if value and canonical(value) not in seen:
            result.append(value)
            seen.add(canonical(value))
    if not result:
        raise ValueError("VLM returned an empty object list")
    return result


def explicit_subject(question):
    # These are question templates, not answer labels. No ground truth is used.
    patterns = (
        r"\b(?:if i|if you|you are|i am) (?:stand|standing) at (.+?) position\b",
        r"\bfrom (.+?)(?:'s|’s) (?:perspective|viewpoint)\b",
        r"\bwhich object is (.+?) facing towards\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.I)
        if match:
            return re.sub(r"(?:'s|’s|['’])$", "", match.group(1).strip())
    return None


def resolve_viewer(response, question, scene, trace=None):
    matches = re.findall(r"\+\+(.*?)\+\+", response, flags=re.S)
    candidate = matches[-1] if matches else response.strip()
    candidate = re.sub(r"^\s*\[?perspective\]?\s*:?\s*", "", candidate, flags=re.I)
    candidate = clean_label(candidate)
    subject = explicit_subject(question)
    def matching(label, include_camera=False):
        key = viewer_key(label) if label is not None else ""
        return [obj for obj in scene if key and (include_camera or viewer_key(obj) != "camera")
                and viewer_key(obj) == key]

    expected_keys, proposed_keys = matching(subject), matching(candidate)
    expected = expected_keys[0] if len(expected_keys) == 1 else None
    proposed = proposed_keys[0] if len(proposed_keys) == 1 else None
    # An explicit, uniquely identified benchmark subject constrains the viewpoint.
    if expected is not None:
        viewer, strategy = expected, "question_subject" if proposed != expected else "vlm_and_question_agree"
    elif len(expected_keys) > 1:
        viewer, strategy = None, "ambiguous_question_subject"
    elif proposed is not None:
        viewer, strategy = proposed, "unique_vlm_scene_label"
    elif (subject is None or canonical(subject) == "camera") and canonical(candidate) == "camera" and re.search(r"\bcamera(?:'s)? (?:perspective|viewpoint)\b", question, re.I):
        camera_keys = matching("camera", include_camera=True)
        viewer, strategy = (camera_keys[0] if len(camera_keys) == 1 else None), "explicit_camera_question"
    else:
        viewer, strategy = None, "unresolved"
    audit = dict(raw_response=response, candidate=candidate, question_subject=subject,
                 available_objects=list(scene), subject_matches=expected_keys, candidate_matches=proposed_keys,
                 resolved_viewer=viewer, strategy=strategy)
    folder = trace or os.environ.get("APC_TRACE_DIR")
    if folder:
        (Path(folder)/"reference_viewer.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if viewer is None:
        raise ReferenceViewerError("No unique reference viewer in the detected scene; inspect reference_viewer.json")
    return viewer


def patch_reference_source(source):
    if "from apc_full_reference import resolve_viewer" in source:
        return source
    tree = ast.parse(source)
    methods = {node.name:node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    objects = methods["get_objects_of_interest"]
    assignments = [node for node in ast.walk(objects) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "objs_of_interest" for target in node.targets)]
    start, end = assignments[0].lineno-1, assignments[-1].end_lineno
    lines = source.splitlines(keepends=True)
    lines[start:end] = ["        from apc_full_reference import parse_objects\n",
                        "        objs_of_interest = parse_objects(match_objs)\n"]
    source = "".join(lines)
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "get_reference_viewer")
    first = next(node for node in method.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "pattern_ref_viewer" for target in node.targets))
    logger = next(node for node in method.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and getattr(node.value.func, "attr", "") == "info")
    lines = source.splitlines(keepends=True)
    lines[first.lineno-1:logger.end_lineno] = [
        "        from apc_full_reference import resolve_viewer\n",
        "        ref_viewer = resolve_viewer(response_ref_viewer, prompt, abstract_scene_dict_camera)\n",
        "        self.logger.info(f'[Perspective Change] Reference viewer: {ref_viewer}')\n",
    ]
    source = "".join(lines)
    ast.parse(source)
    return source


if __name__ == "__main__":
    import hashlib
    import sys
    target = Path(sys.argv[1])/"apc/apc_pipeline.py"
    before = target.read_text(encoding="utf-8")
    after = patch_reference_source(before)
    target.write_text(after, encoding="utf-8")
    Path(sys.argv[2]).write_text(json.dumps(dict(
        policy="unique_scene_key_with_quote_and_person_number_normalization_v2",
        before_sha=hashlib.sha256(before.encode()).hexdigest(),
        after_sha=hashlib.sha256(after.encode()).hexdigest()), indent=2), encoding="utf-8")
