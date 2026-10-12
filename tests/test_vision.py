import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from trutina.api import RequestError, parse_request
from trutina.data import load_examples
from trutina.prompt import encode_question
from trutina.schema import Question
from trutina.vision import (
    IMAGE_PAD,
    VISION_END,
    VISION_START,
    ImageRef,
    image_grid,
    n_image_tokens,
    pixel_patches,
    smart_resize,
)


class CharTok:
    def encode(self, text):
        return [ord(c) for c in text]


def _png(w, h, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _uri(w, h, seed=0) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(w, h, seed)).decode()


@pytest.mark.parametrize("size", [(300, 200), (1920, 1080), (40, 900), (4000, 3000)])
def test_patches_match_hf_processor(size):
    tf = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    proc = tf.Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                       image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                       size={"shortest_edge": 65536, "longest_edge": 1024 * 32 * 32})
    img = Image.open(io.BytesIO(_png(*size))).convert("RGB")
    ref = proc(images=[img], return_tensors="np")
    grid = image_grid(img.size)
    assert tuple(ref["image_grid_thw"][0]) == grid
    np.testing.assert_allclose(pixel_patches(img, grid), ref["pixel_values"], atol=1e-5)


def test_token_budget_is_capped():
    assert n_image_tokens(image_grid((4000, 3000))) <= 1024
    h, w = smart_resize(10, 10)
    assert h * w >= 65536 and h % 32 == 0 and w % 32 == 0


def test_encode_question_with_image():
    q = Question("q", "Is the button red?", kind="noul")
    state = ("Screen:\n", ImageRef(_uri(64, 96)), "\nend")
    enc = encode_question(CharTok(), state, q)
    grid = image_grid((64, 96))
    n = n_image_tokens(grid)
    s = enc.tokens.index(VISION_START)
    assert enc.tokens[s + 1 : s + 1 + n] == [IMAGE_PAD] * n and enc.tokens[s + 1 + n] == VISION_END
    assert enc.grids == (grid,) and len(enc.images) == 1
    assert enc.tokens[enc.answer_pos] == ord(">")
    # text-only states tokenise exactly as before (one segment)
    text = encode_question(CharTok(), "Screen:\nend", q)
    assert text.images == () and VISION_START not in text.tokens


def test_api_accepts_parts_and_rejects_paths():
    q = {"q": {"type": "noul", "instructions": "ok?"}}
    req = parse_request({"state": [{"type": "input_text", "text": "a"}, {"type": "input_image", "image_url": _uri(32, 32)}],
                         "questions": q})
    assert isinstance(req.state[1], ImageRef)
    with pytest.raises(RequestError):
        parse_request({"state": [{"type": "input_image", "image_url": "/etc/passwd"}], "questions": q})
    with pytest.raises(RequestError):
        parse_request({"state": [], "questions": q})


def test_training_data_resolves_relative_images(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(_png(50, 60))
    row = {"state": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "img/a.png"}],
           "questions": {"q": {"type": "noul", "instructions": "ok?"}}, "labels": {"q": True}}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    (ex,) = load_examples(tmp_path / "train.jsonl")
    with ex.state[1].open() as img:
        assert img.size == (50, 60)
