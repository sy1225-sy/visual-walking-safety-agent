import os
import re
import json
import math
import traceback
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from gtts import gTTS


# =====================================================
# 1. 기본 설정
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("사용 디바이스:", device)

# 분석할 이미지 폴더
PHOTO_DIR = "/content/Photo"

# TTS 저장 폴더
TTS_DIR = "/content/tts_outputs"
os.makedirs(TTS_DIR, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

GRID_SIZE = 8

# 모델 이름
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
GEMMA_MODEL_NAME = "google/gemma-2-2b-it"
YOLO_MODEL_NAME = "yolo11n.pt"


# =====================================================
# 2. 모델 로드
# =====================================================

print("\nBLIP Caption 모델 불러오는 중...")
caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(device)
caption_model.eval()
print("BLIP Caption 모델 로드 완료")

print("\nBLIP VQA 모델 불러오는 중...")
vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL_NAME)
vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(device)
vqa_model.eval()
print("BLIP VQA 모델 로드 완료")

print("\nYOLO 모델 불러오는 중...")
yolo_model = YOLO(YOLO_MODEL_NAME)
print("YOLO 모델 로드 완료")

print("\nGemma 모델 불러오는 중...")
gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_NAME)
if device == "cuda":
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    gemma_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_NAME,
        torch_dtype=torch.float32,
    ).to(device)
gemma_model.eval()
print("Gemma 모델 로드 완료")


# =====================================================
# 3. 텍스트 정규화 / strict matching
# =====================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_phrase_strict(text: str, phrase: str) -> bool:
    """
    단어 경계 기반 매칭.
    car != cardboard
    bus != business
    """
    t = normalize_text(text)
    p = normalize_text(phrase)
    if not p:
        return False

    # 한글은 공백 경계가 불안정하므로 그대로 포함 허용
    if re.search(r"[가-힣]", p):
        return p in t

    pattern = r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])"
    return re.search(pattern, t) is not None


def contains_any_strict(text: str, words: List[str]) -> bool:
    return any(contains_phrase_strict(text, word) for word in words)


def join_evidence_text(caption: str, vqa_data: Dict[str, str]) -> str:
    parts = [caption]
    for k, v in vqa_data.items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


# =====================================================
# 4. 한국어 변환 / 조사 처리
# =====================================================

LABEL_KO_MAP = {
    "person": "사람",
    "bicycle": "자전거",
    "bike": "자전거",
    "car": "차량",
    "truck": "트럭",
    "bus": "버스",
    "motorcycle": "오토바이",
    "chair": "의자",
    "bench": "벤치",
    "table": "책상",
    "desk": "책상",
    "box": "상자",
    "cardboard box": "상자",
    "trash can": "쓰레기통",
    "garbage can": "쓰레기통",
    "bin": "쓰레기통",
    "sink": "세면대",
    "toilet": "변기",
    "urinal": "변기",
    "door": "문",
    "elevator": "엘리베이터",
    "elevator door": "엘리베이터 문",
    "stairs": "계단",
    "stair": "계단",
    "staircase": "계단",
    "refrigerator": "냉장고/정수기 후보",
    "water dispenser": "정수기",
    "water cooler": "정수기",
}


def translate_label(label: str) -> str:
    key = normalize_text(label)
    if key in LABEL_KO_MAP:
        return LABEL_KO_MAP[key]
    return label


def clean_object_name(name: str) -> str:
    name = normalize_text(name)
    # 긴 문구 우선 치환
    replacements = [
        ("cardboard box", "상자"),
        ("trash can", "쓰레기통"),
        ("garbage can", "쓰레기통"),
        ("water dispenser", "정수기"),
        ("water cooler", "정수기"),
        ("elevator door", "엘리베이터 문"),
        ("glass door", "문"),
        ("stall door", "문"),
        ("glass block wall", "벽"),
    ]
    for eng, kor in replacements:
        if contains_phrase_strict(name, eng):
            return kor

    for eng, kor in LABEL_KO_MAP.items():
        if contains_phrase_strict(name, eng):
            # refrigerator는 정수기 보정 전에는 후보명 그대로 둠
            if eng == "refrigerator":
                return "냉장고/정수기 후보"
            return kor
    return name


def has_final_consonant(korean_word: str) -> bool:
    if not korean_word:
        return False
    ch = korean_word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def josa_i_ga(word: str) -> str:
    if not re.search(r"[가-힣]", word):
        return "이"
    return "이" if has_final_consonant(word) else "가"


# =====================================================
# 5. 이미지 로드 / 폴더 이미지 검색
# =====================================================

def load_image(image_path: str) -> Image.Image:
    """
    스마트폰 EXIF 회전만 반영한다.
    좌우 반전은 하지 않는다.
    사진 왼쪽 = 사용자 왼쪽, 사진 오른쪽 = 사용자 오른쪽.
    """
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def get_image_files_from_folder(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {folder_path}")

    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file_name))

    image_paths.sort()
    return image_paths


# =====================================================
# 6. BLIP Caption / VQA
# =====================================================

@torch.no_grad()
def generate_caption(image: Image.Image) -> str:
    inputs = caption_processor(image, return_tensors="pt").to(device)
    outputs = caption_model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=5,
    )
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
    return caption.strip()


@torch.no_grad()
def ask_vqa(image: Image.Image, question: str, max_new_tokens: int = 20) -> str:
    inputs = vqa_processor(image, question, return_tensors="pt").to(device)
    outputs = vqa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = vqa_processor.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


def run_vqa_analysis(image: Image.Image) -> Dict[str, str]:
    """
    yes/no만 믿지 않기 위해 open question을 많이 사용한다.
    최종 후보 생성에서는 yes/no 단독 값에 큰 가중치를 주지 않는다.
    """
    questions = {
        "scene_type": "What kind of place is this?",
        "main_object": "What is the main object in the image?",
        "walkway_obstacle": "What object is blocking the walking path?",
        "nearest_object": "What object is closest to the camera?",
        "stairs_present": "Are there stairs in the image? answer yes or no.",
        "stairs_direction": "If there are stairs, are they going up or down?",
        "door_present": "Is there a door in the image? answer yes or no.",
        "sink_present": "Is there a sink in the image? answer yes or no.",
        "toilet_present": "Is there a toilet in the image? answer yes or no.",
        "water_dispenser_present": "Is there a water dispenser in the image? answer yes or no.",
        "vehicle_present": "Is there a car, truck, bus, or motorcycle in the image? answer yes or no.",
        "bicycle_present": "Is there a bicycle in the image? answer yes or no.",
        "person_present": "Is there a person in the image? answer yes or no.",
        "hand_covered": "Is the camera covered by a hand? answer yes or no.",
        "blurry": "Is the image blurry? answer yes or no.",
        "safe_path": "Which side looks safer for walking, left or right?",
    }

    answers = {}
    for key, question in questions.items():
        try:
            answers[key] = ask_vqa(image, question)
        except Exception as e:
            answers[key] = f"error: {e}"
    return answers


# =====================================================
# 7. 8x8 그리드 분석
# =====================================================

def get_grid_overlap_scores(
    box: List[float],
    image_width: int,
    image_height: int,
    grid_size: int = GRID_SIZE,
) -> Tuple[List[int], Dict[str, float]]:
    """
    객체 박스가 8x8 그리드 중 어디를 얼마만큼 차지하는지 계산한다.
    반환:
    - cells: 겹친 셀 번호 리스트, 1~64
    - scores: left/center/right 및 depth 관련 점수
    """
    x1, y1, x2, y2 = box
    cell_w = image_width / grid_size
    cell_h = image_height / grid_size

    cells = []
    left_score = 0.0
    center_score = 0.0
    right_score = 0.0
    bottom_score = 0.0
    total_overlap = 0.0

    for row in range(grid_size):
        for col in range(grid_size):
            cx1 = col * cell_w
            cy1 = row * cell_h
            cx2 = cx1 + cell_w
            cy2 = cy1 + cell_h

            ox = max(0.0, min(x2, cx2) - max(x1, cx1))
            oy = max(0.0, min(y2, cy2) - max(y1, cy1))
            area = ox * oy
            if area <= 0:
                continue

            cell_num = row * grid_size + col + 1
            cells.append(cell_num)
            total_overlap += area

            # 아래쪽일수록 가까운 장애물로 가중치
            # row: 0~7 → 0.6~2.0
            row_weight = 0.6 + (row / (grid_size - 1)) * 1.4

            # 보행에서는 하단 영역이 중요하므로 추가 가중치
            if row >= 5:
                bottom_score += area * row_weight

            # 좌/정면/우 영역: 8칸 기준
            # 0,1 = 왼쪽 / 2,3,4,5 = 정면 / 6,7 = 오른쪽
            # 단, 정면좌/정면우도 좌우 회피 계산에는 일부 반영
            weighted = area * row_weight
            if col <= 1:
                left_score += weighted * 1.0
            elif col in [2, 3]:
                left_score += weighted * 0.45
                center_score += weighted * 1.0
            elif col in [4, 5]:
                right_score += weighted * 0.45
                center_score += weighted * 1.0
            else:
                right_score += weighted * 1.0

    scores = {
        "left": round(left_score, 3),
        "center": round(center_score, 3),
        "right": round(right_score, 3),
        "bottom": round(bottom_score, 3),
        "total": round(total_overlap, 3),
    }
    return cells, scores


def estimate_position_from_grid(scores: Dict[str, float]) -> str:
    left = scores.get("left", 0.0)
    center = scores.get("center", 0.0)
    right = scores.get("right", 0.0)

    # 한쪽이 훨씬 크면 좌/우로 판단
    if right > left * 1.25 and right >= center * 0.45:
        return "오른쪽"
    if left > right * 1.25 and left >= center * 0.45:
        return "왼쪽"
    return "정면"


def estimate_depth_from_box(box: List[float], image_height: int) -> str:
    _, _, _, y2 = box
    bottom_ratio = y2 / max(image_height, 1)
    if bottom_ratio >= 0.72:
        return "가까운"
    elif bottom_ratio >= 0.45:
        return "중간 거리의"
    return "먼"


def is_path_intrusion(scores: Dict[str, float], depth: str) -> str:
    center = scores.get("center", 0.0)
    bottom = scores.get("bottom", 0.0)
    total = scores.get("total", 0.0)
    if total <= 0:
        return "알 수 없음"
    # 중심 영역 또는 하단 영역을 많이 침범하면 보행 경로 침범
    if depth == "가까운" and (center / total > 0.15 or bottom / total > 0.10):
        return "보행 경로 침범"
    if center / total > 0.35:
        return "보행 경로 침범"
    return "보행 경로 밖"


# =====================================================
# 8. YOLO 객체 탐지
# =====================================================

YOLO_IGNORE_IF_ALONE = {
    "clock",
    "cup",
    "bottle",
    "cell phone",
    "remote",
    "traffic light",
    "stop sign",
    "tv",
}


def detect_objects_with_yolo(image: Image.Image) -> List[Dict[str, Any]]:
    image_np = np.array(image)
    image_width = image.width
    image_height = image.height

    results = yolo_model.predict(
        source=image_np,
        conf=0.20,
        verbose=False,
    )

    detections = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            cells, grid_scores = get_grid_overlap_scores(
                bbox, image_width, image_height, GRID_SIZE
            )
            depth = estimate_depth_from_box(bbox, image_height)
            position = estimate_position_from_grid(grid_scores)
            path_status = is_path_intrusion(grid_scores, depth)

            label_ko = translate_label(label)

            detections.append({
                "label": label,
                "label_ko": label_ko,
                "confidence": round(conf, 3),
                "box": [round(v, 1) for v in bbox],
                "grid_cells": cells,
                "grid_scores": grid_scores,
                "position": position,
                "depth": depth,
                "path_status": path_status,
                "ignore_if_alone": normalize_text(label) in YOLO_IGNORE_IF_ALONE,
            })

    return detections


# =====================================================
# 9. 위험 후보 생성 / 교차검증
# =====================================================

# 객체 위험 그룹
STOP_OBJECTS = {"계단", "차량", "트럭", "버스", "오토바이"}
SLOW_OBJECTS = {"사람", "자전거", "의자", "책상", "상자", "쓰레기통", "정수기", "세면대", "변기", "냉장고/정수기 후보"}
CAUTION_OBJECTS = {"문", "엘리베이터", "엘리베이터 문", "벽", "복도", "화장실 입구"}

TEXT_CATEGORIES = {
    "계단": ["stairs", "stair", "staircase", "steps", "계단"],
    "차량": ["car", "vehicle", "sedan", "suv", "automobile", "차량", "자동차"],
    "트럭": ["truck", "lorry", "트럭"],
    "버스": ["bus", "버스"],
    "오토바이": ["motorcycle", "motorbike", "오토바이"],
    "자전거": ["bicycle", "bike", "cycle", "자전거"],
    "사람": ["person", "man", "woman", "people", "human", "사람"],
    "의자": ["chair", "chairs", "의자"],
    "책상": ["table", "desk", "tables", "desks", "책상"],
    "상자": ["box", "boxes", "cardboard box", "carton", "상자"],
    "쓰레기통": ["trash can", "garbage can", "bin", "waste bin", "trash", "쓰레기통"],
    "정수기": ["water dispenser", "water cooler", "dispenser", "정수기"],
    "세면대": ["sink", "washbasin", "basin", "세면대"],
    "변기": ["toilet", "urinal", "변기"],
    "문": ["door", "glass door", "stall door", "문"],
    "엘리베이터 문": ["elevator door", "elevator", "lift", "엘리베이터"],
    "벽": ["wall", "glass block wall", "벽"],
}


def candidate_base_score(label_ko: str) -> float:
    if label_ko in STOP_OBJECTS:
        return 100.0
    if label_ko in SLOW_OBJECTS:
        return 70.0
    if label_ko in CAUTION_OBJECTS:
        return 35.0
    return 25.0


def risk_level_for_candidate(candidate: Dict[str, Any]) -> str:
    obj = candidate.get("object", "")
    depth = candidate.get("depth", "")
    path = candidate.get("path_status", "")

    if obj in STOP_OBJECTS:
        return "상"

    if obj in SLOW_OBJECTS:
        # 가까운 보행 장애물은 중
        if depth in ["가까운", "중간 거리의"] or path == "보행 경로 침범":
            return "중"
        return "하"

    if obj in CAUTION_OBJECTS:
        return "하"

    return "하"


def action_for_risk(risk_level: str) -> str:
    return {
        "상": "정지하세요",
        "중": "서행하세요",
        "하": "주의하세요",
        "판단불가": "다시 촬영하세요",
    }.get(risk_level, "주의하세요")


def make_yolo_candidates(detections: List[Dict[str, Any]], caption: str, vqa_data: Dict[str, str]) -> List[Dict[str, Any]]:
    candidates = []
    evidence_text = join_evidence_text(caption, vqa_data)

    for det in detections:
        raw_label = normalize_text(det["label"])
        label_ko = det["label_ko"]

        # 정수기 보정: YOLO refrigerator + VQA/caption water dispenser 단서
        if raw_label == "refrigerator":
            if contains_any_strict(evidence_text, ["water dispenser", "water cooler", "dispenser", "white machine"]):
                label_ko = "정수기"
            else:
                label_ko = "냉장고/정수기 후보"

        # YOLO 단독 오탐 label은 텍스트 근거가 없으면 버림
        if det.get("ignore_if_alone"):
            if not contains_any_strict(evidence_text, [raw_label]):
                continue

        score = candidate_base_score(label_ko)
        score += float(det.get("confidence", 0.0)) * 50.0

        if det.get("depth") == "가까운":
            score += 25.0
        elif det.get("depth") == "중간 거리의":
            score += 10.0

        if det.get("path_status") == "보행 경로 침범":
            score += 20.0

        gs = det.get("grid_scores", {})
        # 하단/정면 침범이 클수록 점수 증가
        total = max(gs.get("total", 0.0), 1.0)
        score += min(gs.get("bottom", 0.0) / total * 10.0, 15.0)
        score += min(gs.get("center", 0.0) / total * 10.0, 15.0)

        candidates.append({
            "object": label_ko,
            "source": "YOLO",
            "position": det.get("position", "정면"),
            "depth": det.get("depth", "중간 거리의"),
            "path_status": det.get("path_status", "알 수 없음"),
            "confidence": det.get("confidence", 0.0),
            "box": det.get("box", []),
            "grid_cells": det.get("grid_cells", []),
            "grid_scores": det.get("grid_scores", {}),
            "score": round(score, 2),
            "evidence": f"YOLO label={det.get('label')}, conf={det.get('confidence')}",
        })

    return candidates


def text_supports_object(caption: str, vqa_data: Dict[str, str], obj: str) -> bool:
    """
    텍스트 기반 후보 생성.
    yes/no 답변은 단독으로 쓰지 않고, caption/main_object/walkway_obstacle/nearest_object 같은 open answer 중심.
    """
    open_keys = ["scene_type", "main_object", "walkway_obstacle", "nearest_object"]
    open_text = caption + "\n" + "\n".join(str(vqa_data.get(k, "")) for k in open_keys)
    words = TEXT_CATEGORIES.get(obj, [])
    return contains_any_strict(open_text, words)


def infer_text_candidate_position_depth(obj: str, caption: str, vqa_data: Dict[str, str]) -> Tuple[str, str, str]:
    """
    YOLO가 없을 때 텍스트 근거만으로 기본 위치/거리감 설정.
    좌우는 알기 어려우므로 정면, 거리감은 nearest_object와 일치하면 가까운.
    """
    nearest = vqa_data.get("nearest_object", "")
    if contains_any_strict(nearest, TEXT_CATEGORIES.get(obj, [])):
        depth = "가까운"
        path_status = "보행 경로 침범"
    else:
        depth = "중간 거리의"
        path_status = "알 수 없음"

    return "정면", depth, path_status


def make_text_candidates(caption: str, vqa_data: Dict[str, str], yolo_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    existing_objects = {c["object"] for c in yolo_candidates}

    # 텍스트 후보는 open answer + caption에 근거가 있을 때만 생성한다.
    # vehicle_present yes 같은 yes/no 단독으로 차량 생성 금지.
    for obj in [
        "계단", "차량", "트럭", "버스", "오토바이", "자전거", "사람",
        "상자", "쓰레기통", "정수기", "세면대", "변기", "의자", "책상", "문", "엘리베이터 문", "벽"
    ]:
        if obj in existing_objects:
            continue

        if not text_supports_object(caption, vqa_data, obj):
            continue

        # car/cardboard 방지: 차량은 더 엄격히 확인
        if obj == "차량":
            # caption/open answer에 car/vehicle/sedan/suv/automobile 정확 단어가 있어야 함
            if not text_supports_object(caption, vqa_data, "차량"):
                continue
        if obj == "버스":
            # bus/business 방지
            if not text_supports_object(caption, vqa_data, "버스"):
                continue

        position, depth, path_status = infer_text_candidate_position_depth(obj, caption, vqa_data)
        score = candidate_base_score(obj)
        if depth == "가까운":
            score += 20.0
        if path_status == "보행 경로 침범":
            score += 15.0

        # 문/벽/엘리베이터는 단독으로 너무 강하게 올라가지 않도록 제한
        if obj in CAUTION_OBJECTS:
            score = min(score, 45.0)

        candidates.append({
            "object": obj,
            "source": "BLIP/VQA",
            "position": position,
            "depth": depth,
            "path_status": path_status,
            "confidence": None,
            "box": [],
            "grid_cells": [],
            "grid_scores": {"left": 0.0, "center": 0.0, "right": 0.0, "bottom": 0.0, "total": 0.0},
            "score": round(score, 2),
            "evidence": f"Caption/Open VQA에서 {obj} 단서 확인",
        })

    return candidates


def merge_similar_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    같은 object 후보가 여러 개면 높은 점수 위주로 병합.
    단 YOLO 위치 정보가 있는 후보를 우선한다.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        obj = c["object"]
        if obj not in merged:
            merged[obj] = c.copy()
            continue

        old = merged[obj]
        # 점수는 합산하되 너무 커지지 않게 일부만 반영
        new_score = max(old["score"], c["score"]) + min(old["score"], c["score"]) * 0.15

        # YOLO 후보가 있으면 위치/box는 YOLO 우선
        if old["source"] != "YOLO" and c["source"] == "YOLO":
            base = c.copy()
        else:
            base = old.copy()

        base["score"] = round(new_score, 2)
        base["evidence"] = old.get("evidence", "") + " | " + c.get("evidence", "")
        merged[obj] = base

    result = list(merged.values())
    result.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return result


def build_candidates(caption: str, vqa_data: Dict[str, str], detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    yolo_candidates = make_yolo_candidates(detections, caption, vqa_data)
    text_candidates = make_text_candidates(caption, vqa_data, yolo_candidates)
    candidates = merge_similar_candidates(yolo_candidates + text_candidates)

    # 명백한 오류 후보 제거: 상자(cardboard)가 vehicle/car로 오인된 경우 방지
    # strict matching으로 대부분 해결되지만, 후보 증거에도 cardboard가 있고 차량 근거가 약하면 제거
    filtered = []
    for c in candidates:
        obj = c.get("object")
        ev = normalize_text(c.get("evidence", ""))
        cap = normalize_text(caption)
        if obj == "차량" and contains_phrase_strict(cap, "cardboard box"):
            # Caption에 cardboard box가 있고 정확한 car/vehicle 단어가 없으면 차량 제거
            if not contains_any_strict(cap, ["car", "vehicle", "sedan", "suv", "automobile"]):
                continue
        filtered.append(c)

    filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return filtered


# =====================================================
# 10. Gemma 후보 검토
# =====================================================

def call_gemma(prompt: str, max_new_tokens: int = 180) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = gemma_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = gemma_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(gemma_model.device)

    # do_sample=False이면 temperature 경고가 나지 않도록 temperature 전달 안 함
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=gemma_tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return gemma_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json_output(text: str) -> Dict[str, Any]:
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for candidate in reversed(fenced):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    blocks = re.findall(r"\{.*?\}", text, re.DOTALL)
    for candidate in reversed(blocks):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {"raw": text}


def select_candidate_with_gemma(candidates: List[Dict[str, Any]], caption: str, vqa_data: Dict[str, str]) -> Dict[str, Any]:
    if not candidates:
        return {}

    # 후보가 1개면 바로 사용
    if len(candidates) == 1:
        return candidates[0]

    candidate_lines = []
    for idx, c in enumerate(candidates[:8], start=1):
        candidate_lines.append(
            f"{idx}. object={c['object']}, source={c['source']}, "
            f"position={c['position']}, depth={c['depth']}, "
            f"path={c['path_status']}, score={c['score']}, evidence={c['evidence']}"
        )

    prompt = f"""
당신은 시각장애인 보행 안전 에이전트의 후보 검토 모듈입니다.

중요 규칙:
- 아래 후보 목록에 없는 물체를 새로 만들지 마세요.
- 후보 ID만 선택하세요.
- Caption과 VQA가 말해도 후보 근거가 약하면 선택하지 마세요.
- 실제 보행 경로에 가까운 물체, 화면 하단에 가까운 물체를 우선하세요.
- 차량/계단은 실제 근거가 있을 때만 선택하세요.
- car가 cardboard box 안에 포함된 경우 차량이 아니라 상자입니다.

[Caption]
{caption}

[VQA]
{json.dumps(vqa_data, ensure_ascii=False)}

[후보 목록]
{chr(10).join(candidate_lines)}

반드시 JSON만 출력하세요.
{{
  "선택후보ID": 1,
  "선택이유": "짧게 설명"
}}
"""

    try:
        raw = call_gemma(prompt)
        print("\n  [Gemma 원본 출력]")
        print(raw)
        parsed = parse_json_output(raw)
        selected_id = int(parsed.get("선택후보ID", 1))
        if 1 <= selected_id <= min(len(candidates), 8):
            selected = candidates[selected_id - 1]

            # Gemma 선택 이유가 다른 물체를 말하는 모순 보정
            reason = normalize_text(parsed.get("선택이유", ""))
            for idx, c in enumerate(candidates[:8], start=1):
                obj = c["object"]
                if obj != selected["object"] and contains_phrase_strict(reason, obj):
                    return c

            return selected
    except Exception as e:
        print("  Gemma 후보 선택 실패:", e)

    # 실패 시 점수 1위 사용
    return candidates[0]


# =====================================================
# 11. 회피 방향 계산 - 8x8 grid 기반
# =====================================================

def object_weight(obj: str) -> float:
    if obj in STOP_OBJECTS:
        return 2.5
    if obj in SLOW_OBJECTS:
        return 1.8
    if obj in CAUTION_OBJECTS:
        return 0.8
    return 1.0


def choose_avoid_direction_by_grid(selected: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> str:
    obj = selected.get("object", "")
    risk = risk_level_for_candidate(selected)

    # 계단/차량류는 회피 방향보다 정지 후 확인 우선
    if obj in STOP_OBJECTS:
        return "정지 후 확인"

    # YOLO box가 없는 경우 좌우 판단 불가
    if not selected.get("grid_scores") or selected.get("grid_scores", {}).get("total", 0.0) <= 0:
        pos = selected.get("position", "정면")
        if pos == "왼쪽":
            return "오른쪽"
        if pos == "오른쪽":
            return "왼쪽"
        return "정지 후 좌우 확인" if risk in ["상", "중"] else "천천히 이동"

    # 전체 후보 기준 좌/우 위험 점수 합산
    left_risk = 0.0
    right_risk = 0.0
    center_risk = 0.0

    for c in all_candidates:
        gs = c.get("grid_scores", {}) or {}
        total = gs.get("total", 0.0)
        if total <= 0:
            continue

        w = object_weight(c.get("object", ""))
        base = max(c.get("score", 1.0), 1.0) / 50.0 * w

        left_risk += (gs.get("left", 0.0) / total) * base
        right_risk += (gs.get("right", 0.0) / total) * base
        center_risk += (gs.get("center", 0.0) / total) * base

        if c.get("depth") == "가까운":
            left_risk *= 1.05
            right_risk *= 1.05
            center_risk *= 1.05

    selected_gs = selected.get("grid_scores", {})
    selected_total = max(selected_gs.get("total", 0.0), 1.0)
    selected_center_ratio = selected_gs.get("center", 0.0) / selected_total
    selected_right_ratio = selected_gs.get("right", 0.0) / selected_total
    selected_left_ratio = selected_gs.get("left", 0.0) / selected_total

    # 선택 객체 자체가 오른쪽을 더 많이 막으면 왼쪽 회피
    if selected_right_ratio > selected_left_ratio * 1.15:
        return "왼쪽"
    if selected_left_ratio > selected_right_ratio * 1.15:
        return "오른쪽"

    # 정면에 걸친 경우 전체 좌우 위험이 낮은 쪽 선택
    if selected_center_ratio > 0.20 or selected.get("position") == "정면":
        if right_risk > left_risk * 1.15:
            return "왼쪽"
        if left_risk > right_risk * 1.15:
            return "오른쪽"
        return "정지 후 좌우 확인"

    pos = selected.get("position", "정면")
    if pos == "왼쪽":
        return "오른쪽"
    if pos == "오른쪽":
        return "왼쪽"
    return "정지 후 좌우 확인"


# =====================================================
# 12. 최종 문장 생성
# =====================================================

def make_situation(candidate: Dict[str, Any]) -> str:
    obj = candidate.get("object", "장애물")
    pos = candidate.get("position", "정면")
    depth = candidate.get("depth", "중간 거리의")

    if obj == "계단":
        return "정면에 계단이 있습니다."

    return f"{pos} {depth} 위치에 {obj}{josa_i_ga(obj)} 있습니다."


def make_risk_reason(candidate: Dict[str, Any]) -> str:
    obj = candidate.get("object", "장애물")
    depth = candidate.get("depth", "")
    path = candidate.get("path_status", "")

    if obj == "계단":
        return "계단은 발을 헛디디거나 넘어질 위험이 있어 먼저 멈춰 확인해야 합니다."
    if obj in {"차량", "트럭", "버스", "오토바이"}:
        return f"{obj}{josa_i_ga(obj)} 가까이 있거나 움직일 수 있어 충돌 위험이 있습니다."
    if obj in SLOW_OBJECTS:
        if path == "보행 경로 침범" or depth == "가까운":
            return f"{obj}{josa_i_ga(obj)} 보행 경로와 가까워 부딪힐 수 있습니다."
        return f"{obj}{josa_i_ga(obj)} 주변에 있어 주의가 필요합니다."
    if obj in CAUTION_OBJECTS:
        return f"{obj}{josa_i_ga(obj)} 주변에 있어 이동 시 주의가 필요합니다."
    return "보행 중 주의가 필요한 요소가 있습니다."


def make_guidance(risk_level: str, situation: str, avoid_direction: str, candidate: Dict[str, Any]) -> str:
    action = action_for_risk(risk_level)
    obj = candidate.get("object", "")

    if risk_level == "판단불가":
        return "다시 촬영하세요. 장면을 정확히 판단하기 어렵습니다. 멈춘 뒤 다시 촬영해 주세요."

    if risk_level == "상":
        if obj == "계단":
            return f"{action}. {situation} 난간과 발밑을 확인한 뒤 천천히 이동하세요."
        return f"{action}. {situation} 주변을 확인한 뒤 이동하세요."

    if risk_level == "중":
        if avoid_direction == "왼쪽":
            return f"{action}. {situation} 왼쪽 공간을 확인한 뒤 천천히 피해가세요."
        if avoid_direction == "오른쪽":
            return f"{action}. {situation} 오른쪽 공간을 확인한 뒤 천천히 피해가세요."
        if avoid_direction == "정지 후 좌우 확인":
            return f"{action}. {situation} 멈춘 뒤 좌우 공간을 확인하고 이동하세요."
        return f"{action}. {situation} 주변을 확인하며 천천히 이동하세요."

    return f"{action}. {situation} 주변을 확인하며 이동하세요."


def generate_final_result(selected: Dict[str, Any], all_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not selected:
        return {
            "위험도": "판단불가",
            "위험 행동": "다시 촬영하세요",
            "위험 요소": "판단불가",
            "위치": "알 수 없음",
            "거리감": "알 수 없음",
            "위험 이유": "뚜렷한 위험 후보를 찾지 못했습니다.",
            "회피 방향": "다시 촬영",
            "상황 설명": "장면 정보가 부족합니다.",
            "안내 멘트": "다시 촬영하세요. 장면을 정확히 판단하기 어렵습니다. 멈춘 뒤 다시 촬영해 주세요.",
            "판단불가": True,
        }

    risk_level = risk_level_for_candidate(selected)
    action = action_for_risk(risk_level)
    avoid = choose_avoid_direction_by_grid(selected, all_candidates)
    situation = make_situation(selected)
    reason = make_risk_reason(selected)
    guidance = make_guidance(risk_level, situation, avoid, selected)

    return {
        "위험도": risk_level,
        "위험 행동": action,
        "위험 요소": selected.get("object", "장애물"),
        "위치": selected.get("position", "정면"),
        "거리감": selected.get("depth", "중간 거리의"),
        "위험 이유": reason,
        "회피 방향": avoid,
        "상황 설명": situation,
        "안내 멘트": guidance,
        "판단불가": False,
    }


# =====================================================
# 13. TTS 저장
# =====================================================

def safe_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", name)


def save_tts(guidance_sentence: str, file_base_name: str) -> str:
    if not guidance_sentence:
        guidance_sentence = "주의하세요. 주변을 확인하며 이동하세요."

    tts_filename = f"{safe_filename(file_base_name)}_warning.mp3"
    tts_path = os.path.join(TTS_DIR, tts_filename)

    print(f"\n  [TTS 변환 중] {guidance_sentence}")
    tts = gTTS(text=guidance_sentence, lang="ko", slow=False)
    tts.save(tts_path)
    print(f"  [TTS 완료] 저장 경로: {tts_path}")
    return tts_path


# =====================================================
# 14. 이미지 1장 분석
# =====================================================

def visual_warning_agent(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    image_name = os.path.basename(image_path)

    print("\n" + "=" * 80)
    print(f"분석 이미지: {image_name}")
    print("=" * 80)

    image = load_image(image_path)

    print("\n  [1단계] BLIP Caption 생성 중...")
    caption = generate_caption(image)
    print(f"  Caption: {caption}")

    print("\n  [2단계] BLIP VQA 분석 중...")
    vqa_data = run_vqa_analysis(image)
    for k, v in vqa_data.items():
        print(f"  - {k}: {v}")

    print("\n  [3단계] YOLO 객체 탐지 중...")
    detections = detect_objects_with_yolo(image)
    if detections:
        for d in detections:
            gs = d.get("grid_scores", {})
            print(
                f"  - {d.get('label_ko', d['label'])} / 원래 label={d['label']} "
                f"/ conf={d['confidence']} / 위치={d['position']} / 거리감={d['depth']} "
                f"/ 보행경로={d['path_status']} / box={d['box']} "
                f"/ grid={d['grid_cells']} / 좌={gs.get('left', 0)} 정면={gs.get('center', 0)} 우={gs.get('right', 0)}"
            )
    else:
        print("  탐지된 YOLO 객체 없음")

    print("\n  [4단계] 후보 교차검증 중...")
    candidates = build_candidates(caption, vqa_data, detections)
    if candidates:
        for idx, c in enumerate(candidates, start=1):
            gs = c.get("grid_scores", {}) or {}
            print(
                f"  후보 {idx}: {c['object']} / source={c['source']} / 위치={c['position']} "
                f"/ 거리감={c['depth']} / score={c['score']} / grid={c.get('grid_cells', [])} "
                f"/ 좌={gs.get('left', 0)} 정면={gs.get('center', 0)} 우={gs.get('right', 0)} "
                f"/ evidence={c['evidence']}"
            )
    else:
        print("  검증된 후보 없음")

    print("\n  [5단계] Gemma 후보 검토 중...")
    selected = select_candidate_with_gemma(candidates, caption, vqa_data)

    final = generate_final_result(selected, candidates)

    print("\n" + "-" * 80)
    print("[최종 결과]")
    print(f"  위험도: {final['위험도']}")
    print(f"  위험 행동: {final['위험 행동']}")
    print(f"  위험 요소: {final['위험 요소']}")
    print(f"  위치: {final['위치']}")
    print(f"  거리감: {final['거리감']}")
    print(f"  위험 이유: {final['위험 이유']}")
    print(f"  회피 방향: {final['회피 방향']}")
    print(f"  상황 설명: {final['상황 설명']}")
    print(f"  안내 멘트: {final['안내 멘트']}")
    print("-" * 80)

    return {
        "image": image_name,
        "caption": caption,
        "vqa": vqa_data,
        "yolo_detections": detections,
        "candidates": candidates,
        "selected_candidate": selected,
        **final,
    }


# =====================================================
# 15. 전체 실행
# =====================================================

def main():
    results = []

    try:
        image_paths = get_image_files_from_folder(PHOTO_DIR)
    except FileNotFoundError as e:
        print("\n폴더 없음:", e)
        print("다운로드 폴더 안에 Photo 폴더를 만든 뒤 이미지를 넣어 주세요.")
        return

    if not image_paths:
        print(f"\n분석할 이미지가 없습니다: {PHOTO_DIR}")
        print("지원 확장자:", SUPPORTED_IMAGE_EXTENSIONS)
        return

    print("\n분석할 이미지 폴더:", PHOTO_DIR)
    print(f"총 {len(image_paths)}장의 이미지를 분석합니다.")

    for image_path in image_paths:
        try:
            result = visual_warning_agent(image_path)
            results.append(result)

            file_base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_tts(result["안내 멘트"], file_base_name)

        except Exception as e:
            print("\n오류 발생:", e)
            traceback.print_exc()

    print("\n\n" + "#" * 80)
    print("전체 분석 결과 요약")
    print("#" * 80)

    for r in results:
        판단불가_표시 = "  ※ 판단불가 (추가 사진 필요)" if r.get("판단불가") else ""
        print(f"\n이미지   : {r['image']}")
        print(f"위험도   : {r['위험도']}{판단불가_표시}")
        print(f"위험 행동: {r['위험 행동']}")
        print(f"위험 요소: {r['위험 요소']}")
        print(f"위치     : {r['위치']}")
        print(f"거리감   : {r['거리감']}")
        print(f"위험 이유: {r['위험 이유']}")
        print(f"회피 방향: {r['회피 방향']}")
        print(f"상황 설명: {r['상황 설명']}")
        print(f"안내 멘트: {r['안내 멘트']}")


if __name__ == "__main__":
    main()
