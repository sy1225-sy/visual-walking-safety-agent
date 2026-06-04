# =====================================================
# 박소연 담당 코드
# 역할: 기본 설정, 모델 로드, 이미지 입력 및 BLIP 장면 이해 모듈
# 설명: 이미지 폴더 설정, 모델 불러오기, 이미지 로드, BLIP Caption/VQA 분석을 담당합니다.
# =====================================================

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
PHOTO_DIR = r"/content/Photo"

# TTS 저장 폴더
TTS_DIR = r"C:\Users\lhwf4\Downloads\tts_outputs"
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


