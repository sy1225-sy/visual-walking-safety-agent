# =====================================================
# 정원영 담당 코드
# 역할: YOLO 객체 탐지, 8x8 그리드 위치 분석, 위험 후보 생성 모듈
# 설명: 객체 위치/거리감/보행 경로 침범 여부를 계산하고 위험 후보를 생성합니다.
# =====================================================

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


