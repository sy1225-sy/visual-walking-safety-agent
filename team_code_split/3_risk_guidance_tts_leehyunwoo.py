# =====================================================
# 이현우 담당 코드
# 역할: Gemma 후보 검토, 최종 위험도 판단, 안내 문장 생성 및 TTS 출력 모듈
# 설명: 후보 검토, 최종 결과 생성, 안내 멘트, TTS 저장, 전체 실행 흐름을 담당합니다.
# =====================================================

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
