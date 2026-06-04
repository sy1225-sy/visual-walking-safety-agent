# 시각장애인 보행 안전 안내 에이전트

이미지를 입력하면 BLIP Caption, BLIP VQA, YOLO, Gemma 모델을 활용해 보행 중 위험 요소를 분석하고 위험도와 안내 멘트를 생성하는 프로젝트입니다.  
Colab 환경에서는 `/content/Photo` 폴더에 사진을 넣으면 해당 폴더 안의 이미지를 자동으로 분석하도록 구성되어 있습니다.

---

## 1. 프로젝트 개요

본 프로젝트는 시각장애인의 보행 안전을 돕기 위한 이미지 기반 위험 안내 에이전트입니다.

전체 흐름은 다음과 같습니다.

1. 사용자가 사진을 `/content/Photo` 폴더에 업로드
2. BLIP Caption 모델로 이미지 장면 설명 생성
3. BLIP VQA 모델로 주요 객체와 보행 관련 상황 분석
4. YOLO 모델로 객체 탐지 및 8x8 그리드 기반 위치 분석
5. Gemma 모델로 위험 후보 검토
6. 최종 위험도, 위험 요소, 위치, 거리감, 회피 방향, 안내 멘트 생성
7. gTTS를 이용해 안내 멘트를 음성 파일로 저장

---

## 2. 사용 모델 및 라이브러리

### 사용 모델

| 구분 | 모델 |
|---|---|
| 이미지 캡션 생성 | `Salesforce/blip-image-captioning-base` |
| 이미지 질의응답 | `Salesforce/blip-vqa-base` |
| 객체 탐지 | `yolo11n.pt` |
| 위험 후보 검토 | `google/gemma-2-2b-it` |
| 음성 안내 생성 | `gTTS` |

### 주요 라이브러리

- `torch`
- `transformers`
- `ultralytics`
- `Pillow`
- `numpy`
- `gtts`
- `huggingface_hub`
- `accelerate`

---

## 3. Colab 실행 환경 설정

Colab에서 GPU 사용을 권장합니다.

상단 메뉴에서 다음과 같이 설정합니다.

```text
런타임 → 런타임 유형 변경 → 하드웨어 가속기 → T4 GPU
```

---

## 4. 설치 명령어

Colab 첫 번째 셀에 아래 코드를 실행합니다.

```python
!pip install -q ultralytics transformers accelerate gtts pillow numpy
```

---

## 5. Hugging Face 로그인

Gemma 모델을 사용하기 위해 Hugging Face 로그인이 필요합니다.
```python
from google.colab import userdata
userdata.get('HF_TOKEN')
```

Coolab에서 왼쪽 보안 비밀을 통해 HF_TOKEN을 입력합니다.

> `google/gemma-2-2b-it` 모델은 사용 전 Hugging Face 페이지에서 접근 권한 동의가 필요합니다.

---

## 6. 폴더 생성 및 사진 경로

아래 경로에 사진 입력 폴더를 생성해줍니다.

```text
/content/Photo
```

`/content/Photo` 폴더 안에 분석할 이미지를 넣으면 코드 실행 시 해당 폴더의 이미지가 자동으로 분석됩니다.

지원하는 이미지 확장자는 다음과 같습니다.

```python
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
```

---

## 7. 실행 방법

### 1단계: 이미지 업로드

Colab 왼쪽 파일 탭에서 `/content/Photo` 폴더를 만들고, 분석할 이미지를 업로드합니다.

예시 폴더 구조는 다음과 같습니다.

```text
/content
 ├── Photo
 │    ├── image1.jpg
 │    ├── image2.png
 │    └── image3.jpeg
 └── tts_outputs
```

### 2단계: 전체 코드 실행

코드 전체를 실행하면 `/content/Photo` 폴더 안의 이미지가 순서대로 분석됩니다.

---

## 8. 출력 결과

최종 출력 형식은 다음과 같습니다.

```text
이미지   : image1.jpg
위험도   : 중
위험 행동: 서행하세요
위험 요소: 자전거
위치     : 정면
거리감   : 가까운
위험 이유: 자전거가 보행 경로와 가까워 부딪힐 수 있습니다.
회피 방향: 오른쪽
상황 설명: 정면 가까운 위치에 자전거가 있습니다.
안내 멘트: 서행하세요. 정면 가까운 위치에 자전거가 있습니다. 오른쪽 공간을 확인한 뒤 천천히 피해가세요.
```

---

## 9. TTS 음성 파일 저장 위치

최종 안내 멘트는 음성 파일로 변환되어 아래 폴더에 저장됩니다.

```text
/content/tts_outputs
```

저장 파일명은 원본 이미지 파일명을 기준으로 생성됩니다.

예시:

```text
image1_warning.mp3
image2_warning.mp3
```

---

## 10. 위험도 기준

| 위험도 | 위험 행동 | 예시 | 의미 |
|---|---|---|---|
| 상 | 정지 필요 | 계단, 차량, 트럭, 버스, 오토바이 | 즉시 멈추고 주변 확인이 필요한 상황 |
| 중 | 서행 필요 | 사람, 자전거, 의자, 책상, 상자, 쓰레기통, 정수기, 세면대, 변기 등 | 천천히 이동하며 회피가 필요한 상황 |
| 하 | 주의 필요 | 문, 엘리베이터 문, 벽, 복도, 화장실 입구 등 | 주변 확인이 필요한 낮은 위험 상황 |
| 판단불가 | 재촬영 필요 | 판단 불가 | 이미지 정보가 부족하거나 판단이 어려운 상황 |
