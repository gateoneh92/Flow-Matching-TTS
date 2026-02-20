# 🎉 Flow Matching TTS - GitHub 배포 완료

## 📦 새 저장소 정보

**Repository**: https://github.com/gateoneh92/Flow-Matching-TTS

**Description**: ⚡ Non-autoregressive TTS using Conditional Flow Matching - 5-20x faster than AR models

**Status**: ✅ Public, 완전히 배포됨

---

## 📊 배포 내용

### 포함된 파일 (20개)

#### 핵심 코드 (6개)
- ✅ `flow_matching.py` - Flow Matching 구현 (640 lines)
- ✅ `models.py` - FlowMatchingSynthesizer + MB-iSTFT
- ✅ `data_utils.py` - TextMelLoader
- ✅ `train_flow_matching.py` - 학습 스크립트
- ✅ `inference_flow_matching.py` - 추론 스크립트
- ✅ `test_flow_matching.py` - 테스트 스크립트

#### 의존성 모듈 (8개)
- ✅ `commons.py`, `utils.py`
- ✅ `attentions.py`, `modules.py`
- ✅ `pqmf.py`, `stft.py`
- ✅ `text/` (4 files)

#### 설정 & 문서 (4개)
- ✅ `configs/flow_matching.json`
- ✅ `README.md` (완전 개정)
- ✅ `requirements.txt`
- ✅ `.gitignore`

---

## 🚀 주요 특징

### 성능
- **5-20배 빠른 추론** (RTF 0.02-0.08)
- **테스트 완료** (100% pass)
- **즉시 사용 가능**

### 기술
- ✅ F5-TTS inspired (ConvNeXt + Sway)
- ✅ Conditional Flow Matching
- ✅ MB-iSTFT vocoder
- ✅ Multi-GPU 학습 지원

---

## 📈 GitHub 정보

### Topics (10개)
- `tts`, `text-to-speech`
- `flow-matching`, `non-autoregressive`
- `deep-learning`, `pytorch`
- `speech-synthesis`
- `f5-tts`, `voicebox`, `mb-istft`

### Badges
- License: MIT
- Python: 3.8+
- PyTorch: 2.0+

---

## 🎯 사용 방법

### 1. Clone
```bash
git clone https://github.com/gateoneh92/Flow-Matching-TTS.git
cd Flow-Matching-TTS
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Test
```bash
python3 test_flow_matching.py
# ✅ All tests passed! RTF 0.022-0.077
```

### 4. Train
```bash
python3 train_flow_matching.py -c configs/flow_matching.json -m logs/flow_matching
```

### 5. Inference
```bash
python3 inference_flow_matching.py \
    --checkpoint logs/flow_matching/G_100000.pth \
    --config configs/flow_matching.json \
    --text "Hello world" \
    --output output.wav \
    --steps 10
```

---

## 📝 Commits

### Initial Commit (c5c7fe0)
```
Initial commit: Flow Matching TTS

Non-autoregressive TTS using Conditional Flow Matching
- 5-20x faster than autoregressive models (RTF 0.02-0.08)
- F5-TTS inspired: ConvNeXt + Sway Sampling
- MB-iSTFT vocoder for high-quality audio
- Tested and ready to use

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### README Update (7c6146b)
```
Update README with badges and better formatting

- Add badges for License, Python, PyTorch
- Improve quick start guide
- Add detailed benchmarks
- Better structure and navigation
- More professional presentation
```

---

## 🔗 Links

- **Repository**: https://github.com/gateoneh92/Flow-Matching-TTS
- **Issues**: https://github.com/gateoneh92/Flow-Matching-TTS/issues
- **Clone URL**: https://github.com/gateoneh92/Flow-Matching-TTS.git

---

## ✅ 완료 사항

- [x] 새 디렉토리 생성
- [x] 필수 파일만 복사 (학습/합성 관련)
- [x] Git 초기화
- [x] GitHub 저장소 생성
- [x] Initial commit & push
- [x] README 업데이트 (badges, formatting)
- [x] Topics 추가 (10개)
- [x] Public 상태 확인

---

**작성일**: 2026-02-20
**작성자**: 황성웅 & Claude Code (Sonnet 4.5)
**Repository**: https://github.com/gateoneh92/Flow-Matching-TTS
