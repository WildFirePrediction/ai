# 백엔드 연동 

산불 확산 예측 AI 서버와 백엔드 서비스를 연동하는 방법을 안내합니다.

---

## 📋 목차

1. [API 명세](#api-명세)
2. [에러 처리](#에러-처리)

---

## API 명세

### 백엔드가 받게 될 요청

> 백엔드 서버에서 AI 서버로 화재 발생 데이터를 보낼 필요는 없습니다.
> 
> AI 서버는 **화재 감지 + 예측** 모두 처리하며, 결과를 백엔드로 전송합니다.
> 
> 오직 **AI 서버에서 백엔드로 전송**되는 HTTP 요청 형식입니다.

#### 엔드포인트

```
POST {EXTERNAL_BACKEND_URL}
Content-Type: application/json
```

#### Request Body

```json
{
  "fire_id": "12345",
  "fire_location": {
    "lat": 36.5684,
    "lon": 128.7294
  },
  "fire_timestamp": "2025-12-02T14:30:00",
  "inference_timestamp": "2025-12-02T14:31:23.456789",
  "model": "a3c_16ch_v3_lstm_rel",
  "predictions": [
    {
      "timestep": 1,
      "timestamp": "2025-12-02T14:40:00",
      "predicted_cells": [
        {
          "lat": 36.5685,
          "lon": 128.7295,
          "probability": 1.0
        },
        {
          "lat": 36.5686,
          "lon": 128.7296,
          "probability": 1.0
        }
      ]
    },
    {
      "timestep": 2,
      "timestamp": "2025-12-02T14:50:00",
      "predicted_cells": [
        {
          "lat": 36.5687,
          "lon": 128.7297,
          "probability": 1.0
        }
      ]
    }
  ]
}
```

#### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `fire_id` | string | 산림청 화재 고유 ID |
| `fire_location` | object | 최초 발화 지점 좌표 |
| `fire_location.lat` | float | 위도 (WGS84) |
| `fire_location.lon` | float | 경도 (WGS84) |
| `fire_timestamp` | string | 화재 발생 시각 (ISO 8601) |
| `inference_timestamp` | string | AI 추론 완료 시각 (ISO 8601) |
| `model` | string | 사용된 모델명 |
| `predictions` | array | 타임스텝별 예측 결과 배열 |
| `predictions[].timestep` | int | 예측 단계 (1~5) |
| `predictions[].timestamp` | string | 해당 타임스텝의 예상 시각 |
| `predictions[].predicted_cells` | array | 해당 시각에 불이 번질 것으로 예측되는 셀 배열 |
| `predicted_cells[].lat` | float | 예측 셀의 위도 |
| `predicted_cells[].lon` | float | 예측 셀의 경도 |
| `predicted_cells[].probability` | float | 확산 확률 (0.0~1.0) |

#### Response (백엔드가 반환해야 할 응답)

> AI 서버는 백엔드의 응답을 기대하지 않지만, 성공 여부를 로깅하기 위해 HTTP 상태 코드를 확인합니다.

**성공 응답:**
```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "success",
  "message": "Predictions received"
}
```

**실패 응답:**
```
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "status": "error",
  "message": "Invalid data format"
}
```

---


## 에러 처리

### 백엔드 전송 실패 시 동작

현재 AI 서버는 백엔드 전송 실패 시에도 추론을 계속 진행합니다:

1. 예측 결과는 로컬 파일로 저장됨 (`inference/rl/outputs/api/`)
2. 에러 로그가 콘솔에 출력됨
3. 다음 화재 이벤트 처리는 정상적으로 계속됨

### 일반적인 에러와 해결방법

| 에러 메시지 | 원인 | 해결방법 |
|------------|------|---------|
| `Backend timeout after 30s` | 백엔드 응답 지연 | 백엔드 서버 상태 확인, 타임아웃 값 증가 고려 |
| `Failed to send to backend: Connection refused` | 백엔드 서버 미응답 | 백엔드 URL 확인, 네트워크 연결 확인 |
| `Failed to send to backend: SSL certificate verify failed` | HTTPS 인증서 문제 | 유효한 SSL 인증서 확인 |
| `Inference system not initialized` | Flask 서버 초기화 실패 | 모델 파일 경로 확인, GPU 메모리 확인 |
| `Flask inference server is not responding` | Flask 서버 미실행 | Flask 서버 시작 확인 |

### 백엔드에서 처리해야 할 사항

1. **중복 요청 필터링**
   - 같은 `fire_id`에 대해 중복 요청이 올 수 있습니다
   - 백엔드에서 `fire_id`를 기준으로 중복 제거 처리 권장

2. **데이터 검증**
   - 위경도 범위 검증 (한국: 위도 33~39, 경도 124~132)
   - 타임스탬프 유효성 검증
   - `predicted_cells` 배열이 비어있지 않은지 확인

3. **비동기 처리 권장**
   - AI 서버는 30초 타임아웃으로 대기합니다
   - 백엔드에서 즉시 200 응답 후 비동기로 데이터 처리하는 것을 권장합니다
