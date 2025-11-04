# 데이터 전처리


# Pre-Config

✅x화점 데이터에서 ***Confidence(관측 신뢰도)***가 ***l(low)***인 데이터는 삭제한다

---

# Config

## A3C_KOR (한국 산림 대상 Standalone A3C)

MDP<S,A,P,R>

> $s \in S$, $s = (x,y,te,l,w,d,rh,i,tm,r,dem,rsp,fsm)$

상태 $s$는 지형의 각 타일을 의미한다. **Agent**의 행동 배경으로, **A3C** 모델의 ***input layer***로 직접 입력된다.

| 분류 | 설명 | 단위 | 변환 | 출처 |
| --- | --- | --- | --- | --- |
| x | 경도 (LON) | `degrees` |  | NASA(VIIRS) |
| y | 위도 (LAT) | `degrees` |  | NASA(VIIRS) |
| te | 온도 | `celcius` |  | NASA(VIIRS) |
| l | 토지피복도 | **categorical** | One-hot encoding | LCM |
| w | 풍속 | `m/s` |  | KMA(AWS) |
| d | 풍향 | `degrees` |  | KMA(AWS) |
| rh | 상대습도 | `%` |  | KMA(AWS) |
| i | 화재 intensity | integer |  | NASA(VIIRS) |
| tm | 화재 경과일 | day (integer) |  | NASA(VIIRS) |
| r | 강수량 | `mm` |  | KMA(AWS) |
| dem | 고도 | `m`  |  | DEM |
| rsp | 상대경사위치 | **Normalized (0~1)** |  | RSP |
| fsm | 임상도 | **categorical** | One-hot encoding | FSM |
| ndvi | 식생지수 | **Normalized (0~1)** |  | NDVI |

기존 **MCTS-A3C** 모델은 평평한 지형인 캐나다의 Saskatchewan 산림 데이터셋을 기반으로 구축되었으므로, 입력 피처 종류를 그대로 사용하는 것은 한국의 지형에 맞지 않다. 

산림 지형이 많고 작고 큰 언덕이 빈번하게 나타나는 한국의 지형을 고려해 DEM(고도), RSP(상대경사)를 입력 피처로 추가한다. 또한, 혼효림이 흔한 한국은 산림 내부 이질성이 크다. 임상도가 없으면 연료 연속성 위험을 놓칠 수 있으므로 FSM(임상도)도 입력 피처로 추가한다.

계절적 특징 (봄철 낙엽수 낙엽기, 가을철 초지 건조기) 등을 놓칠 수 없으므로 2주 단위로 업데이트되는 NDVI데이터도 입력 피처로 추가한다.

> $a \in A$, $a = (N,NE,NW,S,SE,SW,E,W,NONE)$


행동 $a$는 상태 $s$에서 인접 상태로 불이 확산 가능한 방향이다. 북,북동,북서,남,남동,남서,동,서,움직이지 않음 이 있다.

> $P(s\prime|s,a)$


상태 $s$에서 행동 $a$를 취했을 때 상태 $s\prime$으로 전이할 확률.

MCTS-A3C 모델의 전이 확률 모델을 그대로 사용한다.

> $R$
> 

보상 함수
MCTS-A3C 모델의 보상 함수를 그대로 사용한다.

---

# 산불 데이터

## Config

> 추출 ***feature*** → $x,y,te,i,tm$
> 

| Feature | Column (NASA-VIIRS) |
| --- | --- |
| x | LONGITUDE |
| y | LATITUDE |
| te | BRIGHTNESS, BRIGHT_T31 |
| i | FRP |
| tm | ACQ_DATE, ACQ_TIME |

산불 episode 집합 $E$ - spatiotemporal clustering으로 공간축 기준 분화

For $e \in E$, $e = \{x,y,te,i,tm\}$

- 거리 스케일 $\Delta S \le 2km$ (타일 하나당 400m이므로 25개의 타일 기반)
- 시간 스케일 $\Delta t \le 168h (\text{7 days})$

## Feature Transformation

### $x$, $y$

위경도를 투영 좌표계로 변환

```python
import pyproj

proj_src = pyproj.Proj("EPSG:4326")
proj_dst = pyproj.Proj("EPSG:5179")
x, y = pyproj.transform(proj_src, proj_dst, lon, lat)
```

- 단위 : m
- $x,y$ 값은 z-score정규화 혹은 0-1 scaling

### $te$

**BRIGHT_T31**은 $10.8 

μ

m$에서의 밝기 온도(**K)**로, 지표면 온도 (LST)로 쓰인다.

- 단위 : °C
- 변환식 : $te = \text{BRIGHT\_T31} - 273.15$
- 정규화 : 

$te_{norm} = \frac{te - \mu_{te}}{\sigma_{te}}$

⚠️ $te$를 외부 기온 데이터 (KMA AWS)대체할 가능성도 존재한다. 전처리한 **BRIGHT_T31**은 우선 ****화염 온도 proxy의 역할로 둔다.

### $i$

**BRIGHTNESS**는 $4 

μm$

에서의 밝기 온도(**K**)이고, **FRP**는 화재 강도(**MW**)이다.

물리적으로 **FRP**가 더 안정적이므로 **FRP**중심으로 처리하고, **BRIGHTNESS**는 보정 변수로 사용한다.

- 단위 : 없음
- 변환식 : $i = \alpha \cdot \text{FRP}+\beta\cdot(\text{BRIGHTNESS}-273.15)$
    - $\alpha$, $\beta$는 단위 스케일링 위한 상수로, 적절한 값으로 조정한다.
    - 단위 스케일링으로는 로그 스케일을 적용한다 : 
    
    $i’ = \log(1 + i)$
- 정규화 : 

$i_{norm} = \frac{i’ - \mu_i}{\sigma_i}$

### $tm$

ACQ_DATE, ACQ_TIME에 대해 하나의 dataframe으로 합치고, timedelta로 경과시간을 구한다

- 단위 : 시간 (H)
- 변환식 : $tm = (t-t_0)\cdot {\text{total\_seconds()}\over3600.0}$
- 정규화 : 필요 시 $\log(1+tm)$ 또는 z-score 정규화

## Feature Embedding

$[ x\_norm, y\_norm, te\_norm, i\_norm, tm\_norm]$

---

# 정적 데이터

## 고도 (DEM)

- 데이터 타입 : 연속형(float)
- 변환식
$dem\_norm = (dem - dem\_min) / (dem\_max - dem\_min)$

## 경사 (RSP)

- 데이터 타입 : 연속형(float)
- 전처리 없이 그대로 사용 (Normalized value)

## 토지피복도 (LCM)

- 데이터 타입 : 범주형 (categorical)
- 전처리
    - 각 클래스를 integer label로 매핑 → One-hot encoding
    - 걍 Pytorch 임베딩 레이어 사용

```python
lcm_embed = nn.Embedding(n_lcm_classes, embed_dim) #embed_dim = 8~16
```

## 임상도 (FSM)

- 데이터 타입 : 범주형 (categorical)
- 전처리
    - Class 별 integer encoding
    - One-hot또는 embedding사용

---

# 준정적 데이터

## 식생지수 (NDVI)

- 데이터 타입 : 연속형 (float)
- 전처리
    - NDVI 값 범위 : -1~1
    - Normalized to [0,1]

---

# 동적 데이터

## 풍속/풍향, 대온습도

## 강수

## 바람 예보/ 관측 시계열 (⚠️Inference시에만)

## 관측 신뢰도 (Pre-confg)

---

# 최종 상태 벡터

```python
s = [
    x_norm, y_norm,
    te_norm, i_norm, tm_norm,
    dem_norm, rsp_norm,
    *lcm_encoded, *fsm_encoded,
    ndvi_norm,
    w_norm, d_x, d_y, rh_norm, r_norm
]

```

---

# Composition

## 타일링

### 좌표계 정의

- 전체 영역 = $(x\text{min}, x\text{max}), (y\text{min}, y\text{max})$
- 타일 크기 = $\Delta r$ = $400\text{m}$

> 원점 (anchor) - 분석 영역의 좌하단을 400m 격자에 스냅
> 

```python
x0 = floor(xmin / 400) * 400
y0 = floor(ymin / 400) * 400
```

> 타일 인덱스
> 

```python
ix = floor((x - x0) / 400)
iy = floor((y - y0) / 400)
```

> 그리드 크기
> 

```python
W = ceil((xmax - x0) / 400)
H = ceil((ymax - y0) / 400)
```
