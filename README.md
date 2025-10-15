# Wildfire Data Preconfiguration

(MDP) <S,A,P,R>
- s \in S, s = (x, y, te, l, w, d, rh, r, i,tm) 
- a \in A, a = (N, S, E, W, NE, NW, SE, SW, STAY)
- P: Transition Probability, P(s'|s,a)
- R: Reward Function, R(s,a,s')

## 화재 데이터

### Nasa FIRMS

`VIIRS - 300m X 300m`

* x - `LONGITUDE`
* y - `LATITUDE`
* te - `BRIGHTNESS`, `BRIGHT_T31`
* i - `FRP`

## 정적 데이터

### RSP (Relative Slope Position)(상대경사위치)

### FSM (Forest Stand Map)(임상도)

### GISBuildingInfo (GIS 건물정보)

### LCM (Land Cover Map)(토지 피복도)

* l - (water, vegetation, bare land, built up, other)

### DEM (Digital Elevation Model)(고도)


## 동적 데이터

### KMA-AWS (기상청 AWS매분자료)

* w - WS1, WS10 (1분, 10분 평균 풍속)
* d - WD1, WD10 (1분, 10분 평균 풍향)
* te - TA (1분 평균 기온)
* rh - HM (1분 평균 상대습도)
* r - RN-15m, RN-60m (15분, 60분 평균 강수량)

