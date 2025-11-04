# WildFirePrediction-AI

## 개요

중앙대학교 25-2 캡스톤디자인 프로젝트로 진행되는 **산불 확산 예측 서비스 개발**의 AI Repo

## 강화학습 모델
### A3C (Asynchronous Advantage Actor-Critic)
- 병렬로 여러 에이전트를 학습시켜 안정성과 수렴 속도를 향상시키는 강화학습 알고리즘
- 기존의 MCTS-A3C 모델 구조를 기반으로 한국 산불 데이터에 맞게 피처를 수정하여 사용
### 마코프 결정 과정
(MDP) <S,A,P,R>
- $s \in S, s = (x,y,te,l,w,d,rh,i,tm,r,dem,rsp,fsm)$
- $a \in A, a = (N, S, E, W, NE, NW, SE, SW, STAY)$
- $P$: Transition Probability, $P(s'|s,a)$
- $R$: Reward Function, $R(s,a,s')$

## 데이터 구성

### 화재 데이터

#### NASA FIRMS VIIRS (화재 감지 데이터)

### 정적 타일링 데이터

#### DEM (Digital Elevation Model)(고도)

#### RSP (Relative Slope Position)(상대경사위치)

#### FSM (Forest Stand Map)(임상도)

#### NDVI (Normalized Difference Vegetation Index)(식생지수)

#### LCM (Land Cover Map)(토지 피복도)


### 동적 타일링 데이터

#### KMA_AWS (기상청 AWS매분자료/풍속,풍향,상대습도,강수량,기온)


