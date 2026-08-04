# 01. 문헌 조사: Focus Token Selection for SmolVLA

## 조사 범위

`00_hypothesis.md`의 핵심 주장은 SmolVLA action expert가 초기 8개 층에서는 전체 visual token을 사용하고, 후기 cross-attention에서는 현재 noisy-action query와 관련 높은 visual token만 카메라별로 선택하면 background attention dilution을 줄여 LIBERO success rate와 계산 효율을 함께 개선할 수 있다는 것이다.

조사일은 2026-08-04이다. 논문 원문과 저자가 연결한 공식 저장소를 우선 사용했다. 논문이 보고한 결과와 이 가설에 대한 추론을 구분한다.

## 핵심 결론

- 50% visual-token 유지율은 선행연구에서 성능과 효율의 균형점으로 반복 관찰되어 primary intervention으로 합리적이다.
- action-query 기반 saliency는 semantic/prefill attention만 사용하는 pruning보다 VLA action에 더 직접적인 근거가 있다.
- 그러나 pruning이 flow-matching VLA의 success rate를 자동으로 높이지는 않는다. SAFE-Pruner의 `π0.5` 결과는 계산량을 줄이면서 success rate가 1.0pp 하락한 직접 반례다.
- shallow attention만으로 필요한 token을 일찍 제거하면 후기 층의 saliency 변화와 subtask 전환을 놓칠 수 있다. 따라서 이 가설의 late-only, current-action-aware 설계는 중요한 안전장치다.
- FocusVLA는 autoregressive VLA-Adapter 계열이고 SmolVLA는 flow-matching action expert이므로, 본 실험은 FocusVLA 재현이 아니라 구조를 제한한 SmolVLA ablation이다.

## 논문별 근거

### 1. FocusVLA

- 논문: [FocusVLA: Focused Visual Utilization for Vision-Language-Action Models](https://arxiv.org/html/2603.28740)
- 직접 관련성: 가장 높음
- 공식 구현: 논문은 code를 공개할 예정이라고만 밝히며, 조사 시점에 공식 repository 링크가 없다.

FocusVLA는 visual-token 과다로 attention이 희석되고 task-irrelevant background가 action accuracy를 저해한다는 문제를 제시한다. Patch-level Focus는 action query와 vision key의 cross-attention score로 top-k visual token을 선택한다(Eq. 9). 이는 가설의 scaled QK compatibility와 직접 대응한다.

Table 2의 통제된 cascaded/no-gate/VLM 비교는 다음과 같다.

| Visual tokens | Retention | LIBERO average SR |
|---:|---:|---:|
| 512 | 100% | 97.0% |
| 384 | 75% | 97.8% |
| 256 | 50% | 98.0% |
| 128 | 25% | 96.3% |

50%는 dense보다 +1.0pp였고 25%는 dense보다 -0.7pp였다. 이는 50%를 primary로, 25%와 75%를 sensitivity로 두는 선택을 지지한다. 다만 FocusVLA 전체 성능에는 cascaded attention과 channel gate도 기여하므로 top-k만의 SmolVLA 효과로 일반화할 수 없다.

### 2. VLA-Pruner

- 논문: [Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference](https://arxiv.org/html/2511.16449)
- 공식 구현: [MINT-SJTU/VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner)
- 직접 관련성: 높음

VLA-Pruner는 visual-language prefill attention과 action-decode attention의 중요 token이 다르며, action-to-vision attention이 locally focused하다고 보고한다. action query에서 visual token 중요도를 계산해야 한다는 점은 본 가설을 직접 지지한다.

Table 4의 flow-matching `π0` 결과에서 50% token retention은 다음과 같다.

| Method | Avg. SR | Latency | FLOP ratio |
|---|---:|---:|---:|
| Vanilla, 100% | 94.03% | 104.53 ms | 100% |
| VLA-Pruner, 50% | 95.07% | 69.45 ms | 60.9% |

즉 50% 조건에서 +1.04pp, 1.505× speedup이 보고됐다. 그러나 VLA-Pruner는 training-free inference pruning이며 semantic prefill과 이전 action-decode attention을 결합하고 prefix token 자체를 줄인다. full prefix/cache를 유지하고 후기 expert lookup만 줄이는 현재 가설과 구현 의미가 다르다.

### 3. SAFE-Pruner

- 논문: [SAFE-Pruner: Semantic Attention-Guided Future-Aware Token Pruning for Efficient Vision-Language-Action Manipulation](https://arxiv.org/html/2605.29662)
- 공식 구현: 논문/arXiv에서 공식 repository를 확인하지 못함
- 역할: 직접 반례와 설계 경고

SAFE-Pruner는 attention saliency가 층과 subtask 전환에 따라 변하므로 shallow attention만으로 pruning하면 task-critical token을 너무 일찍 버릴 수 있다고 분석한다.

Table 1의 flow-matching `π0.5` 결과는 vanilla 96.5%에서 SAFE-Pruner 95.5%로 -1.0pp였다. 동시에 FLOPs는 2.115T에서 1.482T, latency는 35.28ms에서 24.33ms로 감소했다. 효율 개선만으로 accuracy 향상 claim을 지지할 수 없다는 직접 반례다.

Table 5의 OpenVLA-OFT ablation에서도 vanilla 96.8%에 비해 shallow-token-pruning-only(`w/o forecast`)는 94.5%로 -2.3pp였다. late-only 적용과 현재 action query에 기반한 재선택을 유지해야 한다는 근거다.

### 4. SmolVLA

- 논문: [SmolVLA: A vision-language-action model for affordable and efficient robotics](https://arxiv.org/html/2506.01844)
- 공식 구현: [LeRobot SmolVLA](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)
- 대상 baseline: local LeRobot HEAD `bad0260a461557a09dc3a16a327091dbebd3217d`

SmolVLA는 SmolVLM2 backbone의 첫 16개 LLM layer, flow-matching action expert, 50-action chunk, 10-step inference를 사용한다. VLM은 frozen이고 action expert를 학습한다. 논문은 LIBERO 4 suites × 10 tasks의 binary success rate를 사용한다.

현재 local config도 `num_vlm_layers=16`, `num_expert_layers=-1`, `self_attn_every_n_layers=2`, `chunk_size=50`, `num_steps=10`을 확인했다. 따라서 본 실험의 action-expert 16-layer 전제는 baseline과 일치한다.

### 5. Compose by Focus

- 논문: [Compose by Focus](https://arxiv.org/html/2509.16053v2)
- 공식 구현: [han20192019/skill-composition-code](https://github.com/han20192019/skill-composition-code)
- 관련성: 간접적

이 연구는 manipulation에서 attention을 task-relevant region에 집중하고 distractor를 억제하는 것이 skill composition에 유리하다는 간접 근거를 제공한다. 하지만 SmolVLA token pruning이나 동일한 flow-matching 비교를 다루지 않으므로 primary 설계 수치의 근거로 사용하지 않는다.

## 지지와 반증의 종합

| 관찰 | 가설에 대한 의미 |
|---|---|
| FocusVLA 50%: +1.0pp | 50% primary와 최소 의미 효과 +1.0pp의 근거 |
| FocusVLA 25%: -0.7pp | 과도 pruning의 정보 손실 가능성 |
| VLA-Pruner `π0` 50%: +1.04pp, 1.505× | flow-matching VLA에서도 action-aware 50% pruning이 성공할 수 있음 |
| SAFE-Pruner `π0.5`: -1.0pp | 효율 향상이 accuracy 향상을 보장하지 않음 |
| SAFE shallow-only: -2.3pp | 조기/고정 saliency pruning의 위험 |

문헌은 가설을 가능성 있는 것으로 만들지만 확정하지 않는다. 가장 보수적인 결론은 “50% late action-aware selection이 검증할 가치가 있으며, success rate와 효율을 분리해 판정해야 한다”이다.

## 실험 계획에 반영할 사항

1. Primary는 Dense 대 Focus-50 하나로 고정한다.
2. 25%와 75%는 primary 선택에 사용하지 않는 sensitivity다.
3. 최소 의미 개선은 문헌 효과 크기에 맞춰 +1.0 percentage point로 사전등록한다.
4. 초기 layer와 full prefix/cache는 dense로 보존하고 late expert cross-attention만 선택한다.
5. 각 denoising step과 대상 layer에서 current noisy-action query로 saliency를 다시 계산한다.
6. success rate 하락과 효율 개선만 나타나면 accuracy claim은 unsupported로 판정한다.
7. suite/task/seed별 일관성 및 paired CI를 함께 요구한다.

## 한계

- FocusVLA와 SmolVLA의 policy family가 다르다.
- 각 논문의 token 수, camera 구성, training budget, 평가 episode 수가 동일하지 않다.
- VLA-Pruner와 SAFE-Pruner는 주로 inference-time pruning이며 본 가설은 fine-tuning에도 selection을 적용한다.
- hard top-k index는 비미분 가능하므로 “end-to-end”는 선택된 K/V 경로에 gradient가 흐른다는 제한된 의미다.
- 문헌의 단일 평균 수치는 본 실험의 3-seed paired uncertainty를 대신할 수 없다.
