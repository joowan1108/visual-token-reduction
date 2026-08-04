# 02. 구현 매핑: Focus Token Selection for SmolVLA

## 범위와 provenance

대상은 `00_hypothesis.md`의 late action-aware visual-token selection이다. 분석일은 2026-08-04이며 local HEAD는 `bad0260a461557a09dc3a16a327091dbebd3217d`다. 다음 local 파일과 심볼을 직접 대조했다.

- `src/lerobot/policies/smolvla/configuration_smolvla.py::SmolVLAConfig`
- `src/lerobot/policies/smolvla/modeling_smolvla.py::VLAFlowMatching`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py::SmolVLMWithExpertModel`
- `tests/policies/smolvla/test_smolvla_rtc.py`
- `tests/processor/test_smolvla_processor.py`

참조 자료:

- [FocusVLA](https://arxiv.org/html/2603.28740): 공식 코드는 아직 공개되지 않아 논문 수식만 참조
- [SmolVLA](https://arxiv.org/html/2506.01844), [공식 LeRobot 구현](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)
- [SAFE-Pruner](https://arxiv.org/html/2605.29662): 공식 repository 미확인, 논문 수식과 반례만 참조
- [VLA-Pruner](https://arxiv.org/html/2511.16449), [공식 구현](https://github.com/MINT-SJTU/VLA-Pruner): OpenVLA 계열 구현이므로 코드 직접 이식 대상이 아님

## baseline 실행 흐름

### 학습

```text
lerobot-train
  → lerobot_train.py::train / update_policy
  → SmolVLAPolicy.forward
  → VLAFlowMatching.forward
  → embed_prefix(images, language, state)
  → sample noise/time and embed_suffix(noisy actions, timestep)
  → SmolVLMWithExpertModel.forward(use_cache=False)
  → action_out_proj
  → flow-matching MSE
```

`VLAFlowMatching.forward`는 `x_t = t * noise + (1 - t) * actions`, `u_t = noise - actions`를 만들고 action expert가 예측한 `v_t`와 elementwise MSE를 계산한다. local `sample_time`은 `Beta(1.5, 1.0) * 0.999 + 0.001`과 일치한다.

### 추론

```text
SmolVLAPolicy.select_action / predict_action_chunk
  → VLAFlowMatching.sample_actions
  → full prefix prefill + DynamicCache
  → euler_integrate(num_steps=10)
      → denoise_step(current x_t, timestep)
      → embed_suffix
      → SmolVLMWithExpertModel.forward(cached prefix)
      → action_out_proj
```

현재 noisy-action query는 각 Euler step의 `x_t`와 timestep에서 생성된 suffix hidden state의 expert query projection이다. 가설대로라면 selection은 action chunk당 한 번이 아니라 **각 denoising step과 각 대상 expert layer에서 다시 계산**해야 한다.

## prefix와 layer 구조

`VLAFlowMatching.embed_prefix`의 순서는 다음과 같다.

```text
camera 0 patch tokens
camera 1 patch tokens
...
language tokens
state token(s)
optional padding
```

`add_image_special_tokens=True`이면 각 camera patch span 앞뒤에 special token이 추가된다. 이 token들은 visual patch top-k 대상에서 제외해야 한다.

local 기본값은 다음과 같다.

| 항목 | 값 |
|---|---|
| `attention_mode` | `cross_attn` |
| `num_vlm_layers` | 16 |
| `num_expert_layers` | -1, 즉 VLM과 동일한 16 |
| `self_attn_every_n_layers` | 2 |
| `chunk_size` | 50 |
| `num_steps` | 10 |
| `use_cache` | true |
| `compile_model` | false |

`SmolVLMWithExpertModel.forward`는 cache prefill에서는 모든 layer를 `forward_attn_layer`로 실행한다. 그 외에는 `layer_idx % 2 == 0`인 0, 2, ..., 14층이 self-attention이고 1, 3, ..., 15층이 `forward_cross_attn_layer`다.

따라서 가설의 0-indexed layer 8–15 중 **visual-prefix cross-attention selection 대상은 9, 11, 13, 15층**이다. 8, 10, 12, 14층까지 변경하면 self-attention도 바꾸므로 가설의 “cross attention layer 구간에만” 조건을 위반한다.

학습은 `use_cache=False`로 prefix K/V를 직접 계산하고, 추론은 full `DynamicCache`를 읽는다. 동일한 selection helper가 두 경로 모두에서 작동해야 한다.

## 논문 개념과 local 코드 매핑

| 개념 | local 적용 지점 | 고정 adaptation | 검증 |
|---|---|---|---|
| action Q–vision K top-k | `smolvlm_with_expert.py::forward_cross_attn_layer` | 실제 expert Q/K projection과 head layout을 재사용 | 수동 tensor와 index 일치 |
| camera별 ranking | `modeling_smolvla.py::embed_prefix` | camera별 patch `[start,end)` metadata 전달 | camera별 `ceil(rN)` |
| spatial order 보존 | 신규 private selection helper | `topk` index를 camera 내부 오름차순 정렬 | shuffled score test |
| non-visual 보존 | prefix span과 attention mask | patch span 밖 position 전부 retain | language/state count 동일 |
| early dense | `SmolVLMWithExpertModel.forward` layer 분기 | layer 0–7 기존 경로 | layer별 KV 길이 |
| late cross-attn sparse | `forward_cross_attn_layer` | 9/11/13/15에서만 gather | boundary test |
| full cache 보존 | `sample_actions`, `denoise_step`, `DynamicCache` | cache는 crop/prune하지 않고 조회 view만 gather | cache length 동일 |
| flow objective 유지 | `VLAFlowMatching.forward` | noise/time/loss/integrator 불변 | fixed input loss regression |

## 선택 연산

각 sample `b`, 대상 layer `l`, camera `c`에 대해 실제 attention과 같은 projected query/key로 다음 pre-softmax score를 계산한다.

\[
z_{b,h,q,v}^{(l)} = Q_{b,h,q}^{(l)}(K_{b,h,v}^{(l)})^\top / \sqrt{d_h},
\]

\[
s_{b,v}^{(l)} = \frac{1}{HQ}\sum_{h,q}z_{b,h,q,v}^{(l)}.
\]

camera `c`의 valid visual patch 수가 `N`이면 다음을 사용한다.

\[
k = 0\;(N=0), \qquad k=\max(1,\lceil rN\rceil)\;(N>0).
\]

- padding mask를 score에 적용해 invalid patch가 선택되지 않게 한다.
- camera별 `topk` 후 index를 오름차순 정렬한다.
- 선택 position의 expert-projected K와 V, 해당 attention-mask column을 함께 gather한다.
- visual span 밖의 language/state/special/padding layout은 그대로 유지한다.
- ranking은 softmax 전 logit으로 충분하다. 계측용 top-k attention mass만 masked softmax로 계산한다.
- head 수가 다르면 기존 attention interface가 사용하는 grouped-query head 처리를 그대로 따른다.
- score aggregation은 수치 안정성을 위해 float32로 수행하고 실제 attention dtype으로 되돌린다.

## 최소 변경 경계

### 변경 대상

1. `src/lerobot/policies/smolvla/configuration_smolvla.py`

   - `focus_token_keep_ratio: float = 1.0`
   - `focus_token_start_layer: int = 8`
   - `0 < ratio <= 1`과 layer 범위 검증

   기본 `1.0`은 기존 checkpoint와 dense behavior를 보존한다. camera별 selection, score aggregation, order restore는 이번 실험에서 고정된 알고리즘이므로 추가 config로 만들지 않는다.

2. `src/lerobot/policies/smolvla/modeling_smolvla.py`

   - `embed_prefix`에서 camera별 patch span metadata 생성
   - 학습 `forward`와 추론 `sample_actions`/`denoise_step`을 통해 metadata 전달
   - image preprocessing, flow loss, Euler integration은 변경하지 않음

3. `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

   - `forward_cross_attn_layer`의 expert 경로에서만 selection
   - training direct-prefix와 inference cached-prefix에 같은 private helper 사용
   - full cache는 수정하지 않음

4. 기존 SmolVLA test module

   - local test 구조를 재사용해 최소 회귀 테스트 추가
   - 새 framework나 별도 pruning subsystem은 추가하지 않음

### 변경하지 않을 대상

- `src/lerobot/scripts/lerobot_train.py`
- dataset/dataloader
- optimizer/scheduler 공통 구현
- flow-matching 공통 유틸
- evaluator의 success 판정
- processor normalization/tokenization
- checkpoint state-dict 형식

새 dependency는 필요 없다. `torch.topk`, `torch.gather`, 기존 mask/cache API만 사용한다.

## checkpoint와 설정 호환성

- 신규 설정에는 trainable parameter가 없어야 하며 기존 state-dict key를 바꾸지 않는다.
- 기존 checkpoint를 신규 config default `keep_ratio=1.0`으로 strict load할 수 있어야 한다.
- Dense와 Focus 조건은 같은 initial state-dict hash를 사용한다.
- `num_vlm_layers=16`, expert 16층, `attention_mode=cross_attn`, `self_attn_every_n_layers=2`가 아니면 layer 경계 의미가 달라지므로 실험 시작 전에 실패시킨다.
- 가설의 scheduler decay 100K는 local default 30K와 다르므로 resolved config에서 반드시 명시한다.
- 비교 checkpoint `sungkyunner/smolvla_libero_baseline`은 config, processor, dataset stats 호환성을 별도로 확인하고 primary baseline으로 재사용하지 않는다.
- `compile_model=False`를 주 실험에서 고정한다. ragged span/top-k의 compile 지원은 후속 범위다.

## 위험요소

1. **절감 범위가 작다.** vision encoder, VLM prefill, full cache, early layers, late self-attention과 MLP는 그대로며 네 개 late cross-attention의 QK/AV만 줄어든다.
2. **selection overhead가 이득을 상쇄할 수 있다.** SmolVLA는 이미 camera당 visual token 수가 작아 top-k, sort, gather 비용이 dense attention보다 클 수 있다.
3. **cache memory는 줄지 않는다.** full `DynamicCache` 보존 조건 때문에 peak memory가 같거나 임시 gather tensor로 늘 수 있다.
4. **hard top-k는 비미분 가능하다.** 선택된 K/V 경로에는 gradient가 흐르지만 index 경계에는 흐르지 않는다.
5. **학습/추론 경로가 다르다.** direct K/V와 cached K/V 중 하나만 적용하면 train–eval mismatch가 난다.
6. **special/empty camera 처리 오류가 치명적이다.** special token을 patch로 세거나 false-masked camera를 활성화하면 budget과 output이 깨진다.
7. **효율만 좋아질 수 있다.** SAFE-Pruner의 flow-matching 반례처럼 success rate가 하락할 가능성을 primary 판정에서 분리해야 한다.

## 최소 테스트

1. 두 camera의 점수를 다르게 만든 helper test로 25/50/75%, `ceil`, camera 독립성, spatial order를 확인한다.
2. language/state/special token과 fully masked camera의 보존을 확인한다.
3. `keep_ratio=1.0`에서 fixed seed/noise/time의 기존 dense loss와 action output이 허용 오차 내 동일한지 확인한다.
4. 0–7과 8/10/12/14는 dense, 9/11/13/15만 sparse인지 확인한다.
5. 작은 batch의 `forward().backward()`와 `sample_actions()`가 finite하며 양 경로에서 cache length가 불변인지 확인한다.
6. 기존 `smolvla_base` checkpoint를 신규 parameter 없이 dense/sparse config로 load한다.

## 구현 전 승인 게이트

사용자 승인 후 `hypothesis_implementer`가 source를 수정하기 전에 다음을 다시 기록한다.

- local HEAD와 dirty working tree
- 실제 LIBERO camera 순서와 runtime patch span
- baseline/intervention initial state hash
- 비교 checkpoint의 processor/config/stats 호환성
- sparse layer를 9/11/13/15로 해석한다는 점
- head/query 평균 score, `compile_model=False`, dense backward compatibility

승인 전에는 source를 수정하거나 `hypothesis_implementer`를 실행하지 않는다.
