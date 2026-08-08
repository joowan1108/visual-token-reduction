# Literature review: Minimal End-to-End Skill Linking for SmolVLA

## Review scope

- 조사일: 2026-08-07
- 대상 가설: 현재 skill embedding과 transition head를 SmolVLA action expert와 공동 fine-tuning하면 action-only SmolVLA보다 LIBERO-Long complete success가 향상되는가?
- 출처 제한: 원 논문, 저자 공식 project/code, Hugging Face dataset repository와 LeRobot 공식 문서만 사용했다.
- 근거 등급:
  - **Direct**: 제안한 최소 intervention 자체를 통제 실험으로 검증
  - **Indirect-supporting**: skill/subtask supervision 또는 unified training의 효용을 보이지만 구조가 다름
  - **Indirect-contradicting**: 다른 병목이나 더 강한 구조가 필요할 가능성을 보임

## 2026-08-08 dataset amendment

실험 입력은 semantic mapping과 anomaly 정제를 포함한 [`sungkyunner/libero_10_subtask_semantic_clean`](https://huggingface.co/datasets/sungkyunner/libero_10_subtask_semantic_clean) revision `d619815aeba9c06c70fc558838137dd57a651ce1`로 고정했다. 이 revision은 495 episodes, 136,425 frames, 941 contiguous atomic segments, 446 adjacent-subtask boundaries와 `meta/subtasks.parquet`를 포함한다. 따라서 아래 원본 dataset 검토에서 지적한 이름 ontology 부재는 해소됐다. 다만 seen LIBERO-10의 9개 관측 directed pair만 다루므로 unseen composition 근거는 여전히 없다.

학습 pair는 자연스럽게 연속한 같은 episode 경계에서만 가져오며 경계당 하나의 start index만 사용한다. 이는 문헌 근거를 새로 추가하는 intervention이 아니라 pair 중복으로 효과를 부풀리지 않기 위한 실험 통제다.

## 결론

확인한 문헌 중 **작은 numeric skill embedding + linear transition head + 하나의 shared action expert 공동 fine-tuning**을 planner, MoE, scene graph, phase masking 없이 검증한 연구는 없다. 따라서 현재 가설을 직접 지지하는 선행 결과는 없으며, 본 실험은 기존 방법의 단순 재현이 아니라 최소 구성요소의 독립 효과를 측정하는 ablation이다.

간접 근거는 양방향이다.

- LoHoVLA와 AtomicVLA는 명시적 subtask/atomic-skill supervision이 long-horizon 성능과 함께 향상될 수 있음을 보고한다.
- Long-VLA는 단계 정보를 shared policy에서 end-to-end로 학습하는 방향을 지지한다.
- 반면 Compose by Focus는 atomic skill 성능이 높아도 composed scene의 visual distribution shift 때문에 composition이 실패할 수 있음을 보인다.
- AtomicVLA의 이득은 plan generation과 SG-MoE까지 포함하고, Long-VLA의 이득은 phase decomposition과 input masking까지 포함한다. 이 결과를 작은 transition head의 효과로 환원할 수 없다.

따라서 `+5%p` 개선은 문헌에서 도출된 기대 효과가 아니라 본 실험의 사전 정의된 최소 의미 효과다. Oracle skill conditioning이 action-only baseline을 이기지 못하면 transition predictor를 확장하기 전에 가설부터 기각해야 한다.

## Evidence summary

| 연구 | 가설 관련 근거 | 보고된 핵심 결과 | 근거 등급 | 정확한 위치 |
|---|---|---|---|---|
| SmolVLA | Flow-matching action expert와 shared visual-language context를 제공하는 직접 baseline | 0.45B 모델 LIBERO-Long 71%, 4-suite 평균 87.3% | Baseline only | Sec. 3.1, Table 2, Tables 6–9 |
| LoHoVLA | 명시적 subtask generation과 action prediction의 공동 학습 | Vanilla VLA 대비 다수 seen/unseen long-horizon task에서 큰 성공률 증가 | Indirect-supporting | Secs. 3.3–3.4, Algorithm 1, Table 2 |
| AtomicVLA | atomic abstraction을 action generation에 조건으로 제공 | LIBERO-Long 95.2% 대 π0 85.2%; SG-MoE ablation 95.2% 대 일반 MoE 88.6% | Indirect-supporting, structurally confounded | Secs. 3.3–3.5, Tables 1, 2, 5 |
| Compose by Focus | atomic policy만으로 composition이 보장되지 않음을 실증 | π0의 simulation composition 평균 약 0.30, scene-graph policy 약 0.86 | Indirect-contradicting | Sec. IV-B/C, Table I; Sec. V, Tables II–III |
| Long-VLA | phase state를 shared policy 안에서 unified training | L-CALVIN D→D 평균 길이 4.75 대 RoboVLMs 2.88; ablation에서 base 4.11→4.81 | Indirect-supporting and contradicting | Secs. 3.1–3.2, Tables 2–4, Appendix Tables 6–8 |

`약 0.30`과 `약 0.86`은 Compose by Focus Table I(b)의 다섯 simulation task 값을 이 문서에서 산술 평균한 값이며 논문이 별도로 보고한 aggregate가 아니다.

## 1. SmolVLA

### Citation and source

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. **SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.** arXiv:2506.01844, 2025. [Paper](https://arxiv.org/abs/2506.01844), [official LeRobot implementation](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla).

### Reported evidence

- **Dataset/protocol:** LIBERO Spatial, Object, Goal, Long과 Meta-World를 multi-task simulation setting으로 평가한다. LIBERO protocol은 task당 10 trials이며 완전 성공만 1로 채점한다. Pretraining은 481 community datasets, 22.9K episodes, 10.6M frames에서 200K steps, global batch 256으로 수행한다. Action expert는 50-action chunk를 flow matching으로 학습한다. 근거: Secs. 3.1, 3.2, 4.1, 4.3.
- **Baseline/intervention:** SmolVLA 자체의 핵심 intervention은 compact VLM, visual-token reduction, VLM layer skipping, interleaved cross/self-attention action expert다. 명시적 current-skill state나 transition objective는 없다.
- **Primary result:** Table 2에서 0.45B SmolVLA는 LIBERO Spatial/Object/Goal/Long에서 90/96/92/71%, 평균 87.3%를 보고한다. π0 VLM-initialized variant는 Long 48%, robotics-pretrained π0는 Long 73%지만 initialization과 pretraining이 달라 단일 요소의 인과 비교가 아니다.
- **Relevant ablation:** Table 6에서 interleaved CA+SA는 LIBERO 평균 85.5%로 CA 79.0%, SA 74.5%보다 높다. Tables 8–9는 VLM layer 수와 expert capacity만 바꾸어도 결과가 크게 달라짐을 보인다.
- **Seeds/uncertainty:** 논문은 LIBERO task당 10 trials를 명시하지만 training seed 수, confidence interval, statistical significance test는 보고하지 않는다.

### Relevance and limitation

SmolVLA는 본 실험의 직접 baseline과 insertion point를 정당화한다. 그러나 skill-aware auxiliary objective의 효용을 검증하지 않았으므로 가설 지지는 아니다. 또한 작은 evaluation count와 seed/CI 부재 때문에 5%p 수준 차이는 자체적으로 불확실할 수 있다.

## 2. LoHoVLA

### Citation and source

Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, and Zhijie Deng. **LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks.** arXiv:2506.00411, 2025. [Paper](https://arxiv.org/abs/2506.00411).

### Reported evidence

- **Dataset:** LoHoSet은 Ravens 기반 20 long-horizon tasks에 task당 1,000 expert demonstrations를 제공한다. 본문 실험의 stage 1은 14 long-horizon tasks에 task당 1,000 demonstrations를 사용하고, stage 2는 pick-and-place primitive별 10,000 demonstrations를 추가한다. Subtask는 simulator state와 manually designed rules로 생성한다. 근거: Secs. 3.2, 4.1.
- **Intervention:** PaliGemma backbone이 language subtask와 discretized action tokens를 생성한다. Loss는 `L_text + L_action`이고, reward/failure count를 이용한 hierarchical closed-loop replanning을 사용한다. 근거: Secs. 3.3–3.4, Algorithm 1.
- **Baseline:** Vanilla VLA는 같은 dataset에서 intermediate language output 없이 action만 예측한다. 다만 LoHoVLA는 3-epoch text stage와 1-epoch joint stage를 거치고 vanilla는 5 epochs로 학습되어 intervention이 단일 head 차이만은 아니다.
- **Effect size:** Table 2에서 seen task B의 success는 vanilla 0.0% 대 LoHoVLA 91.5%, E는 3.5% 대 81.0%다. Unseen task I는 1.5% 대 52.0%, K는 33.0% 대 54.5%다. Primitive task A에서는 vanilla 79.0%가 LoHoVLA 77.5%보다 높아 명시적 hierarchy가 항상 이득인 것은 아니다.
- **Seeds/uncertainty:** training seeds, 평가 instance 수, CI 또는 significance test는 논문에 명시되지 않는다.

### Relevance and limitation

명시적 subtask supervision과 action supervision을 shared representation에서 학습하는 것이 action-only보다 유리할 수 있다는 가장 가까운 간접 근거다. 그러나 language plan decoder, 대규모 subtask-labelled data, reward-based completion signal과 replanning이 모두 포함된다. 본 transition head는 test-time subtask reward 없이 자체 예측으로 전환해야 하므로 LoHoVLA보다 어려운 조건이다.

## 3. AtomicVLA

### Citation and source

Likui Zhang, Tao Tang, Zhihao Zhan, Xiuwei Chen, Zisheng Chen, Jianhua Han, Jiangtong Zhu, Pei Xu, Hang Xu, Hefeng Wu, Liang Lin, and Xiaodan Liang. **AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots.** arXiv:2603.07648, CVPR 2026. [Paper](https://arxiv.org/abs/2603.07648), [official project](https://zhanglk9.github.io/atomicvla-web/), [official code](https://github.com/zhanglk9/AtomicVLA).

### Reported evidence

- **Intervention:** VLM이 task plan과 atomic skill abstraction을 생성하고, abstraction embedding이 SG-MoE router를 통해 shared expert와 top-1 atomic expert를 혼합한다. 근거: Sec. 3.3, Eqs. 1–3.
- **Annotations:** End-effector translation/rotation/gripper trajectory로 segment를 만들고 InternVideo2.5로 label을 보정해 reasoning chain을 구성한다. 공식 code도 별도 reasoning annotation JSON 준비를 요구한다. 근거: Sec. 3.5와 official repository Data Preparation.
- **Dataset/protocol:** LIBERO 4 suites, CALVIN ABC→D, Franka real-world tasks를 사용한다. LIBERO/real에는 5 experts, CALVIN에는 8 experts를 사용한다. Real-world data는 short task당 50, long task당 100 trajectories, 총 550 trajectories다. 근거: Sec. 4.1.
- **Effect size:** Table 1에서 AtomicVLA는 LIBERO 평균 96.6%, Long 95.2%로 π0의 평균 94.2%, Long 85.2%보다 각각 +2.4%p, +10.0%p다. Table 2에서 CALVIN average length는 4.09 대 π0 3.87이다. Table 5에서 SG-MoE 95.2%는 일반 MoE 88.6%, timestep-conditioned MoDE 89.5%보다 높다.
- **Seeds/uncertainty:** training/evaluation seeds, CI와 significance test는 보고하지 않는다.

### Relevance and limitation

Atomic skill representation이 action generation에 직접 조건으로 들어간다는 점은 skill embedding 가설을 간접 지지한다. 그러나 Table 5는 **skill-guided specialized experts**가 성능의 중요한 부분임을 동시에 시사한다. 하나의 shared expert에 embedding을 더하는 현재 방법은 specialization과 planning/recovery 경로가 없으므로 같은 효과를 기대할 직접 근거가 없다.

## 4. Compose by Focus

### Citation and source

Han Qi, Changhe Chen, and Heng Yang. **Compose by Focus: Scene Graph-based Atomic Skills.** arXiv:2509.16053, 2025. [Paper](https://arxiv.org/abs/2509.16053), [official code](https://github.com/han20192019/skill-composition-code), [official benchmark](https://github.com/han20192019/skill-composition-benchmark).

### Reported evidence

- **Intervention:** Relevant object/relation만 남긴 3D scene graph를 GNN으로 처리하고 diffusion policy로 atomic skill을 실행한다. High-level decomposition에는 VLM/ChatGPT를 사용하고, real-world evaluation에서는 LLM이 subtask completion도 감지한다. 근거: Secs. III, IV-B, V-A.
- **Baselines/protocol:** Diffusion Policy, DP3, π0와 같은 expert demonstrations 및 step limit으로 비교한다. Simulation task마다 50 randomized initial-position seeds를 평가한다. 이는 training seeds가 아니다. Real-world task는 각 20 trials이고 atomic skill당 50 demonstrations를 수집한다.
- **Effect size:** Table I(b) simulation composition에서 scene graph는 0.78/0.79/0.93/0.88/0.90인 반면 π0는 0.15/0.02/0.77/0.07/0.49다. Table I(a)에서는 여러 baseline도 single-skill 성능이 높다. Table II real-world vegetable composition은 scene graph 0.97, π0 0.05, DP3 0.20이다. Table III tool usage는 0.90 대 π0 0.075다.
- **Uncertainty:** randomized initial conditions와 trial 수는 명시하지만 training seeds, CI, significance test는 보고하지 않는다.

### Contradicting evidence for the minimal head

이 논문의 핵심 관찰은 atomic execution 능력 자체보다 composed scene에서 생기는 distractor와 object-count 변화가 failure를 만든다는 것이다. 따라서 transition label이 정확해도 SmolVLA의 visual representation이 composition-induced shift를 견디지 못하면 task success는 향상되지 않을 수 있다. 동시에 결과는 external planner, scene graph와 completion detector가 포함된 system-level 비교이므로 scene graph 하나의 순수 효과로 볼 수 없다.

## 5. Long-VLA

### Citation and source

Yiguo Fan, Pengxiang Ding, Shuanghao Bai, Xinyang Tong, Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Siteng Huang, Zhaoxin Fan, Badong Chen, and Donglin Wang. **Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation.** CoRL 2025, arXiv:2508.19958. [Paper](https://arxiv.org/abs/2508.19958), [official project](https://long-vla.github.io/).

### Reported evidence

- **Dataset/intervention:** CALVIN trajectories를 moving/interaction phase로 나눈 L-CALVIN을 만들고 phase-aware masking을 적용한다. 64-frame windows를 task detector로 label하고 object state change 10–15 frames 전에 boundary를 둔다. Moving phase는 third-person/detection cues, interaction phase는 gripper-centric cues를 선호한다. 근거: Secs. 3.1–3.2, Appendix B.1.
- **Unified training:** Separate moving/interaction policy 대신 하나의 policy에서 phase identifier와 masking을 학습한다. 이는 phase state와 action을 jointly train한다는 점에서 현재 방향과 유사하다.
- **Effect size:** Table 2의 L-CALVIN D→D에서 Long-VLA average sequence length는 4.75로 GR-1 2.96, RoboVLMs 2.88보다 높다. Appendix Table 8의 ABCD→D에서는 8.24로 RoboVLMs 6.04보다 높다.
- **Ablation:** Table 3에서 base unified policy는 Real Sorting 2.3, Cleaning 1.4, Sim D→D 4.11이고 decomposition+input adaptation+unified Long-VLA는 5.5, 2.8, 4.81이다. Table 4에서는 HULC도 2.65→3.30으로 개선된다.
- **Seeds/uncertainty:** real-world figures는 조건별 `/20` 결과를 제시하지만 training seeds, CI, significance test는 명시하지 않는다. 공식 project page는 조사 시점에도 code를 “Coming Soon”으로 표시한다.

### Relevance and limitation

Shared end-to-end policy 안에 phase state를 넣는 방향은 간접 지지된다. 반대로 주요 이득이 phase-specific visual input adaptation에서 발생한다는 결과는 transition head만으로 부족할 수 있음을 시사한다. 또한 Long-VLA의 moving/interaction phase는 semantic atomic skill ID와 동일하지 않다.

## Original-source dataset evidence: `lerobot/libero_10_subtask`

### Verified facts

공식 [dataset card](https://huggingface.co/datasets/lerobot/libero_10_subtask/blob/main/README.md)의 `meta/info.json`에는 다음이 명시돼 있다.

- LeRobot codebase version `v3.0`
- 500 episodes, 138,090 frames, 10 tasks, 10 Hz
- `subtask_index`: scalar `int64`
- two 256×256 video observations, 8D state, 7D action

공식 [Dataset Viewer](https://huggingface.co/datasets/lerobot/libero_10_subtask/viewer)는 `task_index` 범위 `0..9`, `subtask_index` 범위 `0..15`를 표시한다. 따라서 **numeric field와 관측된 값 범위**는 확인됐다.

### Metadata inconsistency

Repository의 [현재 `meta/` tree](https://huggingface.co/datasets/lerobot/libero_10_subtask/tree/main/meta)에는 `episodes/`, `info.json`, `stats.json`, `tasks.parquet`만 있고 `subtasks.parquet`는 없다. 반면 [LeRobot 공식 subtask 문서](https://huggingface.co/docs/lerobot/en/dataset_subtask)는 완전한 subtask dataset이 다음 둘을 함께 가져야 한다고 설명한다.

1. frame-level `subtask_index`
2. `meta/subtasks.parquet`의 index-to-string mapping

공식 확인식도 `"subtask_index" in dataset.features and dataset.meta.subtasks is not None`이다. 그러므로 이 repository는 raw numeric annotation은 제공하지만 자연어 subtask resolution을 포함하는 **완전한 공식 subtask metadata contract는 충족하지 않는다**.

### Consequences resolved by the clean revision

- 선택한 clean revision은 공식 sibling의 semantic mapping을 포함하고 label `0` anomaly episode를 제거했다.
- Full scan 결과는 495 episodes, 941 atomic segments, 446 boundaries와 9 directed pair types다.
- 각 boundary 전 구간은 `H=10`보다 길며 최소 길이는 74 frames다.
- 이 audit은 데이터 구조와 관측 edge coverage를 확인하지만 annotation 생성 과정 자체의 외부 ground truth를 새로 제공하지는 않는다.

이 데이터셋을 “추가 편집 없이 직접 사용 가능”이라고 단정하는 것은 action baseline에는 타당하지만, proposed skill-linking supervision에는 **조건부**다. 구현 전에 loader smoke test와 class/edge audit가 통과해야 한다.

## Supporting evidence versus counterevidence

### Supporting

1. SmolVLA는 language/vision context와 action expert를 연결하는 공개 baseline을 제공한다.
2. LoHoVLA는 explicit subtask supervision과 joint language/action learning이 action-only VLA보다 나을 수 있음을 보인다.
3. AtomicVLA는 atomic abstraction을 action path에 주입하는 것이 long-horizon 성능과 공존할 수 있음을 보인다.
4. Long-VLA는 단계 정보를 shared policy에서 end-to-end로 학습하는 것이 separate policies보다 나을 수 있음을 보인다.

### Contradicting or cautionary

1. Compose by Focus는 정확한 sequencing만으로 해결되지 않는 visual distribution shift를 보여준다.
2. AtomicVLA의 SG-MoE ablation은 skill-specific capacity가 중요한 대안 설명임을 보여준다.
3. Long-VLA는 phase-aware sensory selection이 주요 병목일 수 있음을 보여준다.
4. LoHoVLA는 subtask completion reward를 이용하지만 proposed inference에는 그런 oracle signal이 없다.
5. 모든 관련 논문에서 training-seed variance와 paired CI 보고가 부족해 작은 효과의 재현성을 판단하기 어렵다.
6. `libero_10_subtask`에는 semantic mapping과 annotation provenance가 없어 skill ID 의미와 boundary correctness가 불확실하다.

## Exact evidence gap

확인한 primary sources는 다음 실험을 보고하지 않는다.

1. 동일 SmolVLA initialization, 동일 LIBERO episodes, 동일 optimizer/budget을 사용하고
2. 하나의 shared action expert에 작은 discrete current-skill embedding만 추가하며
3. 같은 hidden state에서 linear `{CONTINUE, next-skill, DONE}` head를 학습하고
4. skill embedding, transition head, action expert를 공동 fine-tuning하되
5. planner, language plan decoder, MoE, scene graph, phase masking, oracle completion reward를 사용하지 않고
6. ground-truth-skill과 predicted-skill closed loop를 분리 평가하는 실험.

따라서 본 연구가 답할 수 있는 가장 좁은 질문은 다음과 같다.

> Seen LIBERO-10 trajectories와 numeric subtask IDs에서, 최소 skill state와 transition auxiliary supervision이 shared SmolVLA action expert의 complete success를 개선하는가?

이 실험만으로 unseen atomic-skill composition, semantic skill understanding 또는 general-purpose planning을 주장할 수 없다.

## Requirements for `03_experiment_plan.md`

1. Dataset loader가 raw `subtask_index`를 반환하는지 먼저 확인하고 실패하면 구현을 중단한다.
2. skill별 frame/episode 수, boundary 수, directed edge count, repeated/reversed ID 패턴을 구현 전에 저장한다.
3. action-only, oracle current-skill, oracle transition, predicted transition 조건을 분리한다.
4. transition head의 효과와 skill conditioning의 효과를 분리해 해석한다.
5. action chunk가 annotation boundary를 넘는 비율을 측정하고 target 정의를 결과 전에 고정한다.
6. training seeds를 최소 3개 사용하고 동일 initial states로 paired evaluation한 뒤 paired bootstrap 95% CI를 보고한다.
7. Transition F1 향상만으로 가설을 지지하지 않고 complete success와 atomic regression을 함께 판정한다.
8. Oracle skill conditioning이 baseline을 개선하지 못하면 더 복잡한 transition mechanism을 추가하지 않고 가설을 기각한다.

## Final assessment

현재 가설은 **plausible but unproven**이다. 문헌은 skill-aware end-to-end learning의 가능성을 지지하지만, 성공 사례들은 모두 현재 제안보다 더 강한 planning, specialization, visual adaptation 또는 completion feedback을 사용한다. 가장 큰 사전 위험은 transition classification 자체보다 `(a)` numeric skill label의 의미 일관성, `(b)` annotation boundary와 50-action chunk의 불일치, `(c)` predicted-skill exposure bias, `(d)` composed-scene visual shift다.

따라서 최소 intervention을 먼저 시험하는 것은 타당하지만, 결과가 null이어도 atomic-skill composition 전체가 무효라는 결론은 낼 수 없다. 반대로 성공하더라도 seen LIBERO-10 sequence에서의 skill-state supervision 효과로만 제한해 주장해야 한다.
