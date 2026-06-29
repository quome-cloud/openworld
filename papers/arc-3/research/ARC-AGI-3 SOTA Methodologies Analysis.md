# **Interactive Adaptability and Cognitive Architectures: A Critical Synthesis of State-of-the-Art Methods on ARC-AGI-3**

The transition from passive pattern-matching to active adaptability represents the defining evolution of the Abstraction and Reasoning Corpus (ARC-AGI) benchmark series.1 While the original ARC-AGI-1 and its intermediate successor, ARC-AGI-2, evaluated few-shot grid transformation in static, offline settings 2, the launch of ARC-AGI-3 in March 2026 establishes a fundamentally new testbed for agentic intelligence.2 Rather than presenting static demonstration pairs, ARC-AGI-3 places artificial agents within dynamic, turn-based, video-game-like 64x64 environments governed by a 16-color palette.4 The agent is provided no natural-language rules, instruction manuals, or target objectives; it must explore the environment, infer the mechanics of its action space through empirical trial, build an internal world model, formulate abstract goals, and plan multi-step action sequences to achieve them.2  
This interactive paradigm directly targets fluid adaptability—the capacity to solve novel problems without relying on pre-existing crystallized knowledge or memorized shortcuts.8 To enforce this, the benchmark measures Relative Human Action Efficiency (RHAE).13 The score penalizes brute-force search quadratically through the formula:  
![][image1]  
This value is capped at 1.15 to balance out outlier luck, preventing a single unusually efficient run from distorting an entire score.10 If an agent exceeds five times the baseline human action budget, the level attempt is terminated.14 Human baselines are established via focus groups of first-time players, defining the reference as the median action count per level to prevent outlier distortion.15 Under these first-run conditions, untrained human subjects solve 100% of the environments within an average of 7.4 minutes per session.2 In contrast, standard frontier language models perform poorly on the interactive benchmark, with Gemini 3.1 Pro scoring 0.37%, GPT-5.4 achieving 0.26%, and Claude Opus 4.6 reaching 0.25%, demonstrating that scale-driven pattern-matching remains fundamentally bounded by knowledge coverage when confronted with complete novelty.14

## **High-Performance Algorithmic Implementations**

To bridge this immense gap, researchers have pioneered diverse agentic scaffolding, symbolic abstraction, and workspace optimization techniques. Table 1 presents a comparative analysis of the leading algorithmic architectures evaluated on the public ARC-AGI-3 dataset.

| System Name | Author / Developer | Architecture Type | ARC-AGI-3 Public Score | Compute / Inference Cost | Primary Algorithmic Core |
| :---- | :---- | :---- | :---- | :---- | :---- |
| baseline1 | Sergey Rodionov | Executable World Models | 63.7% 17 | $350 17 | Programmatic Python world model construction, MDL refactoring, GPT-5.5 18 |
| Vision \- CL v1 | Vansh | Multimodal Continual Learning | 63.1% 17 | $4,788 17 | Continuous vision-language weights carried across levels and games 17 |
| Read-Grep-Bash | Alexis Fox et al. | Narrative Scripting Agent | 50.2% 17 | N/A | Log parsing, shell scripting, and bash-driven environment automation 17 |
| TELL | Hesai & Waci | Dynamic Memory Aggregation | 43.9% 17 | $1,406 17 | Natural language hypothesis consolidation inside a structured markdown file 17 |
| DreamTeam | Elad Sarafian et al. | Programmatic Workspace Optimization | 38.4% 19 | N/A | Multi-agent role graph simulating and optimizing workspace artifacts 19 |
| Graph Explorer | Evgenii Rudakov et al. | Training-free Graph Search | 30/52 Levels 21 | $0 (Local) 22 | Masked status bar parsing, state-space hashing, Dijkstra frontier exploration 22 |
| StochasticGoose | Dries Smit (Tufa Labs) | Deep Change Prediction / RL | 12.58% 4 | N/A | 4-layer CNN for change prediction and stochastic coordinate-click selection 23 |
| a-evolve MAS | Zhan Shi et al. | Multi-Agent Orchestration | 12.3% 17 | $5,300 17 | Nine learned skills mined dynamically from competition logs 17 |
| OpenClaw | ARC Prize Foundation | Tool-Using Agent Harness | 5.2% 17 | $2,912 17 | OpenClaw harness adapted with memory and code execution modules 17 |

## **Verifier-Driven Executable World Models**

The baseline1 architecture represents a crucial paradigm shift, moving away from using large language models as final authorities toward employing them as proposal mechanisms inside verification loops.24 The system separates the game-play logic into a scripted external controller and a coding agent operating in an initialized Python workspace.25 This workspace contains abstract templates for transition dynamics, state reconstruction, and path planning, exposing general interfaces without any hand-coded, game-specific logic.18 The core innovation is the construction of an *executable world model*.24 Rather than learning an opaque latent simulator, the agent writes, edits, and debugs a Python codebase representing its current hypothesis of the environment.24 To encourage generalization, the controller triggers a refactoring routine whenever progress stalls, forcing the agent to simplify the code.25 This serves as an effective proxy for the Minimum Description Length (MDL) simplicity bias, preventing the model from memorizing accidental special cases.18  
The workspace provides template components for three functions: reconstructing and rendering game states from observations, implementing an executable world model of the transition dynamics, and planning actions inside that model.25 These correspond to:

* world\_model\_engine.py (transition dynamics) 25  
* world\_model\_state\_io.py (state reconstruction and rendering) 25  
* world\_model\_main\_planner.py (planning in the learned model) 25

The runtime reliability of baseline1 is maintained by three unified components: the world-model verifier, the planner verifier, and the plan executor.26 The world-model verifier continuously executes the programmatic hypotheses against historically recorded action-observation logs, ensuring that any code modification preserves historical consistency.26 The planner verifier verifies that the main planner can successfully reach completion states within the simulated code before acting.26 Finally, the plan executor manages the boundary between simulation and the actual environment.26 As actions are dispatched, the executor monitors the game state step-by-step; if the observed visual ASCII frame diverges even slightly from the programmatically simulated frame, execution halts instantly to record mismatch artifacts and trigger a model repair cycle.26

\[Inference Model Proposes Code Update\]  
                  |  
                  v  
       \+--------------------+  
       |  Code Repository   | \<---  
       \+--------------------+  
                  |  
                  \+--------------------------------+  
                  |                                |  
                  v                                v  
     \+--------------------------+     \+--------------------------+  
     |   World-Model Verifier   |     |     Planner Verifier     |  
     | (Checks Historical Logs) |     |  (Verifies Path Solves)  |  
     \+--------------------------+     \+--------------------------+  
                  |                                |  
                  \+---------------+----------------+  
                                  |  
                                  v  
                       \+--------------------+  
                       |   Plan Executor    |  
                       \+--------------------+  
                                  |  
                         
                                  |  
                                  v  
                       \+--------------------+  
                       |    Environment     |  
                       \+--------------------+  
                                  |  
                        
                                  |  
              (If Mismatch: Stop Execution & Halt Run)

By restricting environmental interaction to confirmed, verified plans, baseline1 prevents action-budget waste.18 The evaluation runs under strict recorded playthrough conditions: the process starts from a fresh workspace, target environments are exposed only once, and no previous files, logs, or external parameters are carried over.25 Furthermore, the system is audited against environmental leaks.18 The container isolates the agent from internet services except via secure proxies, closes Codex web search, and masks game-identifying names, process arguments, or references to ARC within environment variables.27 This structural encapsulation guarantees that success reflects genuine online adaptation.27

| Underlying Model Engine | Games Fully Solved | Mean Per-Game RHAE | Primary Computational Constraint |
| :---- | :---- | :---- | :---- |
| GPT-5.5 (High-Effort Thought Configuration) | 15 / 25 | 58.12% | Restricted Weekly OpenAI Codex Token Limits 18 |
| GPT-5.4 (High-Effort Thought Configuration) | 8 / 25 | 41.29% | Scaled Inference Budgets per Level 18 |
| Claude / GPT Configurations (Older Versions) | 7 / 25 | 32.58% | State-mismatch tracking limitations 25 |

## **Programmatic Workspace Optimization**

The DreamTeam framework by Sarafian et al. addresses the fundamental constraint that modern frontier language models cannot adapt their parameters during test-time play.19 To overcome this, the authors introduce *workspace optimization*, which adapts the structured external context—prompts, files, and executable code—that the frozen model reads and writes.19 This optimization paradigm maps traditional parameter training to a structured workspace through typed analogies, utilizing artifacts as parameters, interaction logs as training data, prediction failures as loss functions, and textual feedback as gradients.19 DreamTeam instantiates this optimization as a decentralized multi-agent role graph, partitioning the workspace into six interlocking surfaces.19

                     \+--------------------+  
                     |    Team Leader     |  
                     \+--------------------+  
                               |  
            \+------------------+------------------+  
            |                                     |  
            v                                     v  
  \+--------------------+                \+--------------------+  
  |      Observer      |                |      Prober        |  
  \+--------------------+                \+--------------------+  
            |                                     |  
            v                                     v  
  \+--------------------+                \+--------------------+  
  |     Simulator      |                |  Strategy Library  |  
  \+--------------------+                \+--------------------+  
            |                                     |  
            \+------------------+------------------+  
                               |  
                               v  
                     \+--------------------+  
                     |       Critic       |  
                     \+--------------------+

These six roles operate sequentially to form a closed-loop system.19 The Observer transductively parses incoming visual grids into structured state representations.19 The Simulator predicts transitions and maintains a summary history to assist the team when navigating partially observable elements.19 The Strategist (Information Extraction module) maintains a strategy library of behavior policies and sub-goals used when the dynamics are still immature.19 The Prober generates directed, information-seeking action sequences to explore unmapped transition regions.19 The Critic captures prediction mismatches and code-regression breakages, formulating precise credit-assignment text that directs failures to the specific artifact responsible for the error.19 Finally, the Team Leader reviews the system state to decide whether to plan, probe, or initiate code repair.19 This systemic delegation optimizes the workspace dynamically, achieving a score of 38.4% RHAE while reducing physical action usage by 31% compared to basic programmatic baseline systems.19

| Workspace Surface | Role Owner | Inductive Programmatic Output | Transductive Operational Output | Primary Target Feedback Path |
| :---- | :---- | :---- | :---- | :---- |
| Observation Model | Observer | State-rendering verification: render() 19 | Per-step structured state extraction: ![][image2] 19 | Simulator transition engine input 19 |
| Dynamics Model | Simulator | Programmatic transition engine: predict(), history() 19 | Mapped state summary history: ![][image3] 19 | Observer parsing updates 19 |
| Strategy Library | Information Extraction | Dynamic sub-goals ![][image4] and local policies ![][image5] 19 | Multi-step imagined plan rollouts 19 | Team Leader goal arbitration 19 |
| Probe Context | Target Exploration | Action sequence verification parameters 19 | Target coordinate-seeking action sequence 19 | Critic transition analysis 19 |
| Failure Routing | Critic | Regression set verification suite 19 | Directed programmatic repair directives 19 | Strategist, Prober, and Team Leader 19 |
| Goal/Action Context | Team Leader | Executable task-stopping criteria 19 | Chosen plan action sequence or default exploration | Environment dispatch controller 19 |

Evaluation robustness in DreamTeam is managed by executing regression verification.19 Whenever the coding agent proposes a program patch, the harness replays a regression dataset consisting of historically observed state transitions.19 If a transition that was successfully handled by a previous iteration fails under the patched code, the system flags a regression breakage.19 This mismatch is captured as a concrete counterexample and immediately routed back to the specific role holding field ownership over the violated workspace schema, forcing precise local code debugging instead of global catastrophic forgetting.19

## **Systematic Graph-Based State-Space Mapping**

The graph-based exploration strategy proposed by Rudakov et al. serves as an efficient, training-free baseline, demonstrating how the deterministic rules of ARC-AGI-3 allow the environment to be mapped directly as a directed graph.21 Operating entirely without neural updates or offline training datasets, the methodology models spatial states as nodes and control actions as directional edges.22 The system operates through two principal modules: the Frame Processor and the Level Graph Explorer.22  
The Frame Processor processes the visual grid by segmenting it into single-color connected components to isolate functional elements.22 Crucially, the processor detects and masks the status bar to prevent changes in UI step counters from registering as novel environment states, compressing the unique state space.22 It then hashes the masked grid to generate unique, cryptographic identifiers for each state, storing adjacent nodes and untested edges.22

                     \+--------------------+  
                     |    Visual Frame    |  
                     \+--------------------+  
                               |  
                               v  
                     \+--------------------+  
                     |  Image Segmenter   |  
                     \+--------------------+  
                               |  
                               v  
                     \+--------------------+  
                     | Status Bar Masker  |  
                     \+--------------------+  
                               |  
                               v  
                     \+--------------------+  
                     |   State Hasher     |  
                     \+--------------------+  
                               |  
                               v  
                     \+--------------------+  
                     |Level Graph Explorer|  
                     \+--------------------+

The Level Graph Explorer maps the state space as a directed graph where nodes represent unique grid hashes and edges represent transitions.22 In coordinate-click environments (where the raw action space is ![][image6] pixels), the explorer groups potential interactions into prioritized salience tiers based on morphological cues, ignoring masked status bar regions.22 The explorer then executes a hierarchical navigation policy:

                  |  
                  v  
       \+--------------------+  
       |   Tested Actions   |  
       |  in State s under  | \---\> (Yes) \---\>  
       |    Priority p?     |  
       \+--------------------+  
                  |  
                (No)  
                  v  
       \+--------------------+  
       |  Reachable State   |  
       |  s' with Untested  | \---\> (Yes) \---\>  
       |    Actions under   |  
       |    Priority p?     |  
       \+--------------------+  
                  |  
                (No)  
                  v  
    
                  |  
                  v  
     

This structured exploration was evaluated across both public and private holdout environments under a strict cap of 4,000 interactions per game.7 The method successfully completed a median of 30 levels across six games, significantly outperforming standard LLM-based agents.21

| Game Environment | Operational Control Type | Level Index | Median Step Count | Success Rate (Over 5 Runs) |
| :---- | :---- | :---- | :---- | :---- |
| ft09 (Public) | Click-Based Coordinates 22 | Level 1 | 125 | 1.00 22 |
|  |  | Level 2 | 177 | 1.00 22 |
|  |  | Level 3 | ![][image7] | 1.00 22 |
|  |  | Levels 4–10 | Unsolved | 0.00 22 |
| ls20 (Public) | Directional Arrow Keys 22 | Level 1 | 124 | 1.00 22 |
|  |  | Level 2 | ![][image8] | 1.00 22 |
|  |  | Levels 3–8 | Unsolved | 0.00 22 |
| vc33 (Public) | Click-Based Coordinates 22 | Level 1 | 9 | 1.00 22 |
|  |  | Level 2 | 7 | 1.00 22 |
|  |  | Level 3 | 36 | 1.00 22 |
|  |  | Level 4 | 321 | 1.00 22 |
|  |  | Level 5 | 287 | 1.00 22 |
|  |  | Level 6 | ![][image9] | 0.80 22 |
|  |  | Level 7 | ![][image10] | 0.80 22 |
|  |  | Level 8 | 917 | 0.80 22 |
|  |  | Level 9 | Unsolved | 0.00 22 |
| sp80 (Private) | Hybrid Click & Key Arrows 22 | Level 1 | 227 | 1.00 22 |
|  |  | Level 2 | ![][image11] | 1.00 22 |
|  |  | Level 3 | ![][image12] | 0.40 22 |
|  |  | Level 4 | Unsolved | 0.00 22 |
| lp85 (Private) | Click-Based Coordinates 22 | Level 1 | 143 | 1.00 22 |
|  |  | Level 2 | ![][image13] | 1.00 22 |
|  |  | Level 3 | ![][image14] | 1.00 22 |
|  |  | Level 4 | Unsolved | 0.00 22 |
| as66 (Private) | Hybrid Click & Key Arrows 22 | Level 1 | 39 | 1.00 22 |
|  |  | Level 2 | 44 | 1.00 22 |
|  |  | Level 3 | 123 | 1.00 22 |
|  |  | Level 4 | Unsolved | 0.00 22 |

The level-by-level performance highlights the emergence of a "compositionality cliff".4 While early levels (acting as tutorial environments introducing isolated rules) are solved with minimal action counts and high success rates, later levels that layer multiple learned transitions together exhibit exponential growth in exploration steps, occasionally resulting in failure.8

## **Deep Change Prediction and Memory Scaffolding**

Approaching the interactive exploration challenge from a parametric perspective, the StochasticGoose agent developed by Tufa Labs utilizes a reinforcement learning model to predict environment changes.23 The model employs a 4-layer Convolutional Neural Network (CNN) containing 32 to 256 channels to process 16-channel one-hot encoded visual grids.23 Rather than directly predicting completion paths, the network optimizes a binary cross-entropy loss to evaluate state-transition outcomes:  
![][image15]  
where ![][image16] is the binary frame-change label, ![][image17] is the estimated change probability, and ![][image18] represents light entropy regularization to promote exploratory diversity.23  
The CNN backbone routes outputs to two distinct heads: the Action Head, which estimates change probabilities for standard directional controls (ACTION1–ACTION5), and the Coordinate Head, which predicts spatial click probabilities (ACTION6) using 2D convolutional layers to preserve spatial inductive biases.23 The model trains continuously on a deduplicated experience buffer of 200,000 unique state-action pairs.23 When transitioning to a new level, the experience buffer is cleared and model parameters are dynamically reset, preventing stale mechanical associations from interfering with novel rules.23 The exploration policy relies on stochastic sampling biased toward actions with high change probabilities, achieving a score of 12.58% RHAE on preview sets by successfully minimizing unproductive search actions.4  
Beyond program-synthesis engines and graph search, other leading community models exploit the trade-offs of continuous memory and natural-language structuring.17 Vansh's Vision \- Continual Learning v1 integrates a multimodal agent that maintains continuous neural weights across sequentially traversed levels and games.17 This permits the retention of high-level visual representations and behavioral priors, achieving a competitive 63.1% RHAE, though requiring a substantial financial and computational footprint of $4,788 per full run.17  
In contrast, the TELL agent by Hesai and Waci represents an extremely cost-effective approach to semantic state-tracking.17 TELL operates as a single-conversation agent that records, compounds, and updates its understanding of game mechanics within a persistent natural language file called MEMORY.md.17 By continuously editing this text-based knowledge base and executing plans through structured prompts, TELL achieves 43.9% RHAE at a fraction of the cost ($1,406).17 Finally, the Read-Grep-Bash agent achieves 50.2% RHAE by utilizing automated bash scripts and python-based search tools to parse real-time environment logs, proving that direct, program-level access to trace logs can compensate for the lack of model parameters adaptation.17

## **Analytical Syntheses and Cognitive Directives**

Analyzing the collective performance of SOTA agents on ARC-AGI-3 reveals critical insights regarding cognitive adaptation in novel environments. First, programmatic world-modeling approaches like baseline1 and DreamTeam demonstrate a profound advantage over standard latent-dynamics reinforcement learning.19 While state-of-the-art model-based RL agents require thousands of environment transitions to construct a reliable latent world model, symbolic programmatic models can generalize immediately from a handful of transitions.19 By defining transition rules as explicit, inspectable Python code, the model-space search is constrained by the syntax and semantics of programmatic execution, facilitating fast few-shot generalization.24  
Second, the success of these systems illuminates the concept that the scaffolding is the core locus of intelligence.14 When human engineers construct a bespoke, game-specific harness to solve a particular environment, the actual generalization does not reside in the model but in the human's design.14 To achieve true, robust adaptability, the cognitive scaffold itself must remain strictly game-general.18 The frozen model must be forced to program, debug, and optimize its own execution harness dynamically in response to environment surprises, transitioning the locus of intelligence from static weights to a self-evolving external runtime.19  
The collective performance of these systems on ARC-AGI-3 provides crucial architectural directives for the design of future general-purpose agents. This analysis synthesizes three primary takeaways for researchers developing cognitive frameworks.  
First, general fluid adaptability requires a departure from scaling parameter-space knowledge toward scaling inference-time verification.2 As confirmed by frontier model evaluations, scaling training data merely densifies knowledge coverage, leading to catastrophic failure when encountering out-of-distribution environments.2 Future agentic intelligence should be evaluated on its adaptation efficiency—how quickly and with how few physical environment interactions it can synthesize, test, and repair its internal beliefs.11  
Second, cognitive architectures must treat their external workspaces as trainable substrates.19 In deterministic, rule-governed domains, much of the relevant structure resides outside model parameters, in tool schemas, state files, and execution logs.29 By implementing workspace optimization, where model outputs are routed as precise patches to specific, typed files, systems can achieve robust, online learning without incurring the cost and instability of continuous weight fine-tuning.19  
Finally, programmatic world modeling represents the most promising path forward for low-data, few-shot tasks.24 The union of large language models acting as intuitive, approximate proposal engines with exact programmatic interpreters acting as verifiers provides a reliable foundation for agentic autonomy.24 By developing architectures that maintain, verify, and refactor executable codebases representing environmental dynamics, researchers can construct agents capable of navigating unexpected real-world workflows, APIs, and edge cases with human-like efficiency.26

#### **Works cited**

1. Leaderboard \- ARC Prize, accessed June 25, 2026, [https://arcprize.org/leaderboard](https://arcprize.org/leaderboard)  
2. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence, accessed June 25, 2026, [https://arcprize.org/media/ARC\_AGI\_3\_Technical\_Report.pdf](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf)  
3. ARC-AGI-1 & ARC-AGI-2 Guide \- ARC Prize, accessed June 25, 2026, [https://arcprize.org/guide/1](https://arcprize.org/guide/1)  
4. The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2603.13372v1](https://arxiv.org/html/2603.13372v1)  
5. \[2603.24621\] ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence \- arXiv, accessed June 25, 2026, [https://arxiv.org/abs/2603.24621](https://arxiv.org/abs/2603.24621)  
6. ARC Prize 2026 \- ARC-AGI-3 | Kaggle, accessed June 25, 2026, [https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)  
7. What is ARC-AGI-3? \- ARC Explainer, accessed June 25, 2026, [https://arc.markbarney.net/arc3](https://arc.markbarney.net/arc3)  
8. ARC-AGI-3: The New Interactive Reasoning Benchmark | DataCamp, accessed June 25, 2026, [https://www.datacamp.com/blog/arc-agi-3](https://www.datacamp.com/blog/arc-agi-3)  
9. ARC Prize 2026 \- ARC-AGI-3 Competition, accessed June 25, 2026, [https://arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)  
10. ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2603.24621v2](https://arxiv.org/html/2603.24621v2)  
11. ARC-AGI-3, accessed June 25, 2026, [https://arcprize.org/arc-agi/3](https://arcprize.org/arc-agi/3)  
12. What Is ARC AGI 3? The Interactive AI Benchmark Humans Solve at 100% | MindStudio, accessed June 25, 2026, [https://www.mindstudio.ai/blog/what-is-arc-agi-3-interactive-benchmark](https://www.mindstudio.ai/blog/what-is-arc-agi-3-interactive-benchmark)  
13. ARC-AGI-3 Launch: SOTA Models Score Under 1% and the Human Baseline Is Rigged, accessed June 25, 2026, [https://adam.holter.com/arc-agi-3-launch-sota-models-score-under-1-and-the-human-baseline-is-rigged/](https://adam.holter.com/arc-agi-3-launch-sota-models-score-under-1-and-the-human-baseline-is-rigged/)  
14. AGI Is Not a Compute Problem. ARC-AGI-3 Just Proved It. | by Siddhant Nitin Patil, accessed June 25, 2026, [https://pub.towardsai.net/agi-is-not-a-compute-problem-arc-agi-3-just-proved-it-950fa3b1b241](https://pub.towardsai.net/agi-is-not-a-compute-problem-arc-agi-3-just-proved-it-950fa3b1b241)  
15. Measuring Human Performance on ARC-AGI-3, accessed June 25, 2026, [https://arcprize.org/blog/arc-agi-3-human-dataset](https://arcprize.org/blog/arc-agi-3-human-dataset)  
16. When Artificial Intelligence Confronts the Unknown: ARC-AGI-3 and the Future of Adaptive Intelligence \- Human Capital Innovations, accessed June 25, 2026, [https://www.innovativehumancapital.com/article/when-artificial-intelligence-confronts-the-unknown-arc-agi-3-and-the-future-of-adaptive-intelligenc](https://www.innovativehumancapital.com/article/when-artificial-intelligence-confronts-the-unknown-arc-agi-3-and-the-future-of-adaptive-intelligenc)  
17. ARC-AGI Community Leaderboard, accessed June 25, 2026, [https://arcprize.org/leaderboard/community](https://arcprize.org/leaderboard/community)  
18. \[2605.05138\] Executable World Models for ARC-AGI-3 in the Era of Coding Agents \- arXiv, accessed June 25, 2026, [https://arxiv.org/abs/2605.05138](https://arxiv.org/abs/2605.05138)  
19. Workspace Optimization: How to Train Your Agent \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2605.09650v1](https://arxiv.org/html/2605.09650v1)  
20. \[2605.09650\] Workspace Optimization: How to Train Your Agent \- arXiv, accessed June 25, 2026, [https://arxiv.org/abs/2605.09650](https://arxiv.org/abs/2605.09650)  
21. \[2512.24156\] Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks \- arXiv, accessed June 25, 2026, [https://arxiv.org/abs/2512.24156](https://arxiv.org/abs/2512.24156)  
22. (PDF) Graph-Based Exploration for ARC-AGI-3 Interactive ..., accessed June 25, 2026, [https://arxiv.org/html/2512.24156](https://arxiv.org/html/2512.24156)  
23. DriesSmit/ARC3-solution: My submission to the ARC-AGI-3 Developer Preview Agent Compitition. \- GitHub, accessed June 25, 2026, [https://github.com/DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution)  
24. Executable World Models for ARC-AGI-3 in the Era of Coding Agents \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2605.05138v2](https://arxiv.org/html/2605.05138v2)  
25. Executable World Models for ARC-AGI-3 in the Era of Coding Agents Project code will be released publicly after acceptance. \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2605.05138](https://arxiv.org/html/2605.05138)  
26. Executable World Models for ARC-AGI-3 in the Era of Coding Agents \- arXiv, accessed June 25, 2026, [https://arxiv.org/pdf/2605.05138](https://arxiv.org/pdf/2605.05138)  
27. astroseger/arc-3-agents-baseline1 \- GitHub, accessed June 25, 2026, [https://github.com/astroseger/arc-3-agents-baseline1](https://github.com/astroseger/arc-3-agents-baseline1)  
28. Field ownership across the workspace. Each field appears in exactly one... | Download Scientific Diagram \- ResearchGate, accessed June 25, 2026, [https://www.researchgate.net/figure/Field-ownership-across-the-workspace-Each-field-appears-in-exactly-one-schema-and-is\_tbl3\_404752776](https://www.researchgate.net/figure/Field-ownership-across-the-workspace-Each-field-appears-in-exactly-one-schema-and-is_tbl3_404752776)  
29. Adapting the Interface, Not the Model: Runtime Harness Adaptation for Deterministic LLM Agents \- arXiv, accessed June 25, 2026, [https://arxiv.org/html/2605.22166v2](https://arxiv.org/html/2605.22166v2)  
30. ARC Prize 2025: Technical Report \- arXiv, accessed June 25, 2026, [https://arxiv.org/pdf/2601.10904](https://arxiv.org/pdf/2601.10904)  
31. Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3, accessed June 25, 2026, [https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAXCAYAAAA7kX6CAAAAzklEQVR4XmNgGAXDFdij8d3Q+BiAB4j/A/EvKH0NiN8BsRWyImygFYjVgJgRiL2AeB4QZ6CowAE4kdjCQHwWiU8U0ADiGHRBKDgOxProgiBgCsQP0QWRAMjvcuiCrkCsjSZ2D4jjgdgciC2BeAcQWyArsAHib0BcBsSeQCwIxMEMEKeBAmsFEB8F4v1QNhwsgdKgQNnJAHHSTYQ0GOxF44MBKxKbiQHiV2TADsTf0cSIAg0MEFeAQAJCmDBYCMTPgVgBiPegShEGIOcOIAAACy8eodKj3B0AAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAA/klEQVR4Xt2TMWoCURCGR0Qs0luJaJuIFhHUnMBjeAAhJwikzyE0EDCFaGuhaCNprO1tFISEBJF00X+YyTq+atRG/OCD3e+9HXy7SHRVxGEJ1mAqWHPxCLfqfbDmpk0y4GSWdOYAfvgzjMfAA7rwBj6QvFg3dyQDfmBa27c2Fw2SzU3TxtpcdOAUJkybkXNAnmRjIejc1kFjPmDRhv+fH7NR21vQGO4ZG1YaLTz0i+RILfgCy7AK+7AS7SR5OBzQI/mkzIbky7zDCRzpdcQffLIB5OACzunwaENzHXEbBoVfbtbcJ+GvuT+aZ9oftb7Pfl5J/nBZODhc8sPHuCB2rTA1Q7uog4EAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAABGUlEQVR4XmNgGAX4gD4Q7wHiu0D8EogXokpjB+xAPAeIVwKxPRAzA7EqEP+HYoLgNBB/RxcEgvcMRBjQAMSr0QWBQJABYqguugQyMADiv0CsjC4BBF4MkPDACz4yEOFEfIBQIDkAcTu6IDLAZwA3A0RuCpo4GzIHnwH5DBA5fySxBAaIt+EAZoAUsiAQ7IWKr0UT/wXEU5EFXjBAFD4CYicg5gLiQKgYCEdD1WkAcRhUbBoQy0LFwcAOiL9CJRcwQKLvKJSPDi6jC8AAIxDzIPFBmn8i8UGAkwEzQBn4gLgcSsNANhDPQOLDQBMQi6ELwtL6NwaIsw9C+SzIiqDgEBIbwyAfIPZGF0QDuQwQl2F4gxQggC5ANgAAJn89RzjTAOoAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAuElEQVR4Xu2TwQoBYRSF70oWLHkjKWXFe9jIJBaWFE/gYZSVrdcgxg4pHDP31+3EpEvZzFdf8885M2czjUiOhwqsO3xJE94cZtKR9KEeF8pHI4F8LOH/YxEXStZYgYMw1udCeTe2hgsOvWNXOOIwjA24UHisBWfwBKewarrn2NCGBh4LLO3N49+M4Q5u9XqGbUk/xhHu4Ub7A7wkb4o0YE3PXzOBJQ69rMy5aM4uunAOx1x4KXPwE+4ufFELgqYe2wAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJAAAAAZCAYAAADe+aeoAAAEmUlEQVR4Xu2ZSchcRRDHyxjjvkBccI3ighdXREEUBAWPHvQgYg5qYhYX0BiCHsx3UkFBcFfcDoIYchLcQAVRBEPAgFtUVFTc9w1xt35U9zf9la/fdA/MN6P2D/7M6+r35tXUq+5X3SPSaDQajUaj0YBjVHepZpy9i+tVX3jjPLJUdaNqmWp71+c5ULXGGycIsRuVk1RXqi5RHeb6ItuorlXdrdrD9Xlq4piFAP+kOju0T1D9otpl9oy57Kz6S/Wn75gHSATuu3to3yzmS46VYv0EdBp4R/r9zbGjarPq5MR2i+qKpA3vydzvf0z1R9KO1MYxywLVJrFMjDwi9mXMSF3cJpNLIO77pmv3/fDvxAI4LQn0q/T7m+MC+ed1zC4/qw5NbJzzTNI+Ntj8ZFAbxyyfSN2F76oukvIEWqi6zxsd53lDhlWq31W7Jjam692SdspqsZFLEk06gXiAxI4ZoybekdwDxvZ8OD43tHnFpWxVbUjatXHsJedYjifDZ2kCAXXV/t4YuFz1lTdmYGR9HI63E0uOHIeIJQ5MQwLdIRa7cSQQyQAzoX3ZbK+xRWyiiNTEcSjRscfFprvTxYpjHkAKdQ8jKFKTQHCv6n1nW6c6ztlyUDByT+qA18VGNO9vbOuT8yJvySAwpQlEEXlGpUogdhSpMI4EivazwvHMbK/xTbAvlvo4DiU6sFdi40e+LVYfRW5VPZW0axOIKfIeGawcrlF9Oegeyqky8PW6xE6y+8AuUZ2StEsTiGSmLugSr4E3xIL+mupV1St22VCIXWScCbRnOKa4jhwVbOgIqYtjEV2OkanYHg3tnWQwgiL01yRQhAL8B8kX6DmOF7vnh87Ow8FOP1wsNvuklCbQOGAgprEbZwIBS3faxJci+1mxhGeVDaVxLMY7ADsE20ehzRLPM2oCMTp+FBsZNRwgds/nnP2GYF8e2kzX6ewDk0ygdNaGcScQrFD9pvpU7PX5gdjsCaVxLKbLgUXBFjcKeeAcfyZWjMWVG+KYjagSeI1RvB0p9n1Hz+3uBZ8oFp9wdqZh/GBjDViyU5QTvOgric6sx/E+4bz5IsYu+vK9DOKWFrbD6HpOgK1rIDMJEG9gqf9QOC6NYzEkhXcsZmnfjmnO8RzUP37avFrqXmWXim3EpVCc40tux5VNUvpLZiCCfmalamFLw8cbKGQpgnO8LN3XYbsqafNMXkrabBj660aJYxZmD38D9hMYOX4llpIbEV0wEljBdX0fhTSrvxL2FQvQ4YmN4rbPj4PE+tf7jglxv3T7G4vYdOGSQpL46/YWe1UdnNg4Jx3YLAD4VyFllDj2woPlYrL8cxkUXF0wHX4t9oN5+LwaKNxyLFQ94I2O872hh7Wqb1UbVS+qnhZbfXRBMc3rDF/5pD7ab84Z8we/kdgRM/whhsQucruYjxcmNg/bEiTMg6oXwvG26Qliv5nX98Nivzf+PeWpiWMRJ6puEtu9ZRU2zbCDeqf0v2L/razwBgd/W1DoniM2A3nYGGRnn4XPsIL4vxzH/yWneUOjUQrLbfaMGo2RYEsj/XOz0Wg0Go1GozGF/A351mxBnqAjewAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAABwklEQVR4Xu2WSytFURTHl1cYIEIkZagMlZFSlIGPYGLgA5gbMDBmYCgyEOUxkLwGyugaeJVQHhkwF2VAEf6rfY7WWXede6/RPaf2r36D/d/7nu5Zd9+1N5HH4/F4EkUzzMBZOAN/4DcslYty8AH3YT9chZPR6XSzAw/EmIvEBZoWWRyD8BiWiIw/OyTGqYZfhpVMBdm4yiUd5HZYrcrfKft5qcUqzkiQLatcMkHZn2OeyM41/Fes0aGgCXbqMAmskXvBYT0hOCW7COdk55pKuAfr9QS4gks6TAKN8BU+wHI1J7khuwgnZOcWVfCMogXiHbMCy0SWGL7InUD5uCO7CNygrTyOanIHAhf1Em5Ep5NDK3yDA3rC4IjsIvxn54RwgQ7hJqxQc4lgFz6KcRfsFWPNAtlFuCc7j4N/kFu4DrfJ9bLEwV+wXYznKPddJzzR9C/9EuSFwr2L/0rc37hJ872rLrKiiIySexneCfNwkdwRzlmfWMdjvtdItuCYyngd95188M38muymz7uYb+1F55PcC1m2iHVhJuGT5Rm2BeMGcs/r/lsRzwW5U8ki3EHci1INH7k95G7VhTRyj8fj8Sh+ASu8ZelV6xSYAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAABxElEQVR4Xu2WyytFURSHV96PJHmUoYFSSsnAXEpGJgYiE8Yyk1L+AZGYkSQxYaC8S0wYIXkVhYkRSmQkwm+17tU6y7333O7kHOyvvrr7t/apu9fZZ59D5HA4HI5QkQM34Rxsh/fwE+bpSXEog1dwDB6RXDcJ0/Sk38w4fFbjJpJFDqksHqswP/I7naRJfO3w94xfzh3Jgioi4yz4GslKopNi0E8yZ9TknLF/gja4p8b1JIvjxysRvSTz5k2ebHM2YIENFaWwyoZBc0myOD6LUiHZ5mTDdVhkC+AMztowSHpIzpkX2GlqycKPITfm2hbiwDfgkLwN4h3Du5HPsNAxQ7LAGlvwoQ6+wxZb8CEXbsF9eAoXveVwwa/hG/gAi00tEbckuy4VuEE7cAlmmlqgNMMGkw2Q7B6+m35kwDVYqbJq9duPcpJzbgGuwANvOVhiHaCDkWzX5LGYIlmcZsKME3FB8ihxk/mQ5m+nQs+MAOEmPJrsBH7AWpXxPM403XCZpEHsNMkr2jY7Fvx1fU7SFAvvRP15ERhPcFuN+0gW16Uyxu6wRvimcqsfx/TzGylKdAfxWRQKWuEI7KAQ/SmHw+H4D3wBc7BiqJkdeNEAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAAB2klEQVR4Xu2XyytEURzHf3lbIIryyE4RFqxQLGWh2Ehkz8rW/0DyKLGTlA3KwmtBbGw8FkJeC4WtDaWU8P11zsyc+5szc6/N3Jk6n/rU3O95NOfMPY8hcjgcDkda8gzf4ACcgtPe4oR8wSc4CC/hsLc48+mCj7BOP4/Ab1gYrWFnCD7AHP2cCz9hS7RGhjMLf0XWB5dEJpkg1a5N5Jz9iCxjeaf4yQnCCql2TSLnLEh/+7BIhgblsF6GqSYymAX4QWoPOffUsLNFqp1cQkEnJx/uwVJZAK7hqgzDIDKYRVJftAG+wGazkgXetLldp8iDTg5TAC/IO0H8xqzBbCMLDdtgGnXWI3LJMjw0novJ3l8yeNM/gGfwCm54i8PFNphanW2KXMJ7Btebg+2klomtPz94go5ILVU+8dIG22DKdMbH+3+x9ZeMSngP1+E2BdvvUoZtMDU6Oxa5JEsGpNrxnhWUO1JLie9KvEnvwBJPjRCZpPjJGSV1W64wMq5j3l/y9POpkbWSGiy/eX5w3zcUu0Ca7MITGYYB/1p8YvDFj6kidevtjdZQ2N6wW9itP/Mm/gqrY8VJ4b8afCrZiLxBfjf0lNEBZ+AY2e8eiRiH87Cf7MvM4XA4HD78AS9EZ7ZlBQ05AAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAABfklEQVR4Xu2Wu0oEMRSGD15QCxELRRGsbQVtLOxsbERrKzt7n8EH8BUELbwW3kBBBa3UQrygTyCiiI2NgvofzgRnfmeX7DTZHfLBx27+JAtzNpNEJBKJROqDfg4q0M1B2dmAPxxW4FRsbJ76O6XDPZwPT/K/KM6x1LhSsAoPxa84I/CDQ7AFjzlsdKbgs/gXZxaecQhe4QCHORzATg5T9MAhDkOxmXz6FodphhccVqEN7kv+pn4LlzkMxQTsS74XLc6C1D6vHV5JtkC6YlbEih2cJbF/0FG0ODpnnkMPOuCR2Kq7gevZ7rC8w8FUu0hxmsTmjHOHJ1og3cS3YSv1BWNU/o7er8RvavuwCF849EQvnI9wDe7Ay2x3OPRdnyHfxIrj2j6cwHsOPXkQe5VaxDbpXdiVGVFHuIsdo5muKkaPW+3LO9ar0QvvxIrC7MFzDkMyDKfhp9jDzondfRzu9WP0Jqy57hm1cC12KuXhVpDuRQ3PpNgRHIlEIpEC/AJuxluF4cfaNQAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAAB3klEQVR4Xu2WSygGURTHT96UJI9iZ8eCyEKRjQUpZUOJbKxsbJWUsiWPkthKlFhIniU2ZIGFVxFZ2HnmsVLC/3Tm051jvuFbzUf3V79y/+eQe2bmzhBZLBaLJapIgmtwCjbDG/gBU8wmH4rhFRyAjfDO+flfMAKfjHUNyXD6jcwP/t1zY/0C34z1n+aaZBh5zjoBvjpZZqgpDMPwQGUXcFxlf5YmuG2sy0gGw4+XH4nwGXbpwi9Zhak6NMiC+ToMmjOS4fBZ5McQSV81HIWHJHdcgdnkAw93BabrAjiGkzoMkg6Sc4bPjFZV82KGZDhbcMzJeDD8mBaGmn6AL8A+uQfEd8w0jDWyqGGCZNNFuqDYIenrUTln7G9JhutwFx7BOXc5uoiBl/AWZqiayQLJEPj1bxLpcBge0Cach/GqFii1sEpl3SQb5KsZjl6SnlKVRzqcHJJzbhYuwj13OVi8NsOPSug8CUcFSU+lyr3+nh+nJI9SHMkhvQTTXB0Bwht5UBm/ed5hiZFxH2cmffT9rcJ99yrzIhuekAxFs0zuz4vAeIQbxrqTZINtRsZ43RF8pTmrd9a5JAOs++oID3888lvJi9AdxGdRVNAAB2ELRf5PlZN867TrgsVisVh+5hM5F2gOpyXdOgAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAAB3ElEQVR4Xu2XOy9EQRTHT7xJEM9EVDoJkdCg0CiITiHxikZPSSHxBSQSfAARoaFQeCYeDY1H4RWPgoIGiRCJRIP/ydm1c0/u3t1b3bsyv+SX7PxndrNzdmbuLJHFYrFYQkUO3IILsA8+wx+YZw7y4Avew254SvIZ/4YZ+G6020mKM2Fk8eiFtzAj0s6En7D+b0SK80RSjKpIO4tkNXBWGh3kwijJmCaVc/atspSlBx4Y7UaSCfL28mKOZFytyjljE7EJ83VoUAardRg0NyST47PIixWScXoLJVucbLgBi3QHuIDzOgySIZJz5gMOqD43eCwXoUXlyRaH4R/ghJwF4hWzCNONLDREt0ud7nBhFu4Y7QLyVxwmF27DI3gOl53d4SIN3sEXWKL6NHxmcCGmYDPJNvFbHIYLtEeyVfmJFxo6YKvKxkgmyL+mX/wWp4LknFuCq/DY2R0sbpMZj2T7KtfwKtPw+x506ME1yVbiuxIf0muw0DEiQHgyryo7I7mrmE8iHmfeX/g+xO1DI2sgmWyxkcWjHF5S7AJpsk7O60VgvMFdoz1CUohBI2PcVtgVbIu8roGPsDLW7Qn/1eCnkhvRFcRnUSjogpOwn/x9qWE4DTvJfZtZLBaLJQG/7JxksyNkajIAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAABwklEQVR4Xu2XzStEURiH33xbIEIke8WGFUUWytpObCzsbZWtpSiWIkkssLDAWJCVjY+FkI+SxB+glIjwezt3zHvfzr0zs5ozOk89Ned3TtPcd87XJfJ4PB6PU9TDIzgHZ+AP/IYFclAMH/ABDsJzOBzuzm924L5oc5G4QNMii2II3sGioF0M32D734g8hwvBSiaDbELlknEyYzpVnpx5/wJbcUaCbFXlkmUyY9pUbvs+G3uwQoeCOtiiQxdYJ/OAcfvHFpkxegllWpxSmIDVugNcwhUdukAtfIH3lNpLbEyRKUKPyjMtDlMGzyhcIJ4xa7BQZM7wBd91GMESPBDtSsquOEw5mQPhBF7AzXC3OzTCV9inOyLgPYMLMQu7yCyTbIvDcIEOySxVPvGcYxc+inYr7BbtTMm2OPyH3MINuA1Pw91uwD+wWbTnKf1dx3ZR5MI86TCGGzJLifc33qT53lUVGpFDRsk80CJcILOP8BHOWa8Yx215fykJ2sci6yDzsDUii4Jv5ldk3/R5FvOtPed8UmopaBvEONtyuYb9wWdehs+wKdUdC79q8KlkIzmDeC/Ka8bIvHIMkH2ZeTwejycNv9JuZavzavs3AAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAWCAYAAACSYoFNAAABbklEQVR4Xu2Wuy5GQRSFd1yCiIiCRDwAD6BSeACJeAGlRq9X8SMqryARCULhVkiQUKEQl/AECiqJ0mXtbCc5sxwnMxTnkvmSr1l7mrMyZ2ZEIpFIpFr0cFA3OuEgXIGvcM4d53IKP39xK7WusqyLfcwx/JCwcp7kZymJI6l1teBd/MsZhm8cgm2xomtHSDmT8IxD8AIHOMzgEHZxmKIXDnFYJCHlMM3wgsMc2uCBZB/qt3CVw6L5TzkzYmdNCO3wStyCdMesiZVdKrScBoeeaDHTHHrQAY/Edt0N3HTH5UHLWeDQgyaxckZ54IkWpIf4DmylWWnQchY59GAePnPoST98hBtwF1664/Kg5Sxx6MEJvOfQkwexX6lF7JDeg93OipKg5Sxz+I3+NvpIZPS61VnWtZ5HH7wTK4XZh+ccFoF+3AScFftIvUan4Li4N0by8mX0Jax56MPvWuxWyiLZQXoWVZ4xsSs4EolEIn/gC5RYTlUZVGizAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,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>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAAuUlEQVR4XmNgGAWDFiwH4kdA/A+IA9DkvgJxJ7LAESBmA+L/QLwFWQIqVgLjuKNJVCDxnYH4JxBzwARgDE0GiGJFmAQQNDFAbMUAZ4G4GokPMuQHEHshiYEBHxD/BWIbJDF7BohN/EhiYCAHlZBAErsIFcMKlgBxOpQtzABRiFMxCAgBsQUQMzNAFB5FlYYAUKDXIvEFGCCKkYMVDkASa6BsWyD+xoAahCgAFKU5QNzKAFGIHosjAwAA6QMkIufaJwcAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAbCAYAAABFuB6DAAAAt0lEQVR4XmNgGAUDAjiAOBGIRaF8GSAuAmJTuAooWAal/wPxWyAWh/LXQcXAgA2IraBskOBVmAQQ5ELFwACkEAT0oIJaMAkgmAIVg6kBA5Dul8gCQHCGAclEGFgDxKvQxP4woCnUhQqAaGQAEutCFoA52hBJLAiI7wAxJ5IYwxsGiMKjQPwKiH8C8RxkBTAAUgTCjECsCcRCqNIIAFK0Hl0QHYCiD6SwEF0CHZRAcSUQK6PJDU8AAN3PJEne9OP6AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAaCAYAAADIUm6MAAACXklEQVR4Xu2WzauNURTGl5CvusTF9FLXx4gBmaAwMiFKBtdHSSJRJCn+BaUrKQMxogglA2VgIBG3lImvDEgikYkkX89j7XPO7jlr3/uecz9Mzq+ezruftd599t7vftd+zTp0+K+8hTapWZGz0GQ1qzABWgdt1EDGJGiNmolp0Gk1W4B9X4bGaSCCSQegH9BD6Dv0B9qQJ2Uwdl1NcNg8NlyOW8V+zpsnTk3tM6m9sp7RoMc8tk188gF6pGYbzIK+qakshp5AczLvqpVn/BS6pWaCT4qPeiRYZb5liiy15v302OKBLzD3d2kALIPuqTkMpkAfrXlsg8LBRQPnNoh8cgnarGbiiPkgav1G2lrPbkD/kJqDwRt+q2nlCZFn0HI1E7znM/TOvAA8h96nX+ouNL+WnMH7+tVU+Eh0FSLxjxTe+8v8pcqZDr3M2i+gY+m69J7k8P9uqxmxFjphfsOe1KZWQyuSf66e3YADZGy8BgTm1J7KpzxQgPkDapZ4bfEBss+8I76gESxfeWVSWG1upGsuBvsaCubcUTNiiXlyVL8vmpe70qrymF+oZgYr0f50vcOqD5yleUheWbnDr9C1rN2bXRNWlT7xavATIu+XC8N2V+ZFMGe7mhFMjAY+29y/ktonoZ+N8D92m5+4EXwR835npPaWzItgziI1FVaG+9A8DZh/eL0x74h7md810ZbhZKJVnAvNFK9H2sp66Kia7dBt/g5wEiU4sYNqtgn3tk521Nhp8VZrlVPQFzVHmwvWfBC1CqsXt8qY8wDaq2ZFbqoxlkxMageewh06jBR/ASIViVVUnc58AAAAAElFTkSuQmCC>