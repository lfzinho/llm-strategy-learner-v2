# Literature review

Related work for the research question *"How do LLM agents behave in repeated social dilemmas with communication and reputation?"*

**How to read this file.** Entries marked ✅ were checked against an arXiv or publisher page during the writing of this review, and the summaries follow the abstracts. Entries marked ⚠️ are well-known works cited from memory: the title and authors should be correct, but **confirm the exact reference, year and claims before citing them in the thesis**. I only read abstracts and search summaries, not full texts, so read each paper before relying on details.

The papers are grouped by the part of the project they inform.

---

## A. Foundations: the repeated Prisoner's Dilemma

### A1. Axelrod and Hamilton (1981), "The Evolution of Cooperation", *Science*; Axelrod (1984), *The Evolution of Cooperation* ⚠️
Computer tournaments in which strategies submitted by researchers played the iterated PD. Tit For Tat, a strategy that starts nice and copies the opponent, won. The author draws four properties of successful strategies: nice, retaliatory, forgiving, clear.
- **Use:** the theoretical frame of Step 1. Our opponent pool is built from these strategy families, and "nice / retaliating / forgiving / clear" is a ready vocabulary for describing what an LLM does.

### A2. Knight et al. (2016), "An Open Framework for the Reproducible Study of the Iterated Prisoner's Dilemma" (Axelrod library) ⚠️
The software library behind our opponents, with a few hundred strategies, tournaments, noise and probabilistic endings.
- **Use:** cite as the source of opponents and the match engine. Gives a route to a much larger opponent pool and to held-out evaluation sets.

### A3. Nowak and Sigmund (1998), "Evolution of indirect reciprocity by image scoring", *Nature* ⚠️
Shows that cooperation can be sustained without direct repetition when agents keep a reputation score and help those with good reputations.
- **Use:** the theoretical basis for Step 3 (reputation). Defines the "image score" that our public ledger could implement.

### A4. Fehr and Gächter (2000), "Cooperation and Punishment in Public Goods Experiments", *American Economic Review* ⚠️
Human experiments showing that people pay a cost to punish free-riders, and that this sustains cooperation.
- **Use:** a human reference for third-party and costly punishment, which Step 3 can test in LLM agents. Gives a human baseline to compare with qualitatively.

---

## B. LLMs in the repeated PD and related games

### B1. Akata et al. (2023/2025), "Playing repeated games with Large Language Models", arXiv:2305.16867; *Nature Human Behaviour* ✅
Behavioural game theory on GPT-3, GPT-3.5 and GPT-4 in finitely repeated 2×2 games, against each other, against human-like strategies and against humans. The models do best in self-interested games such as the iterated PD, but poorly in coordination games such as Battle of the Sexes. GPT-4 is notably unforgiving, defecting after a single defection. Giving the model more information about the opponent and using a "social chain-of-thought" prompt changes its behaviour and improves coordination with humans.
- **Relation:** the closest prior work to Step 1. It finds the same sensitivity to information that our tier ladder measures, but without a graded ladder or a best-achievable score.
- **Use:** replicate its "unforgiving GPT-4" observation against Zen Alpha Mimic; borrow the social chain-of-thought idea as an extra block; cite as the main precedent.

### B2. Fontana, Pierri and Aiello (2024), "Nicer Than Humans: How do Large Language Models Behave in the Prisoner's Dilemma?", arXiv:2406.13605 (ICWSM) ✅
100-round iterated PD between LLMs and random opponents with varying hostility. The models rarely defect first and lean to cooperation when the opponent's defection rate is low; Llama2 and GPT-3.5 are more cooperative and forgiving than typical humans, while Llama3 behaves more like humans and is exploitative unless the opponent always cooperates. The authors also propose a meta-prompting technique to check whether a model understands the rules.
- **Relation:** consistent with our small-model result (over-cooperation) and shows that behaviour differs by model family.
- **Use:** justification for comparing several models; their behavioural-economics measures (first-move cooperation, forgiveness, retaliation) can complement our efficiency measure; their rule-comprehension check could be added as a harness sanity test.

### B3. Tewolde, Zhang et al. (2026), "CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas", arXiv:2604.15267 ✅ (publication venue not confirmed)
Reports that recent models consistently defect in one-shot social dilemmas. It tests repetition, reputation systems, third-party mediators and contracts, and finds that contracts and mediation are most effective, while repetition-induced cooperation deteriorates sharply when co-players vary.
- **Relation:** directly overlaps with Steps 2 and 3, since it studies reputation as a mechanism for LLMs. Read it first to define what you add.
- **Use:** a benchmark design to learn from; the finding that cooperation breaks down when co-players vary motivates testing on held-out opponents. Possible competitor or collaborator for novelty, so check what it does *not* cover (for example free-form communication, a graded information ladder).

### B4. Niszczota, Grzegorczyk and Pastukhov (2025), arXiv:2507.18639 ✅ (title not captured; check)
A thirty-round repeated PD experiment with human participants playing a human, a classic bot, or a real-time GPT model, with and without communication. The summary reports high cooperation with LLMs, especially when communication is possible.
- **Relation:** human data on cooperation with an LLM partner, with communication.
- **Use:** possible source of human baseline numbers for Steps 2 and 3, and evidence that communication matters.

### B5. Anwar and Georgalos (2026), arXiv:2603.15852 ✅ (title not captured; check)
Humans in an indefinitely repeated PD chat with an AI chatbot (GPT-5.2) either before every round or only before the first. Cooperation against the AI is initially comparable to human-human levels, but plateaus and never reaches full cooperation.
- **Relation:** an experiment on the same variable Step 2 manipulates (frequency of communication).
- **Use:** motivates a "chat before every round" versus "chat once" condition.

### B6. Other work to look up ⚠️
- Phelps and Russell, "Investigating emergent goal-like behaviour in large language models using experimental economics" (2023): LLMs in iterated PD under different prompted goals.
- Guo, "GPT in game theory experiments" (2023) and Lorè and Heydari, "Strategic behavior of large language models: game structure vs. contextual framing" (2023): the effect of framing, relevant to our decision to use neutral Alpha/Beta labels.
- Mei et al. (2024), "A Turing test of whether AI chatbots are behaviorally similar to humans", *PNAS*: compares chatbots with humans across games including the PD.
- Horton (2023), "Large Language Models as Simulated Economic Agents": the "homo silicus" argument for LLMs as stand-ins for human subjects.

---

## C. Societies of LLM agents

### C1. Park et al. (2023), "Generative Agents: Interactive Simulacra of Human Behavior", arXiv:2304.03442 ✅ (the original motivation for this project)
Agents with a memory stream, reflection and planning, living together in a simulated town. Behaviour such as information spreading and relationship formation emerges from the interaction.
- **Relation:** the source of our memory design (observation, reflection, retrieval by recency, importance and relevance).
- **Use:** the template for Step 2 and 3 agents. Its evaluation of agents' memory of social facts ("who told whom what") is directly relevant to reputation.

### C2. Piatti et al. (2024), "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents" (GovSim), arXiv:2404.16698 ✅
A simulation in which LLM agents share a common resource (fishery, pasture, pollution). All but the most capable models fail to sustain the resource, with the best survival rate below 54%. Failures trace to difficulty reasoning about long-term effects, and an ablation shows that communication reduces overuse by 21%. A reproducibility study exists (arXiv:2505.09289).
- **Relation:** a multi-agent social dilemma with communication, the setting of Step 2.
- **Use:** evidence that communication helps; a model for how to run and report a multi-agent experiment; the reproducibility study is a useful guide to the reporting standard expected.

### C3. Vallinder and Hughes (2024), "Cultural Evolution of Cooperation among LLM Agents", arXiv:2412.10270 ✅
Generations of LLM agents play an iterated Donor Game in which they observe peers' recent behaviour (indirect reciprocity). Societies of Claude 3.5 Sonnet agents achieve higher scores than Gemini 1.5 Flash, which beat GPT-4o. Only Claude 3.5 Sonnet could use costly punishment to reach higher scores, and results vary by random seed.
- **Relation:** the closest prior work to Step 3 (reputation through observation).
- **Use:** the Donor Game is a simpler alternative setting; the seed sensitivity is a reminder to run many seeds; confirms that model choice changes the outcome.

### C4. Gupta et al. (2025), social learning and norm formation in LLM multi-agent systems, arXiv:2510.14401 ✅ (title and authors not confirmed; check)
Appeared in a search for cultural evolution and cooperation among LLM agents; reported as studying norm formation through social learning.
- **Use:** only if it holds after reading; possibly relevant to how a norm of honesty could emerge in Step 2.

### C5. Other work to look up ⚠️
- Shinn et al. (2023), "Reflexion: Language Agents with Verbal Reinforcement Learning": reflection as a learning mechanism, related to our reflection and Dreamer steps.
- Aher, Arriaga and Kalai (2023), "Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies": the methodological case for, and limits of, simulating human studies with LLMs.
- Work on LLM deception and strategic lying in games (for example Diplomacy-playing agents): relevant to measuring promise-breaking in Step 2.

---

## D. How the literature maps onto the project

| Project step | Main references | What they give us |
|---|---|---|
| Step 1: single agent vs classical strategies | A1, A2, B1, B2, C1 | Opponent set, theory, closest precedents, memory design |
| Step 2: LLMs with communication | B3, B4, B5, C2 | Evidence that communication matters; benchmarks and designs to adapt |
| Step 3: reputation | A3, A4, B3, C3 | Theory of indirect reciprocity; closest prior experiments |
| Methods and validity | B6, C5, C2's reproducibility study | Framing effects, simulating humans, reporting standards |

## E. Gaps this project could address

These are hypotheses to check after reading the sources, not established claims.

1. **A graded information ladder.** Prior work varies information in one or two conditions (B1). A five-tier ladder with an exact best-achievable score, where efficiency is measured against a known optimum, seems new, but verify.
2. **Measurement.** Most studies report cooperation rates. Efficiency against the optimum, identification accuracy, and call-level traces (so failures are separable from poor play) are additions.
3. **Learned playbooks and reflective memory in the PD.** Reflexion-style and Generative-Agents-style memory is rarely applied to a repeated game with an exact benchmark.
4. **Communication plus reputation together, in the same harness**, with the single-agent baseline from Step 1. CoopEval (B3) overlaps here and is the first thing to read in detail.

## F. Suggested reading order

1. Akata et al. (B1): the main precedent for Step 1.
2. CoopEval (B3): the main overlap with Steps 2 and 3, to sharpen novelty.
3. Park et al. (C1) and GovSim (C2): agent design and multi-agent methods.
4. Vallinder and Hughes (C3), Nowak and Sigmund (A3): reputation.
5. Fontana et al. (B2) and the human-AI experiments (B4, B5): comparison with humans.
6. Then the ⚠️ items, verifying each reference as you go.

## G. Sources checked

- [Akata et al., arXiv:2305.16867](https://arxiv.org/pdf/2305.16867)
- [Fontana et al., arXiv:2406.13605](https://arxiv.org/pdf/2406.13605)
- [CoopEval, arXiv:2604.15267](https://www.alphaxiv.org/abs/2604.15267)
- [Niszczota et al., arXiv:2507.18639](https://econpapers.repec.org/paper/arxpapers/2507.18639.htm)
- [Anwar and Georgalos, arXiv:2603.15852](https://arxiv.org/html/2603.15852v1)
- [Piatti et al., arXiv:2404.16698](https://arxiv.org/pdf/2404.16698)
- [Vallinder and Hughes, arXiv:2412.10270](https://arxiv.org/pdf/2412.10270)
- [Gupta et al., arXiv:2510.14401](https://arxiv.org/pdf/2510.14401v1)
- [Park et al., arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
