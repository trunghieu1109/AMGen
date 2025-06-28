async def forward_107(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Compute paramagnetic coupling term and transition energy

    # Sub-task 1: Identify and write down the paramagnetic coupling term <H> for electron in B=1T along Oz with small m
    cot_instruction_1 = (
        "Sub-task 1: Identify and write down the paramagnetic coupling term (Hamiltonian expectation value <H>) "
        "for an electron in a magnetic field B parallel to Oz axis, with small orbital magnetic quantum number m and B=1 Tesla. "
        "Recall the formula for the paramagnetic term related to the orbital magnetic moment and magnetic field."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying paramagnetic coupling term, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Calculate transition energy ΔE of Hydrogen atom for wavelength λ=0.4861 μm using E=hc/λ
    cot_instruction_2 = (
        "Sub-task 2: Calculate the transition energy ΔE of the Hydrogen atom corresponding to the wavelength λ=0.4861 micrometers "
        "using the photon energy formula E = hc/λ. Use known constants for Planck's constant h and speed of light c."
    )
    N = self.max_sc if hasattr(self, 'max_sc') else 5
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating transition energy, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most frequent answer for transition energy
    from collections import Counter
    answer_counts = Counter(possible_answers_2)
    best_answer_2 = answer_counts.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Compare magnitudes and select correct choice

    # Sub-task 3: Compare order of magnitude of <H> (from subtask 1) with ΔE (from subtask 2)
    cot_instruction_3 = (
        "Sub-task 3: Compare the order of magnitude of the paramagnetic coupling term <H> from Sub-task 1 "
        "with the transition energy ΔE from Sub-task 2 to determine their relative size."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, comparing magnitudes, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Based on comparison, select correct choice among given options
    cot_instruction_4 = (
        "Sub-task 4: Based on the comparison in Sub-task 3, select the correct choice among the given options: "
        "<H> << ΔE, <H> >> ΔE, <H> = ΔE, or <H> > ΔE that best describes the relationship between the paramagnetic coupling term and the transition energy."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, selecting correct choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
