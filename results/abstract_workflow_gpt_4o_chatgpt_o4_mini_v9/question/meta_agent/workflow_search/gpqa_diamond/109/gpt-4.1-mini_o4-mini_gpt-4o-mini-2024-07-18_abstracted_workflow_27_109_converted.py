async def forward_109(self, taskInfo):
    from collections import Counter

    print("Task Requirement: ", taskInfo)

    sub_tasks = []
    agents = []

    # Stage 0: Convert all star magnitudes to apparent V magnitudes
    cot_instruction_0 = (
        "Sub-task 1: Convert all star magnitudes to apparent V magnitudes using given absolute magnitudes, distances, "
        "and extinction values where applicable. Calculate distance modulus and apply extinction correction using E(B-V) and coefficient 3.1 for stars with color excess. "
        "This enables direct comparison with detection limits of ESPRESSO and HIRES spectrographs."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, converting star magnitudes, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Analyze apparent V magnitudes for ESPRESSO and HIRES detection criteria
    cot_sc_instruction_1_espresso = (
        "Sub-task 2: Analyze each star's apparent V magnitude (computed or given) to determine if it meets the detection criteria for the ESPRESSO spectrograph (apparent V magnitude < 17 mag). "
        "Consider all stars independently and provide detection status."
    )
    cot_sc_instruction_1_hires = (
        "Sub-task 3: Analyze each star's apparent V magnitude (computed or given) to determine if it meets the detection criteria for the HIRES spectrograph (apparent V magnitude < 16 mag). "
        "Consider all stars independently and provide detection status."
    )

    N = self.max_sc
    cot_agents_espresso = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_hires = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_espresso = []
    thinkingmapping_espresso = {}
    answermapping_espresso = {}

    possible_answers_hires = []
    thinkingmapping_hires = {}
    answermapping_hires = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents_espresso[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1_espresso, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_espresso[i].id}, analyzing ESPRESSO detection, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_espresso.append(answer2.content)
        thinkingmapping_espresso[answer2.content] = thinking2
        answermapping_espresso[answer2.content] = answer2

        thinking3, answer3 = await cot_agents_hires[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1_hires, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_hires[i].id}, analyzing HIRES detection, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_hires.append(answer3.content)
        thinkingmapping_hires[answer3.content] = thinking3
        answermapping_hires[answer3.content] = answer3

    # Aggregate most consistent answers by majority vote for ESPRESSO
    counter_espresso = Counter(possible_answers_espresso)
    final_answer_espresso = counter_espresso.most_common(1)[0][0]
    thinking_espresso = thinkingmapping_espresso[final_answer_espresso]
    answer_espresso = answermapping_espresso[final_answer_espresso]

    # Aggregate most consistent answers by majority vote for HIRES
    counter_hires = Counter(possible_answers_hires)
    final_answer_hires = counter_hires.most_common(1)[0][0]
    thinking_hires = thinkingmapping_hires[final_answer_hires]
    answer_hires = answermapping_hires[final_answer_hires]

    sub_tasks.append(f"Sub-task 2 output (ESPRESSO detection): thinking - {thinking_espresso.content}; answer - {answer_espresso.content}")
    print("Step 2 (ESPRESSO): ", sub_tasks[-1])
    sub_tasks.append(f"Sub-task 3 output (HIRES detection): thinking - {thinking_hires.content}; answer - {answer_hires.content}")
    print("Step 3 (HIRES): ", sub_tasks[-1])

    # Stage 2: Identify stars detectable by both ESPRESSO and HIRES, then compare with answer choices
    cot_reflect_instruction_4 = (
        "Sub-task 4: Identify stars that can be detected by both ESPRESSO and HIRES spectrographs by selecting stars that satisfy both detection criteria from subtasks 2 and 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking_espresso, answer_espresso, thinking_hires, answer_hires], cot_reflect_instruction_4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, identifying stars detectable by both instruments, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Compare the identified stars detectable by both ESPRESSO and HIRES with the provided answer choices to determine which choice correctly lists the stars observable by both instruments. "
        "Use debate agents to consider each choice and reach consensus on the correct answer."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing identified stars with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which choice correctly lists the stars observable by both ESPRESSO and HIRES.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
