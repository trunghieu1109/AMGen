async def forward_109(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze observational constraints and star data

    # Sub-task 1: Analyze observational constraints of ESPRESSO and HIRES
    cot_instruction_1 = (
        "Sub-task 1: Analyze the observational constraints of the ESPRESSO and HIRES spectrographs, "
        "specifically the apparent V magnitude limits (ESPRESSO: brighter than 17 mag, HIRES: brighter than 16 mag), "
        "disregarding pointing and altitude limits as per the problem statement."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing observational constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and classify star data
    cot_instruction_2 = (
        "Sub-task 2: Extract and classify the given star data, including coordinates (RA, DEC), apparent or absolute V magnitudes, distances, "
        "and color excess E(B-V) where provided, to prepare for magnitude calculations and observability evaluation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting and classifying star data, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Convert all star magnitudes to apparent V magnitudes with extinction corrections
    cot_sc_instruction_3 = (
        "Sub-task 3: Convert all star magnitudes to apparent V magnitudes, using the distance modulus formula for stars with absolute magnitudes and distances, "
        "and apply extinction corrections where E(B-V) is given (using A_V = 3.1 * E(B-V)) to get the final apparent V magnitude for each star."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, converting magnitudes to apparent V magnitudes with extinction, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most consistent answer (majority vote)
    answer3_final = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate observability and select stars

    # Sub-task 4: Evaluate each star's apparent V magnitude against ESPRESSO and HIRES limits
    cot_instruction_4 = (
        "Sub-task 4: Evaluate each star's apparent V magnitude against the ESPRESSO (brighter than 17 mag) and HIRES (brighter than 16 mag) detection limits "
        "to determine which stars are observable by each spectrograph."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3_final, answer3_final], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating observability of stars, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Identify stars observable by both ESPRESSO and HIRES
    cot_instruction_5 = (
        "Sub-task 5: Identify the stars that can be detected by both ESPRESSO and HIRES spectrographs by intersecting the sets of stars observable by each instrument."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, identifying stars observable by both instruments, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare identified stars with provided answer choices to select correct pair(s)
    debate_instruction_6 = (
        "Sub-task 6: Compare the identified stars observable by both instruments with the provided answer choices (Star1 and Star4, Star4 and Star5, Star2 and Star3, Star3 and Star5) "
        "to select the correct pair(s)."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating correct star pairs, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the correct star pairs observable by both spectrographs.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on star pairs, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
