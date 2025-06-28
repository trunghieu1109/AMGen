async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Calculate Energy Uncertainties
    # Sub-task 1: Analyze the given quantum states lifetimes and their physical significance
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given quantum states lifetimes (10^-9 sec and 10^-8 sec) and explain how lifetime relates to energy uncertainty and resolution of energy levels."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing quantum states lifetimes, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and explain the relationship between lifetime and energy uncertainty using energy-time uncertainty principle
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, explain the relationship between the lifetime of a quantum state and the uncertainty in its energy level using the energy-time uncertainty principle."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, explaining lifetime-energy uncertainty relation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_content = thinkingmapping_2[answer2_content].content
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_content}; answer - {answer2_content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate minimum energy uncertainty (ΔE) for each quantum state
    cot_instruction_3 = (
        "Sub-task 3: Calculate the minimum energy uncertainty (ΔE) for each quantum state using the energy-time uncertainty relation ΔE * Δt >= hbar/2, with given lifetimes 10^-9 sec and 10^-8 sec."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answermapping_2[answer2_content]], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating energy uncertainties, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Determine minimum energy difference and evaluate options
    # Sub-task 4: Determine minimum energy difference required to distinguish the two states
    cot_instruction_4 = (
        "Sub-task 4: Determine the minimum energy difference required between the two states so that their energy levels can be clearly distinguished, based on the calculated energy uncertainties from Sub-task 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining minimum energy difference, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate each provided energy difference option against minimum energy difference
    debate_instruction_5 = (
        "Sub-task 5: Evaluate each energy difference option (10^-9 eV, 10^-11 eV, 10^-8 eV, 10^-4 eV) against the minimum energy difference required to identify which options allow clear resolution of the two energy levels."
    )
    debate_roles = ["Pro", "Con"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating energy difference options, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    # Sub-task 6: Select the correct energy difference option based on debate
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 6: Select the correct energy difference option that satisfies the condition for clear resolution of the two quantum states.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct energy difference option, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
