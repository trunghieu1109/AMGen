async def forward_52(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Analyze spectral data and correlate with structure

    # Sub-task 1: Analyze FTIR and 1H NMR spectral data to identify functional groups and structural features
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given spectral data (FTIR and 1H NMR) to identify key functional groups and structural features of the compound, "
        "including confirmation of the ester group from FTIR and detailed interpretation of the 1H NMR signals (aromatic-H, vinyl-H, and methyl groups) with their splitting patterns and absence of methylene (-CH2) signals."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing spectral data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Correlate spectral features with di-substituted 6-membered aromatic ring structure
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, correlate the spectral features with the structural characteristics of a di-substituted 6-membered aromatic ring compound, "
        "focusing on how the number and type of hydrogens (aromatic, vinyl, methyl) and absence of methylene groups constrain the possible molecular formula and substitution pattern."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, correlating spectral features with structure, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most frequent answer from SC-CoT
    from collections import Counter
    answer2_counter = Counter(possible_answers_2)
    most_common_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer2]
    answer2 = answermapping_2[most_common_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate candidate molecular formulas and select the best fit

    # Sub-task 3: Evaluate each candidate molecular formula against constraints
    cot_reflect_instruction_3 = (
        "Sub-task 3: Evaluate each candidate molecular formula (C11H12O2, C11H14O2, C12H12O2, C12H14O2) against the constraints derived from the spectral data and structural analysis, "
        "including hydrogen count consistency with the NMR signals, presence of ester group, and degree of unsaturation implied by the aromatic ring and vinyl groups."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating candidate formulas, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                                "Critically evaluate the evaluation of candidate molecular formulas for correctness and completeness, and provide limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining evaluation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Prioritize and select the most plausible molecular formula
    debate_instruction_4 = (
        "Sub-task 4: Based on the evaluation of candidate molecular formulas, debate and prioritize the most plausible molecular formula from the given choices "
        "based on best fit to all spectral and structural evidence, ensuring consistency with the number and type of hydrogens, presence of ester, and substitution pattern on the aromatic ring."
    )
    debate_roles = ["Proponent", "Skeptic"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating molecular formula selection, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                    "Sub-task 4: Make final decision on the most plausible molecular formula based on debate outputs.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, making final molecular formula decision, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
