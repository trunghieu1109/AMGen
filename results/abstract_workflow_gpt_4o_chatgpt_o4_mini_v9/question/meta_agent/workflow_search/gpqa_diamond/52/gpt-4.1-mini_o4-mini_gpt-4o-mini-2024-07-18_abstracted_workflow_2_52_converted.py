async def forward_52(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements
    # Sub-task 1: Analyze spectral data (FTIR and 1H NMR) to identify functional groups and proton environments
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given spectral data (FTIR and 1H NMR) to identify and classify the functional groups and proton environments present in the compound, "
        "including confirming the presence of an ester group and characterizing the aromatic, vinyl, and methyl proton signals."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing spectral data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Interpret 1H NMR splitting patterns and chemical shifts to deduce substitution pattern and vinyl proton nature
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, interpret the 1H NMR splitting patterns and chemical shifts of the six signals (two aromatic-H, two vinyl-H, two methyl groups) "
        "to deduce the substitution pattern on the aromatic ring and the nature of the vinyl protons, considering the absence of –CH2 signals."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, interpreting NMR splitting and shifts, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer by frequency
    answer2_final = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[answer2_final]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Integrate and Compare
    # Sub-task 3: Integrate spectral analysis and proton environment interpretation to infer possible molecular structures
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on the outputs from Sub-tasks 1 and 2, integrate the spectral analysis and proton environment interpretation to infer possible molecular structure(s) "
        "consistent with a di-substituted 6-membered aromatic ring containing an ester group and the described proton signals."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2_final, answermapping_2[answer2_final]]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, inferring molecular structures, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3],
                                                  "Please review the inferred molecular structures for consistency with spectral data and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining molecular structure inference, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Compare inferred molecular structure(s) with given molecular formula options to identify best fit
    debate_instruction_4 = (
        "Sub-task 4: Based on the inferred molecular structures, debate and compare the given molecular formula options (C11H12O2, C11H14O2, C12H12O2, C12H14O2) "
        "to identify which formula best fits the spectral data and structural deductions."
    )
    debate_roles = ["Proponent", "Opponent"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, debating molecular formula fit, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                    "Sub-task 4: Make final decision on the molecular formula that best fits the spectral data and structural deductions.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, deciding molecular formula, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
