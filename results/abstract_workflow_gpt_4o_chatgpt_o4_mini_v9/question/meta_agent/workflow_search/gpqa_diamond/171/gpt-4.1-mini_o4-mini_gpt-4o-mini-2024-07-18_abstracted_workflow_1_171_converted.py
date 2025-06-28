async def forward_171(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze problem statement and classify knowns and unknowns
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the problem statement to identify key physical concepts such as excitation of iron atoms, "
        "energy difference, LTE assumption, and classify known quantities (excitation ratio=2:1, energy difference=1.38e-23 J) "
        "and unknowns (effective temperatures T1 and T2)."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing problem statement, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Identify and classify the candidate equations given as choices, noting their mathematical forms, "
        "variables involved (T1, T2, ln(2)), to prepare for comparison with theoretical relations derived from LTE and excitation ratios."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_2, answer0_2 = await cot_agent_0_2([taskInfo, thinking0_1, answer0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, classifying candidate equations, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Derive theoretical relation using Boltzmann distribution
    cot_instruction_1_3 = (
        "Sub-task 3: Derive the theoretical relationship between the excitation ratio of iron atoms in two stars "
        "and their effective temperatures under LTE, using the Boltzmann distribution and given energy difference. "
        "Express the ratio of excited state populations in terms of T1, T2, and the energy difference."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_3, answer1_3 = await cot_agent_1_3([taskInfo, thinking0_1, answer0_1], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, deriving Boltzmann relation, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Manipulate the derived Boltzmann relation to isolate and express ln(2) (natural logarithm of excitation ratio) "
        "in terms of T1 and T2, matching the form of the candidate equations."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1_4, answer1_4 = await cot_agent_1_4([taskInfo, thinking1_3, answer1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, manipulating Boltzmann relation, thinking: {thinking1_4.content}; answer: {answer1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking1_4.content}; answer - {answer1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Compare derived expression with candidate equations
    debate_instruction_2_5 = (
        "Sub-task 5: Compare the derived expression for ln(2) with each of the provided candidate equations to identify "
        "which equation correctly represents the relationship between T1 and T2 given the excitation ratio and energy difference."
    )
    debate_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_5 = self.max_round
    all_thinking2_5 = [[] for _ in range(N_max_2_5)]
    all_answer2_5 = [[] for _ in range(N_max_2_5)]

    for r in range(N_max_2_5):
        for i, agent in enumerate(debate_agents_2_5):
            if r == 0:
                thinking2_5, answer2_5 = await agent([taskInfo, thinking1_4, answer1_4, thinking0_2, answer0_2], debate_instruction_2_5, r, is_sub_task=True)
            else:
                input_infos_2_5 = [taskInfo, thinking1_4, answer1_4, thinking0_2, answer0_2] + all_thinking2_5[r-1] + all_answer2_5[r-1]
                thinking2_5, answer2_5 = await agent(input_infos_2_5, debate_instruction_2_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing derived expression with candidates, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
            all_thinking2_5[r].append(thinking2_5)
            all_answer2_5[r].append(answer2_5)

    final_decision_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_5, answer2_5 = await final_decision_agent_2_5([taskInfo] + all_thinking2_5[-1] + all_answer2_5[-1], "Sub-task 5: Make final decision on which candidate equation matches the derived theoretical expression.", is_sub_task=True)
    agents.append(f"Final Decision agent on candidate equation selection, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2_5.content}; answer - {answer2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Select the correct equation
    cot_instruction_3_6 = (
        "Sub-task 6: Select the correct equation from the given choices that matches the derived theoretical expression "
        "for the effective temperatures T1 and T2, confirming the correct formula for the problem scenario."
    )
    cot_agent_3_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3_6, answer3_6 = await cot_agent_3_6([taskInfo, thinking2_5, answer2_5], cot_instruction_3_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_6.id}, selecting correct equation, thinking: {thinking3_6.content}; answer: {answer3_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking3_6.content}; answer - {answer3_6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_6, answer3_6, sub_tasks, agents)
    return final_answer
