async def forward_3(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Review Maxwell's equations and understand magnetic monopoles

    # Sub-task 1: Review and clearly state the standard Maxwell's equations involving magnetic and electric fields
    cot_instruction_1 = (
        "Sub-task 1: Review and clearly state the standard Maxwell's equations in classical electromagnetism, "
        "focusing on the equations involving the magnetic field (divergence and curl) and the electric field (circulation). "
        "This provides the baseline for comparison with the parallel universe scenario."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, reviewing Maxwell's equations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Understand physical implication of isolated magnetic monopoles and how it modifies Maxwell's equations
    cot_instruction_2 = (
        "Sub-task 2: Understand the physical implication of the existence of isolated magnetic monopoles (isolated North or South poles) in a parallel universe, "
        "and how this concept modifies the assumptions behind Maxwell's equations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, understanding magnetic monopoles, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Analyze modifications to Maxwell's equations due to magnetic monopoles

    # Sub-task 3: Analyze how magnetic monopoles change the divergence equation of the magnetic field
    cot_instruction_3 = (
        "Sub-task 3: Analyze how the presence of magnetic monopoles changes the divergence equation of the magnetic field, "
        "based on the understanding from Sub-task 2 and the baseline equations from Sub-task 1."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing divergence modification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze how magnetic monopoles affect the curl equations of magnetic and electric fields
    cot_instruction_4 = (
        "Sub-task 4: Analyze how the presence of magnetic monopoles affects the curl equations of the magnetic and electric fields, "
        "particularly focusing on the circulation of the electric field and the curl of the magnetic field, using insights from Sub-task 2 and Sub-task 1."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing curl modification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare modified Maxwell's equations with original ones to identify which differ
    debate_instruction_5 = (
        "Sub-task 5: Compare the modified Maxwell's equations (from Sub-task 3 and Sub-task 4) with the original ones to identify which specific equations differ in the parallel universe scenario. "
        "Debate the differences and reach consensus."
    )
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking3, answer3, thinking4, answer4], debate_instruction_5, r, is_sub_task=True
                )
            else:
                input_infos_5 = [taskInfo, thinking3, answer3, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing equations, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on which Maxwell's equations differ in the parallel universe scenario.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding differing equations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Map identified differences to multiple-choice options and evaluate correct choice
    cot_instruction_6 = (
        "Sub-task 6: Map the identified differences in Maxwell's equations to the provided multiple-choice options, "
        "evaluating which choice correctly describes the changes due to magnetic monopoles."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, mapping differences to choices, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
