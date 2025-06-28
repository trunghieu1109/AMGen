async def forward_3(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Extract and Define Baseline and Concepts

    # Sub-task 1: Extract and clearly state the standard Maxwell's equations involving magnetic field divergence, curl, and electric field circulation
    cot_instruction_1 = (
        "Sub-task 1: Extract and clearly state the standard Maxwell's equations in classical electromagnetism, "
        "focusing on the equations involving the magnetic field (both divergence and curl forms) and the electric field circulation, "
        "to establish the baseline for comparison."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracted Maxwell's baseline equations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and define magnetic monopoles and how their hypothetical existence modifies Maxwell's equations, especially magnetic field terms
    cot_instruction_2 = (
        "Sub-task 2: Identify and define the concept of magnetic monopoles (isolated North or South poles) and explain how their hypothetical existence modifies the physical assumptions underlying Maxwell's equations, "
        "particularly the magnetic field terms."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, defined magnetic monopoles and their impact, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Apply Transformations and Assess Impact

    # Sub-task 3: Apply the hypothetical transformation of introducing magnetic monopoles to Maxwell's equations, rewriting equations to include magnetic charge and current terms
    cot_instruction_3 = (
        "Sub-task 3: Apply the hypothetical transformation of introducing magnetic monopoles to the standard Maxwell's equations, "
        "explicitly rewriting the equations to include magnetic charge and current terms, and note which equations are altered."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, applied magnetic monopole transformation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Assess impact of magnetic monopoles on divergence and curl of magnetic field equations
    cot_instruction_4 = (
        "Sub-task 4: Assess the impact of the introduced magnetic monopoles on the divergence of the magnetic field equation and the curl of the magnetic field equation, "
        "determining which of these equations change and how."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, assessed impact on magnetic field equations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Analyze and Classify Differences and Compare to Choices

    # Sub-task 5: Analyze modified Maxwell's equations to classify which specific equations differ due to magnetic monopoles
    cot_instruction_5 = (
        "Sub-task 5: Analyze the modified Maxwell's equations to classify which specific equations differ from the classical form due to the presence of magnetic monopoles, "
        "focusing on divergence and curl of magnetic field and circulation of electric field."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, analyzed and classified equation differences, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare classification results with multiple-choice options to identify correct choice
    debate_instruction_6 = (
        "Sub-task 6: Compare the classification results from Sub-task 5 with the provided multiple-choice options to identify which choice correctly describes the equations that differ in the parallel universe scenario with magnetic monopoles."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6 += all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing classification with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which choice correctly describes the equations that differ.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing correct choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
