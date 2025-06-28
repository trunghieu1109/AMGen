async def forward_179(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and identify all defining features and parameters
    cot_instruction_0 = (
        "Sub-task 1: Extract and identify all defining features and parameters from the problem statement: "
        "number of particles (13), charge of each particle (2e), mass negligible, spatial constraints (12 charges at 2 m from point P, "
        "13th charge fixed at P), and the quantity to find (minimum energy of the system in Joules, correct to three decimals)."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting defining features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Analyze spatial configuration and calculate energy contributions
    # Use Reflexion for detailed analysis and refinement
    cot_reflect_instruction_1 = (
        "Sub-task 2: Analyze the spatial configuration of the charges: understand the geometric arrangement of 12 identical charges constrained on a sphere of radius 2 m around point P, "
        "with the 13th charge fixed at P. Recognize symmetry and minimal energy configuration. Then calculate the electrostatic potential energy contributions between the 13th charge at P and each of the 12 charges, "
        "and among the 12 charges themselves."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking1, answer1]

    thinking2, answer2 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, analyzing spatial configuration and calculating energies, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max_1):
        feedback, correct = await critic_agent_1([taskInfo, thinking2, answer2],
                                               "Critically evaluate the spatial analysis and energy calculations and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_1(cot_inputs_1, cot_reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining spatial analysis and energy calculations, thinking: {thinking2.content}; answer: {answer2.content}")

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Convert total energy to Joules and compare with choices
    debate_instruction_2 = (
        "Sub-task 3: Convert the total electrostatic potential energy into Joules, ensuring correct unit usage and precision to three decimals, "
        "then compare the calculated minimum energy value with the provided multiple-choice options to identify the correct answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]

    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_2.extend(all_thinking2[r-1])
                input_infos_2.extend(all_answer2[r-1])
            thinking3, answer3 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, converting energy and comparing choices, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking2[r].append(thinking3)
            all_answer2[r].append(answer3)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1],
                                                     "Sub-task 3: Make final decision on the minimum energy value in Joules and select the correct choice.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent on final energy answer, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer
