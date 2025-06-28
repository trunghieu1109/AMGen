async def forward_69(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and identify chemical species and relationships
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and identify all chemical species (A, B, C, D, E, F, G, H) and their relationships from the query, "
        "including stoichiometric ratios, reaction sequences, and properties such as acidity, hazard level, and solvent use."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting chemical species and relationships, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 0: Summarize key reaction pathways and transformations
    cot_instruction_0_2 = (
        "Sub-task 2: Summarize the key reaction pathways and transformations based on the extracted species and relationships: "
        "A + 8B → C (bright red), C + 2D → E (hazardous), C + H2O → A + F (strong acid) + G (weak acid), D + B → H (solvent)."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, summarizing reaction pathways, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Analyze chemical nature and possible identities of species
    cot_instruction_1_3 = (
        "Sub-task 3: Analyze the chemical nature and possible identities of species A, B, C, D, E, F, G, and H based on the given reaction stoichiometry, "
        "product properties (color, hazard, acidity), and typical chemical behavior to hypothesize their molecular structures or classes."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, analyzing chemical nature and identities, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 1: Determine structural features of product E
    cot_instruction_1_4 = (
        "Sub-task 4: Determine the structural features of product E based on the reaction of C with 2 equivalents of D, "
        "considering the hazardous nature of E and the known transformations, to infer possible molecular geometry and symmetry elements."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, determining structural features of E, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Classify E into molecular symmetry groups
    debate_instruction_2_5 = (
        "Sub-task 5: Using the inferred structure and symmetry elements of E, classify E into one of the molecular symmetry groups (C2, C2v, D4h, D∞h) "
        "by applying group theory principles and symmetry criteria."
    )
    debate_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_2_5 = [[] for _ in range(N_max_5)]
    all_answer_2_5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_2_5):
            if r == 0:
                thinking_2_5, answer_2_5 = await agent([taskInfo, thinking_1_4, answer_1_4], debate_instruction_2_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_1_4, answer_1_4] + all_thinking_2_5[r-1] + all_answer_2_5[r-1]
                thinking_2_5, answer_2_5 = await agent(input_infos_5, debate_instruction_2_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying molecular symmetry of E, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
            all_thinking_2_5[r].append(thinking_2_5)
            all_answer_2_5[r].append(answer_2_5)

    final_decision_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await final_decision_agent_2_5([taskInfo] + all_thinking_2_5[-1] + all_answer_2_5[-1], "Sub-task 5: Make final decision on the molecular symmetry group of E.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding molecular symmetry group of E, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Stage 2: Validate assigned molecular symmetry group of E
    cot_reflect_instruction_2_6 = (
        "Sub-task 6: Validate the assigned molecular symmetry group of E by cross-checking with known chemical and physical properties "
        "(color, hazard, reaction behavior) and consistency with the reaction scheme."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_2_6 = [taskInfo, thinking_2_5, answer_2_5]

    thinking_2_6, answer_2_6 = await cot_agent_2_6(cot_inputs_2_6, cot_reflect_instruction_2_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_6.id}, validating molecular symmetry group, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")

    for i in range(N_max_6):
        feedback, correct = await critic_agent_2_6([taskInfo, thinking_2_6, answer_2_6],
                                                 "Please review the validation of the molecular symmetry group and provide its limitations.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_6.extend([thinking_2_6, answer_2_6, feedback])
        thinking_2_6, answer_2_6 = await cot_agent_2_6(cot_inputs_2_6, cot_reflect_instruction_2_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_6.id}, refining validation, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")

    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_6, answer_2_6, sub_tasks, agents)
    return final_answer
