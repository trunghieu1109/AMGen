async def forward_47(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Stepwise chemical transformations analysis using Chain-of-Thought for each subtask

    # Sub-task 1: Identify chemical structure of cyclohexanone and key functional groups
    cot_instruction_1 = (
        "Sub-task 1: Identify the chemical structure of cyclohexanone as the starting material, "
        "including its key functional groups and hydrogen environments, to establish the baseline molecular framework before any reactions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying cyclohexanone structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine product 1 formed by bromination of cyclohexanone
    cot_instruction_2 = (
        "Sub-task 2: Based on cyclohexanone structure, determine the product formed when treated with bromine, "
        "including site of bromination and structural changes."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, determining bromination product, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze reaction of product 1 with sodium hydroxide to form product 2
    cot_instruction_3 = (
        "Sub-task 3: Analyze the reaction of product 1 with sodium hydroxide to form product 2, "
        "identifying the chemical transformation and resulting structure."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing reaction with NaOH, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine structure of product 3 after treatment with thionyl chloride and pyridine
    cot_instruction_4 = (
        "Sub-task 4: Determine the structure of product 3 formed by treating product 2 with thionyl chloride and pyridine, "
        "focusing on functional group changes and overall molecular structure."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining product 3 structure, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Identify structure of product 4 after reduction with Lithium tri-tert-butoxyaluminum hydride
    cot_instruction_5 = (
        "Sub-task 5: Identify the structure of product 4 formed by reduction of product 3 with Lithium tri-tert-butoxyaluminum hydride, "
        "including functional groups reduced and final molecular structure."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, identifying product 4 structure, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Analyze final product 4 for chemically distinct hydrogens and select correct answer

    # Sub-task 6: Analyze product 4 to identify all chemically distinct hydrogen atoms
    cot_instruction_6 = (
        "Sub-task 6: Analyze the final structure of product 4 to identify all chemically distinct hydrogen atoms, "
        "considering different chemical environments to determine the number of unique hydrogen types."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, analyzing distinct hydrogens in product 4, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare identified number of distinct hydrogens with multiple-choice options and select correct answer using Debate
    debate_instruction_7 = (
        "Sub-task 7: Based on the identified number of chemically distinct hydrogens in product 4, "
        "compare with the provided multiple-choice options (6, 7, 8, 10) and select the correct answer."
    )
    debate_agents_7 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct hydrogen count, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct number of chemically distinct hydrogens.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct hydrogen count, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
