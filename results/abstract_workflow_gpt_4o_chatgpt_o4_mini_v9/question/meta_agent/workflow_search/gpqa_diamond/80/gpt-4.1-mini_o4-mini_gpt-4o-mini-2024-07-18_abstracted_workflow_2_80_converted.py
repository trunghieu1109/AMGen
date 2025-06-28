async def forward_80(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the target molecule [1,1'-bi(cyclopentylidene)]-2-one
    cot_instruction_1 = (
        "Sub-task 1: Analyze the target molecule [1,1'-bi(cyclopentylidene)]-2-one to understand its structural features, "
        "functional groups, and key transformations required for its synthesis starting from 1,5-dichloropentane."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing target molecule, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze the starting material 1,5-dichloropentane
    cot_instruction_2 = (
        "Sub-task 2: Analyze the starting material 1,5-dichloropentane to identify its functional groups, reactive sites, "
        "and possible transformations relevant to the target molecule synthesis."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing starting material, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze each reagent and condition in the provided choices
    cot_instruction_3 = (
        "Sub-task 3: Analyze each reagent and condition in the provided choices (choice1 to choice4) to classify their chemical roles "
        "(e.g., reduction, halogenation, elimination, oxidation, etc.) and expected transformations on 1,5-dichloropentane or intermediates."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing reagents and conditions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Map, Evaluate, and Select Correct Sequence

    # Sub-task 4: Map the sequence of reagents in each choice to plausible stepwise chemical transformations
    cot_instruction_4 = (
        "Sub-task 4: Map the sequence of reagents in each choice to plausible stepwise chemical transformations starting from 1,5-dichloropentane, "
        "considering the structural requirements of the target molecule identified in subtask_1."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, mapping reagent sequences to transformations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate the chemical feasibility and correctness of each mapped sequence
    debate_instruction_5 = (
        "Sub-task 5: Evaluate the chemical feasibility and correctness of each mapped sequence from Sub-task 4 by checking if the intermediate and final products match the target molecule's structure and functional groups."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating mapped sequences, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct reagent sequence leading to the synthesis of [1,1'-bi(cyclopentylidene)]-2-one.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct sequence, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Select the correct sequence of reagents from the given choices
    cot_reflect_instruction_6 = (
        "Sub-task 6: Based on the evaluation in Sub-task 5, select the correct sequence of reagents from the given choices "
        "that leads to the successful synthesis of [1,1'-bi(cyclopentylidene)]-2-one starting from 1,5-dichloropentane."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, selecting correct reagent sequence, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
