async def forward_113(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the first reaction
    cot_instruction_1 = (
        "Sub-task 1: Analyze the first reaction: butan-2-one + NaCN + A ---> 2-hydroxy-2-methylbutanenitrile. "
        "Identify the type of reaction, the role of NaCN, and the expected function of reagent A in this context."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing first reaction, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Analyze the second reaction
    cot_instruction_2 = (
        "Sub-task 2: Analyze the second reaction: 2-(4-benzylphenyl)-2-hydroxybutanenitrile + B (H2O) ---> 2-(4-benzylphenyl)-2-hydroxybutanoic acid. "
        "Identify the type of reaction, the role of water, and the expected function of reagent B in this context."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing second reaction, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Classify reagents A and B from choices
    cot_instruction_3 = (
        "Sub-task 3: Classify the reagents A and B from the given choices based on their chemical nature and typical roles in organic reactions, "
        "especially in cyanohydrin formation and hydrolysis."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, classifying reagents A and B, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Evaluate and Select Elements

    # Sub-task 4: Evaluate compatibility of reagent A choices
    cot_instruction_4 = (
        "Sub-task 4: Evaluate the compatibility of each choice for reagent A with the first reaction mechanism and product formation, "
        "considering the role of acid or other reagents in cyanohydrin synthesis."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating reagent A compatibility, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Evaluate compatibility of reagent B choices
    cot_instruction_5 = (
        "Sub-task 5: Evaluate the compatibility of each choice for reagent B with the second reaction mechanism and product formation, "
        "considering the role of acid or other reagents in nitrile hydrolysis to carboxylic acid."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating reagent B compatibility, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Select suitable pair of reagents A and B
    debate_instruction_6 = (
        "Sub-task 6: Based on evaluations of reagent A and B compatibility, select the suitable pair of reagents (A and B) from the given choices "
        "that correctly correspond to the reagents needed for the two reactions."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_6 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in debate_roles
    ]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent(
                    [taskInfo, thinking4, answer4, thinking5, answer5], debate_instruction_6, r, is_sub_task=True
                )
            else:
                input_infos_6 = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting suitable reagents, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6(
        [taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on suitable reagents A and B.", is_sub_task=True
    )
    agents.append(f"Final Decision agent, selecting reagents, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
