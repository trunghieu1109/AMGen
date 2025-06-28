async def forward_110(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and identify chemical structures and reaction conditions

    # Sub-task 1: Analyze and identify chemical structures and functional groups of reactants
    cot_instruction_1 = (
        "Sub-task 1: Analyze and identify the chemical structures and functional groups of the reactants: "
        "2-ethyl-2,6-dimethylcyclohexan-1-one and ethyl acrylate, and 1-nitropropane and (E)-but-2-enenitrile. "
        "Classify their reactive sites and relevant steric and electronic features that influence reaction pathways."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing chemical structures and functional groups, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze reaction conditions and reagents
    cot_instruction_2 = (
        "Sub-task 2: Analyze the reaction conditions and reagents (t-BuOK for the first reaction, KOH and H2O for the second reaction) "
        "to understand their roles in the reaction mechanisms and possible product formation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing reaction conditions and reagents, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Predict reaction mechanisms and intermediates

    # Sub-task 3: Predict mechanism for 2-ethyl-2,6-dimethylcyclohexan-1-one + ethyl acrylate under t-BuOK
    cot_sc_instruction_3 = (
        "Sub-task 3: Predict the reaction mechanism and intermediate species for the reaction between "
        "2-ethyl-2,6-dimethylcyclohexan-1-one and ethyl acrylate under t-BuOK conditions, considering steric hindrance and electronic effects identified in stage_0."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, predicting mechanism for first reaction, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose most consistent answer
    most_common_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[most_common_3]
    answer3_final = answermapping_3[most_common_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Predict mechanism for 1-nitropropane + (E)-but-2-enenitrile under KOH and H2O
    cot_sc_instruction_4 = (
        "Sub-task 4: Predict the reaction mechanism and intermediate species for the reaction between "
        "1-nitropropane and (E)-but-2-enenitrile under KOH and H2O conditions, considering steric and electronic factors identified in stage_0."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting mechanism for second reaction, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[most_common_4]
    answer4_final = answermapping_4[most_common_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Evaluate possible products from both reactions

    # Sub-task 5: Evaluate products from first reaction (A)
    debate_instruction_5 = (
        "Sub-task 5: Evaluate the possible products from the first reaction (A) by analyzing the stability of the products "
        "and the influence of steric hindrance on the major product formation, using the predicted intermediates and mechanisms from subtask_3."
    )
    debate_roles = ["Pro Product 1", "Pro Product 2"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking3_final, answer3_final], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking3_final, answer3_final] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating products of first reaction, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on major product of first reaction.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding major product of first reaction, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Evaluate products from second reaction (B)
    debate_instruction_6 = (
        "Sub-task 6: Evaluate the possible products from the second reaction (B) by analyzing the stability of the products "
        "and the influence of steric hindrance on the major product formation, using the predicted intermediates and mechanisms from subtask_4."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    all_thinking6 = [[] for _ in range(N_max_5)]
    all_answer6 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4_final, answer4_final], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4_final, answer4_final] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating products of second reaction, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on major product of second reaction.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding major product of second reaction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 3: Compare evaluated major products with given choices and select correct combination

    reflexion_instruction_7 = (
        "Sub-task 7: Compare the evaluated major products of reactions A and B with the given choices, "
        "and select the correct combination of products that best fits the steric and stability considerations."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], reflexion_instruction_7, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, selecting correct product combination, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
