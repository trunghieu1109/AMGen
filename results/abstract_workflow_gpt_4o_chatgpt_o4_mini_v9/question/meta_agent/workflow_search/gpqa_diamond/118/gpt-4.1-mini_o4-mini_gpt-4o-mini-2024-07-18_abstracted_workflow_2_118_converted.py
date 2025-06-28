async def forward_118(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze starting material and reagents
    cot_instruction_1 = (
        "Sub-task 1: Analyze the starting material 5-bromo-3a,4a-dimethyldecahydrocyclopenta[1,4]cyclobuta[1,2]benzene to identify its structural features, "
        "functional groups, and stereochemistry relevant to the reaction sequence."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting material, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Analyze the reagents and conditions (H2O, PDC, H2CPPh3, TsOH) to understand their chemical roles and expected transformations on the substrate or intermediates."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing reagents and conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Predict intermediates and final product
    cot_sc_instruction_3 = (
        "Sub-task 3: Predict the structure of intermediate A formed by the reaction of the starting material with H2O, "
        "based on the functional groups and reaction conditions identified in subtask_1 and subtask_2."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, predicting intermediate A, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    answer3_final = counter_3.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[answer3_final]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Predict the structure of intermediate B formed by oxidation of A with PDC, considering the functional groups introduced or modified in subtask_3 and the oxidizing nature of PDC."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3_final, answermapping_3[answer3_final], thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting intermediate B, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    counter_4 = Counter(possible_answers_4)
    answer4_final = counter_4.most_common(1)[0][0]
    thinking4_final = thinkingmapping_4[answer4_final]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Predict the structure of intermediate C formed by the reaction of B with H2CPPh3 (Wittig reagent), "
        "using the knowledge of the intermediate B structure and the Wittig reaction mechanism."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4_final, answermapping_4[answer4_final], thinking2, answer2], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting intermediate C, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    counter_5 = Counter(possible_answers_5)
    answer5_final = counter_5.most_common(1)[0][0]
    thinking5_final = thinkingmapping_5[answer5_final]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Sub-task 6: Predict the final product D formed by the acid-catalyzed reaction of C with TsOH, "
        "considering possible cyclizations, rearrangements, or eliminations under acidic conditions."
    )
    debate_roles = ["Proposer", "Opponent"]
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5_final, answermapping_5[answer5_final], thinking2, answer2], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5_final, answermapping_5[answer5_final], thinking2, answer2] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, predicting final product D, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the predicted final product D.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding final product D, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 3: Compare predicted product D with given choices
    cot_reflect_instruction_7 = (
        "Sub-task 7: Compare the predicted structure of product D with the given choices by analyzing their IUPAC names and structural features to identify the correct product."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_reflect_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, comparing predicted product D with choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
