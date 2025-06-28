async def forward_33(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the Pinacol rearrangement reaction description and identify key mechanistic features
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given Pinacol rearrangement reaction description and identify the key mechanistic features, "
        "including the role of acid, formation of carbocation intermediate, and group migration, to understand the reaction context and constraints."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Pinacol rearrangement mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze and classify the structures of vicinal diols A, B, and C
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the mechanistic understanding from Sub-task 1, analyze and classify the structures of the three given vicinal diols (A, B, and C) "
        "in terms of their substituents, stereochemistry, and possible carbocation formation sites relevant to the Pinacol rearrangement."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing vicinal diols A, B, C, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most common answer for consistency
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_answer_2].content}; answer - {most_common_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Predict intermediates and products, then select correct choice

    # Sub-task 3: Predict carbocation intermediates for A, B, and C
    cot_sc_instruction_3 = (
        "Sub-task 3: For each vicinal diol (A, B, and C), predict the carbocation intermediate formed upon protonation and dehydration, "
        "considering the stability and possible group shifts during the Pinacol rearrangement."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, predicting carbocation intermediates, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinkingmapping_3[most_common_answer_3].content}; answer - {most_common_answer_3}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine possible rearranged ketone products for A, B, and C
    cot_sc_instruction_4 = (
        "Sub-task 4: Based on the predicted carbocation intermediates, determine the possible rearranged ketone products for each compound (A, B, and C), "
        "including the position of the carbonyl group and the migrated substituents."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinkingmapping_3[most_common_answer_3], answermapping_3[most_common_answer_3]], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, determining ketone products, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinkingmapping_4[most_common_answer_4].content}; answer - {most_common_answer_4}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare predicted ketone products with given multiple-choice options to identify correct choice
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, compare the predicted ketone products for A, B, and C with the given multiple-choice options "
        "(choice1 to choice4) to identify which choice correctly matches the expected Pinacol rearrangement products."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinkingmapping_4[most_common_answer_4], answermapping_4[most_common_answer_4]], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinkingmapping_4[most_common_answer_4], answermapping_4[most_common_answer_4]] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted products with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on which choice correctly matches the expected Pinacol rearrangement products.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice decision, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
