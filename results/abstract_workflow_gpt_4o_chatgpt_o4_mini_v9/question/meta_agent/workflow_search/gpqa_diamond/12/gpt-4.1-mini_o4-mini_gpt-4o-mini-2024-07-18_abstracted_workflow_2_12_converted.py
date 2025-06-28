async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify elements

    # Sub-task 1: Analyze (R)-(+)-Limonene structure and stereochemistry, identify functional groups and stereocenters
    cot_instruction_1 = (
        "Sub-task 1: Analyze the starting material (R)-(+)-Limonene structure and stereochemistry, "
        "identify functional groups and stereocenters relevant to the subsequent reactions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing (R)-(+)-Limonene, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine structural and stereochemical changes after partial hydrogenation with Pd/C (1 equiv H2)
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis of (R)-(+)-Limonene, determine the structural and stereochemical changes "
        "after partial hydrogenation with Pd/C under hydrogen atmosphere (1 equivalent H2 consumed), leading to product 1. "
        "Consider possible sites of hydrogen addition and stereochemical outcomes."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining hydrogenation changes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze reaction of product 1 with mCPBA to form product 2 (epoxidation and stereochemistry)
    cot_instruction_3 = (
        "Sub-task 3: Analyze the reaction of product 1 with 3-chloroperbenzoic acid (mCPBA) to form product 2, "
        "focusing on the type of oxidation (epoxidation or other) and stereochemical outcome."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing epoxidation with mCPBA, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze reaction of product 2 with sodium methoxide to form product 3 (nucleophilic substitution or ring-opening and stereochemistry)
    cot_instruction_4 = (
        "Sub-task 4: Analyze the reaction of product 2 with sodium methoxide to form product 3, "
        "identifying the nucleophilic substitution or ring-opening mechanism and resulting stereochemistry."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing nucleophilic substitution/ring-opening, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Analyze esterification of product 3 with propanoic acid, DCC, and DMAP to form product 4 (site of esterification and stereochemical implications)
    debate_instruction_5 = (
        "Sub-task 5: Based on product 3, analyze the esterification reaction with propanoic acid, DCC, and catalytic DMAP to form product 4, "
        "focusing on the site of esterification and stereochemical implications."
    )
    debate_roles = ["Pro-Esterification at Primary Alcohol", "Pro-Esterification at Secondary Alcohol"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing esterification site and stereochemistry, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the valid structure and stereochemistry of product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding valid structure of product 4, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Integrate and compare with choices

    # Sub-task 6: Integrate all structural and stereochemical info to deduce valid structure(s) of product 4
    cot_reflect_instruction_6 = (
        "Sub-task 6: Integrate the structural and stereochemical information from all previous steps to deduce the valid structure(s) of product 4, "
        "considering possible isomers formed."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, integrating all info for product 4 structure, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Compare deduced structure(s) with given multiple-choice options to select correct answer
    debate_instruction_7 = (
        "Sub-task 7: Compare the deduced structure(s) of product 4 with the given multiple-choice options, "
        "evaluating stereochemistry, substituents, and functional groups to select the correct answer."
    )
    debate_roles_7 = ["Pro-Choice1", "Pro-Choice2", "Pro-Choice3", "Pro-Choice4"]
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles_7]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Select the correct answer choice for product 4 structure.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct answer choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
