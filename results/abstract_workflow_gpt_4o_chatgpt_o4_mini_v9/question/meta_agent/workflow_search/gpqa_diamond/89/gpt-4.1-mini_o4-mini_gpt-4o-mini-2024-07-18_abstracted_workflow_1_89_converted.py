async def forward_89(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze starting compound and reagents
    # Sub-task 1: Analyze chemical structure and functional groups of 3,4-dimethylhexanedial
    cot_instruction_1 = (
        "Sub-task 1: Analyze and identify the chemical structure and functional groups of the starting compound 3,4-dimethylhexanedial, "
        "including the positions and nature of aldehyde groups and methyl substituents, to establish a clear baseline for subsequent chemical transformations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing starting compound, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Examine and classify each reagent and reaction condition
    cot_instruction_2 = (
        "Sub-task 2: Examine and classify each reagent and reaction condition (KOH, H2O, THF, Heat; CH3CH2MgBr, H3O+; PCC, CH2Cl2; O3, H2O) "
        "to understand their typical chemical behavior and expected transformations on aldehydes or related functional groups."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, classifying reagents and conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Predict products of each reaction step
    # Sub-task 3: Predict product after treatment with KOH, H2O, THF, Heat
    cot_sc_instruction_3 = (
        "Sub-task 3: Predict the product formed when 3,4-dimethylhexanedial is treated with KOH, H2O, THF under heat, "
        "considering base-catalyzed reactions possible on dialdehydes such as aldol condensation or Cannizzaro reaction."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, predicting product after KOH treatment, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose most common answer
    most_common_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_3]
    answer3 = answermapping_3[most_common_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine product after reaction with CH3CH2MgBr, H3O+
    cot_sc_instruction_4 = (
        "Sub-task 4: Determine the product obtained when the intermediate from sub-task 3 is reacted with ethylmagnesium bromide (CH3CH2MgBr) followed by acidic workup (H3O+), "
        "focusing on nucleophilic addition to carbonyl groups and possible formation of alcohols."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting product after Grignard addition, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_4]
    answer4 = answermapping_4[most_common_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Analyze oxidation with PCC in CH2Cl2
    cot_sc_instruction_5 = (
        "Sub-task 5: Analyze the oxidation of the product from sub-task 4 using PCC in CH2Cl2, "
        "predicting conversion of primary or secondary alcohols to aldehydes or ketones, and identify resulting functional groups and structure."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting product after PCC oxidation, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    most_common_5 = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[most_common_5]
    answer5 = answermapping_5[most_common_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Predict final product after ozonolysis (O3, H2O)
    cot_sc_instruction_6 = (
        "Sub-task 6: Predict the final product formed after ozonolysis (O3, H2O) of the compound obtained in sub-task 5, "
        "considering cleavage of double bonds or other reactive sites to form aldehydes, ketones, or carboxylic acids."
    )
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    for i in range(N):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking5, answer5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, predicting product after ozonolysis, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    most_common_6 = Counter(possible_answers_6).most_common(1)[0][0]
    thinking6 = thinkingmapping_6[most_common_6]
    answer6 = answermapping_6[most_common_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Compare final predicted product with multiple-choice options
    # Sub-task 7: Compare and identify correct product
    debate_instruction_7 = (
        "Sub-task 7: Compare the final predicted product structure from sub-task 6 with the given multiple-choice options: "
        "3,4-dimethyl-5,6-dioxooctanal; 3,4-dimethyl-5,6-dioxooctanoic acid; 4,5-dimethylnonane-2,6,7-trione (listed twice). "
        "Identify the correct product based on functional groups, carbon chain length, and substitution pattern."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing final product with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct product choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on product choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
