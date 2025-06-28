async def forward_92(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze calibration curve parameters and Ct values
    # Sub-task 1: Analyze qPCR calibration curve parameters
    cot_instruction_1 = (
        "Sub-task 1: Analyze the qPCR calibration curve parameters (efficiency=100%, R2=1, slope=-3.3) "
        "and explain their expected values and implications for qPCR performance."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing calibration curve parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze provided Ct values for each concentration and triplicates
    cot_instruction_2 = (
        "Sub-task 2: Analyze the Ct values for each concentration (100000, 10000, 1000, 100, 10 copies/µl) "
        "including triplicate technical replicates, identify patterns, consistency, and expected trends relative to target copy number."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing Ct values, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Evaluate Ct values against dilution and replicates
    # Sub-task 3: Evaluate if Ct differences between dilutions ~3.3 cycles
    cot_sc_instruction_3 = (
        "Sub-task 3: Evaluate whether the Ct values correspond appropriately to the known ten-fold serial dilutions, "
        "checking if the Ct difference between each dilution step is approximately 3.3 cycles, consistent with 100% efficiency and slope -3.3."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating Ct differences per dilution, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most frequent answer
    from collections import Counter
    answer_counts_3 = Counter(possible_answers_3)
    final_answer_3 = answer_counts_3.most_common(1)[0][0]
    thinking3 = thinkingmapping_3[final_answer_3]
    answer3 = answermapping_3[final_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Evaluate consistency of technical replicates (deviation > 0.3 cycles?)
    cot_sc_instruction_4 = (
        "Sub-task 4: Evaluate the consistency of technical replicates by calculating the deviation between triplicate Ct values at each concentration "
        "and determine if the deviation exceeds 0.3 cycles, indicating poor reproducibility."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, evaluating replicate consistency, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer_counts_4 = Counter(possible_answers_4)
    final_answer_4 = answer_counts_4.most_common(1)[0][0]
    thinking4 = thinkingmapping_4[final_answer_4]
    answer4 = answermapping_4[final_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate if Ct values agree with expected inverse logarithmic relationship to copy number
    cot_sc_instruction_5 = (
        "Sub-task 5: Evaluate if the Ct values are in agreement with the amount of target nucleic acid in the samples "
        "by comparing the trend of Ct values with expected inverse logarithmic relationship to copy number."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking2, answer2], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, evaluating Ct vs copy number trend, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer_counts_5 = Counter(possible_answers_5)
    final_answer_5 = answer_counts_5.most_common(1)[0][0]
    thinking5 = thinkingmapping_5[final_answer_5]
    answer5 = answermapping_5[final_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Classify explanation for discrepancies based on previous evaluations
    debate_instruction_6 = (
        "Sub-task 6: Based on the evaluations of Ct value trends, technical replicate deviations, and expected Ct differences per dilution, "
        "classify which explanation among the provided choices best accounts for the observed discrepancies in the qPCR results."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                input_infos_6 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5]
            else:
                input_infos_6 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying explanation, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the best explanation for discrepancies.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final classification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
