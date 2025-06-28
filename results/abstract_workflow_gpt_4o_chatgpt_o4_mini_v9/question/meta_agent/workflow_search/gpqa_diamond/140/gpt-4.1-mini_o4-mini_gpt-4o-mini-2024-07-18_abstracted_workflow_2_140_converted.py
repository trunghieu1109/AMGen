async def forward_140(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze the structure and substituents of 1-bromobenzene-2-d
    cot_instruction_1 = (
        "Sub-task 1: Analyze the structure and substituents of 1-bromobenzene-2-d, "
        "identifying the position of bromine and deuterium on the benzene ring, and understand the implications of the 2-d notation (deuterium at position 2)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing structure and substituents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the nature and reactivity of NaNH2 in condensed ammonia
    cot_instruction_2 = (
        "Sub-task 2: Identify the nature and reactivity of the reagent NaNH2 in condensed ammonia solvent, "
        "focusing on its role as a strong base and nucleophile, and typical reaction mechanisms it promotes with aryl halides."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing reagent reactivity, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Classify possible reaction pathways of 1-bromobenzene-2-d with NaNH2
    cot_sc_instruction_3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, classify the possible reaction pathways of 1-bromobenzene-2-d with NaNH2 in condensed ammonia, "
        "including elimination, nucleophilic aromatic substitution, or metalation, considering substrate and reagent properties."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, classifying reaction pathways, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most frequent answer for consistency
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3_content = thinkingmapping_3[answer3_content].content
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_content}; answer - {answer3_content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Determine and Evaluate Products

    # Sub-task 4: Determine possible organic products formed from each reaction pathway
    cot_instruction_4 = (
        "Sub-task 4: Determine the possible organic products formed from each identified reaction pathway of 1-bromobenzene-2-d with NaNH2, "
        "considering isotopic labeling (deuterium) and positional effects on the benzene ring."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answermapping_3[answer3_content]], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, determining possible products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate and count distinct organic products formed
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, evaluate and count the distinct organic products formed, "
        "considering stereochemistry, isotopic substitution, and structural isomers to avoid overcounting identical products."
    )
    debate_roles = ["Pro", "Con"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating and counting products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the number of distinct organic products.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding number of products, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare the number of possible products with given multiple-choice options
    cot_instruction_6 = (
        "Sub-task 6: Compare the number of possible organic products obtained with the given multiple-choice options (1, 2, 3, 4) to select the correct answer."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting correct multiple-choice answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
