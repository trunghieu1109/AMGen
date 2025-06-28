async def forward_45(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze and Classify Elements

    # Sub-task 1: Analyze the structure and stereochemistry of racemic 3-methylpent-1-ene
    cot_instruction_1 = (
        "Sub-task 1: Analyze the structure and stereochemistry of racemic 3-methylpent-1-ene, "
        "identifying all relevant functional groups, double bond position, and chiral centers to understand the starting material's attributes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing racemic 3-methylpent-1-ene structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Classify the type of reaction with Grubbs catalyst
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the analysis of racemic 3-methylpent-1-ene, classify the type of reaction that occurs when treated with Grubbs catalyst, "
        "specifically identifying the metathesis reaction mechanism and its implications on the substrate."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, classifying metathesis reaction, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer
    from collections import Counter
    answer_counts_2 = Counter(possible_answers_2)
    final_answer_2 = answer_counts_2.most_common(1)[0][0]
    thinking2_final = thinkingmapping_2[final_answer_2]
    answer2_final = answermapping_2[final_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {final_answer_2}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Determine pathways and products

    # Sub-task 3: Determine all possible intra- and intermolecular metathesis pathways
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on the metathesis reaction classification and substrate stereochemistry, determine all possible intramolecular and intermolecular metathesis pathways for racemic 3-methylpent-1-ene under Grubbs catalyst conditions. "
        "Consider stereochemistry and double bond location from previous analysis."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2_final, answer2_final]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining metathesis pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3],
                                                  "Please review the proposed metathesis pathways and provide limitations or confirm correctness.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback on pathways, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining pathways, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Enumerate all possible cyclic and acyclic products formed, excluding ethene
    cot_reflect_instruction_4 = (
        "Sub-task 4: Enumerate all possible cyclic and acyclic products formed from the metathesis reactions identified in Sub-task 3, "
        "excluding ethene, and consider stereoisomeric possibilities arising from the racemic mixture."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, enumerating products, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "Please review the product enumeration for completeness and correctness, excluding ethene.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback on products, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate and count distinct possible products formed, excluding ethene
    debate_instruction_5 = (
        "Sub-task 5: Based on the product enumeration from Sub-task 4, evaluate and count the distinct possible products formed from the metathesis of racemic 3-methylpent-1-ene, "
        "ensuring no duplicates and excluding ethene, to determine the total number of unique products."
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating product count, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the total number of unique products excluding ethene.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final product count, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
