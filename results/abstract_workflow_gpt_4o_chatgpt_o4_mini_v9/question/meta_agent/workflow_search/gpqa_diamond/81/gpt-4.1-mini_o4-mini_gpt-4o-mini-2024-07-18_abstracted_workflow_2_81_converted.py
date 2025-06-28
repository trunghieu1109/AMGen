async def forward_81(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and deduce structures stepwise

    # Sub-task 1: Analyze initial reactants cyclooctatetraene and maleic anhydride, determine product 1 structure and stereochemistry
    cot_instruction_1 = (
        "Sub-task 1: Analyze the reaction of cyclooctatetraene with maleic anhydride in 1:1 ratio upon heating. "
        "Determine the type of reaction (likely Diels-Alder), the structure and stereochemistry of product 1 formed. "
        "Provide detailed reasoning on the adduct structure formed."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing initial Diels-Alder reaction, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze transformation of product 1 with methanol and sulfuric acid to product 2
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on product 1 structure, analyze the reaction when heated with methanol and a small amount of sulfuric acid. "
        "Determine chemical changes such as esterification or ring opening, and deduce the structure and stereochemistry of product 2. "
        "Consider all plausible mechanisms and outcomes."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing methanol/sulfuric acid transformation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Select most consistent answer by frequency
    answer2_counter = Counter(possible_answers_2)
    major_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[major_answer2]
    answer2 = answermapping_2[major_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze reaction of product 2 with cyclopentadiene upon heating to form product 3
    cot_instruction_3 = (
        "Sub-task 3: Based on product 2 structure, analyze the reaction with cyclopentadiene upon heating. "
        "Identify the reaction type (likely Diels-Alder), regiochemistry, stereochemistry of the adduct, and deduce the structure of product 3. "
        "Provide detailed reasoning on the major isomer formed."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing final Diels-Alder reaction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate stereochemical configurations and compare with choices

    # Sub-task 4: Evaluate stereochemical configurations of major isomer of product 3
    cot_reflect_instruction_4 = (
        "Sub-task 4: Evaluate the stereochemical configurations of the major isomer of product 3 by comparing possible stereoisomers generated from the reaction sequence. "
        "Use knowledge of stereochemical outcomes of Diels-Alder reactions and reaction conditions to prioritize the most likely major isomer."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, evaluating stereochemistry, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "Please review the stereochemical evaluation and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining stereochemical evaluation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare deduced major isomer structure of product 3 with provided choices
    debate_instruction_5 = (
        "Sub-task 5: Based on the stereochemical evaluation, compare the deduced major isomer structure of product 3 with the provided choices (choice1 to choice4). "
        "Analyze their IUPAC names and SMILES strings to identify which corresponds to the major isomer formed. "
        "Provide a reasoned final decision."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                    "Sub-task 5: Make final decision on the major isomer structure of product 3 among the given choices.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
