async def forward_145(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and Classify Elements

    # Sub-task 1: Analyze 5-fluorocyclopenta-1,3-diene structure and features with Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Analyze the chemical structure and key features of 5-fluorocyclopenta-1,3-diene, "
        "including the position and stereochemical implications of the fluorine substituent, to understand its reactivity and possible sites for reaction."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing 5-fluorocyclopenta-1,3-diene, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Analyze maleic anhydride structure and dienophile characteristics with Chain-of-Thought
    cot_instruction_2 = (
        "Sub-task 2: Analyze the chemical structure and reactive sites of maleic anhydride, "
        "focusing on its dienophile characteristics relevant to Diels-Alder reactions."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing maleic anhydride, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Classify the reaction type considering diene, dienophile, and fluorine substituent influence with Self-Consistency Chain-of-Thought
    cot_sc_instruction_3 = (
        "Sub-task 3: Classify the type of reaction expected between 5-fluorocyclopenta-1,3-diene and maleic anhydride, "
        "considering the diene and dienophile nature of the reactants and the influence of the fluorine substituent on regio- and stereoselectivity."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, classifying reaction type, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Choose the most common answer
    answer3_counter = Counter(possible_answers_3)
    final_answer3 = answer3_counter.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[final_answer3]
    answer3_final = answermapping_3[final_answer3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Predict and Compare Products

    # Sub-task 4: Predict major Diels-Alder adduct with stereochemistry using Reflexion
    cot_reflect_instruction_4 = (
        "Sub-task 4: Predict the major Diels-Alder adduct formed from the reaction of 5-fluorocyclopenta-1,3-diene with maleic anhydride, "
        "including the stereochemistry at all relevant chiral centers, based on the analysis of substituent effects and reaction mechanism."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3_final, answer3_final]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, predicting major Diels-Alder adduct, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                 "Please review the predicted major Diels-Alder adduct and its stereochemistry, provide limitations or corrections.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining prediction, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compare predicted product with given choices using Debate
    debate_instruction_5 = (
        "Sub-task 5: Compare the predicted major product stereochemistry and structure with the given four choices, "
        "identifying which choice matches the predicted product."
    )
    debate_roles = ["Proponent", "Opponent"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing predicted product with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on which choice matches the predicted major product.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, deciding matching choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
