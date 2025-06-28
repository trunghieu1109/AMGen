async def forward_90(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze initial reactants and reagents
    # Sub-task 1: Analyze cyclohexanone structural features relevant to enolate formation using Chain-of-Thought
    cot_instruction_1 = "Sub-task 1: Analyze the initial reactant cyclohexanone and identify its key structural features relevant to enolate formation and subsequent reactions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing cyclohexanone features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Characterize LDA as strong, non-nucleophilic base and its effect on cyclohexanone at low temp using Chain-of-Thought
    cot_instruction_2 = "Sub-task 2: Characterize the reagent LDA (lithium diisopropylamide), focusing on its role as a strong, non-nucleophilic base and its effect on cyclohexanone at low temperature."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, characterizing LDA, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify benzaldehyde features relevant to nucleophilic addition using Chain-of-Thought
    cot_instruction_3 = "Sub-task 3: Identify the structural and reactive features of benzaldehyde relevant to nucleophilic addition reactions with enolates."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing benzaldehyde features, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Analyze DAST and its fluorination reactivity using Chain-of-Thought
    cot_instruction_4 = "Sub-task 4: Analyze diethylaminosulfur trifluoride (DAST) and its typical reactivity patterns, especially its ability to convert hydroxyl groups to fluorides and other fluorination transformations."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, analyzing DAST reactivity, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Predict products
    # Sub-task 5: Determine product 1 formed after LDA, benzaldehyde, acidification using Self-Consistency Chain-of-Thought
    cot_sc_instruction_5 = "Sub-task 5: Determine the product formed (product 1) when cyclohexanone is treated with LDA at low temperature to form the enolate, followed by reaction with benzaldehyde and acidification, considering stereochemistry and functional groups."
    N = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting product 1, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    # Choose most common answer
    answer5_final = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5_final = thinkingmapping_5[answer5_final]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_final.content}; answer - {answer5_final}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Analyze transformation of product 1 with excess DAST to product 2 using Reflexion
    cot_reflect_instruction_6 = "Sub-task 6: Analyze the transformation of product 1 when treated with an excess of DAST, predicting the structural changes, including fluorination sites and stereochemical outcomes to form product 2."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking4, answer4, thinking5_final, answermapping_5[answer5_final]]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, analyzing DAST transformation, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the predicted DAST transformation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining DAST transformation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Compare predicted product 2 with multiple-choice options using Debate
    debate_instruction_7 = "Sub-task 7: Compare the predicted structure of product 2 with the given multiple-choice options, evaluating stereochemistry, functional groups, and overall molecular structure to identify the correct product."
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
            agents.append(f"Debate agent {agent.id}, round {r}, comparing product 2 with choices, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct structure of product 2.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct product 2, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer