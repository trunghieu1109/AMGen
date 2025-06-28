async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction1 = "Sub-task 1: Identify the full structure of (R)-(+)-limonene, its double bond positions and stereocenter configuration."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, identifying (R)-(+)-limonene structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 2: Classify each reaction step (Pd/C hydrogenation, m-CPBA epoxidation, sodium methoxide ring opening, DCC/DMAP esterification) with reagents, conditions, and reaction types."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible2 = []
    thinkingmap2 = {}
    answermap2 = {}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, classifying steps, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible2.append(answer2_i.content)
        thinkingmap2[answer2_i.content] = thinking2_i
        answermap2[answer2_i.content] = answer2_i
    majority2 = Counter(possible2).most_common(1)[0][0]
    thinking2 = thinkingmap2[majority2]
    answer2 = answermap2[majority2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction3 = "Sub-task 3: Analyze how each reaction step modifies limonene: which alkene is hydrogenated first, epoxide formation site, regioselectivity of ring opening, and esterification site."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N3 = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, analyzing modifications, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Review modifications analysis for completeness and accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback on analysis, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Determine structure and stereochemistry of product 1 after partial hydrogenation of limonene."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, determining product 1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction5 = "Sub-task 5: Deduce structure and stereochemistry of product 2 from epoxidation of product 1."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible5 = []
    thinkingmap5 = {}
    answermap5 = {}
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, deducing product 2, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible5.append(answer5_i.content)
        thinkingmap5[answer5_i.content] = thinking5_i
        answermap5[answer5_i.content] = answer5_i
    majority5 = Counter(possible5).most_common(1)[0][0]
    thinking5 = thinkingmap5[majority5]
    answer5 = answermap5[majority5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction6 = "Sub-task 6: Determine regiochemical and stereochemical outcome of epoxide ring opening in product 2 with sodium methoxide."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N6 = self.max_round
    cot_inputs6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, ring opening outcome, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N6):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Evaluate ring opening prediction for correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining ring opening, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    debate_instruction7 = "Sub-task 7: Predict structure and stereochemistry of product 4 from esterification of product 3 with propionic acid using DCC/DMAP."
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R7 = self.max_round
    all_thought7 = [[] for _ in range(R7)]
    all_answer7 = [[] for _ in range(R7)]
    for r in range(R7):
        for agent in debate_agents7:
            if r == 0:
                t7, a7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6, answer6] + all_thought7[r-1] + all_answer7[r-1]
                t7, a7 = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, predicting product 4, thinking: {t7.content}; answer: {a7.content}")
            all_thought7[r].append(t7)
            all_answer7[r].append(a7)
    final_decision7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision7([taskInfo] + all_thought7[-1] + all_answer7[-1], "Sub-task 7: Decide on predicted structure of product 4.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision7.id}, deciding product 4, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    cot_instruction8 = "Sub-task 8: Compare predicted product 4 structure to the provided choices and select the matching choice."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, selecting matching choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer