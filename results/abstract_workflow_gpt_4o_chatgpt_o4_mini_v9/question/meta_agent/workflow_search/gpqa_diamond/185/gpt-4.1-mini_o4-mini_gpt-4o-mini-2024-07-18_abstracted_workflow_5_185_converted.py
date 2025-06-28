async def forward_185(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze the structure and stereochemistry of the starting compound
    cot_instruction_0 = (
        "Sub-task 1: Analyze the structure and stereochemistry of the starting compound (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene to identify its key structural features, "
        "including the bicyclic framework, vinyl substituent, nitrogen position, and stereochemical configuration."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing starting compound structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: Analyze mechanism and classify possible rearranged intermediates
    # Sub-task 2: Analyze Cope rearrangement mechanism with Self-Consistency Chain-of-Thought
    cot_sc_instruction_1 = (
        "Sub-task 2: Analyze the mechanism and characteristics of the Cope rearrangement, specifically how it applies to bicyclic azabicyclo compounds with vinyl substituents, "
        "to understand possible bond shifts and stereochemical outcomes, based on Sub-task 1 output."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    thinkingmapping_1 = {}
    answermapping_1 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_1[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing Cope rearrangement mechanism, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_1.append(answer2.content)
        thinkingmapping_1[answer2.content] = thinking2
        answermapping_1[answer2.content] = answer2
    most_common_answer_1 = Counter(possible_answers_1).most_common(1)[0][0]
    thinking2 = thinkingmapping_1[most_common_answer_1]
    answer2 = answermapping_1[most_common_answer_1]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 3: Classify possible structural changes and rearranged intermediates using Reflexion
    cot_reflect_instruction_1 = (
        "Sub-task 3: Based on outputs from Sub-task 2, classify the possible structural changes and rearranged intermediates from the Cope rearrangement, "
        "considering stereochemistry and ring strain effects."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, classifying rearranged intermediates, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                               "Please review the classification of possible rearranged intermediates and provide limitations.", 
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining classification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Generate possible product structures from Cope rearrangement
    debate_instruction_4 = (
        "Sub-task 4: Generate possible product structures resulting from the Cope rearrangement of (1S,4R)-2-vinyl-2-azabicyclo[2.2.1]hept-5-ene by applying the rearrangement mechanism "
        "to the identified starting structure and predicted intermediates from Sub-task 3."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
                input_infos_4.extend(all_answer4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating product structures, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                    "Sub-task 4: Make a final decision on the generated product structures.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on product generation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 3: Evaluate generated products against given choices
    debate_instruction_5 = (
        "Sub-task 5: Evaluate the generated product structures against the given choices (choice1 to choice4) by comparing their structural features, hydrogenation states (tetrahydro), "
        "ring fusion patterns, and nomenclature to identify the correct product of the Cope rearrangement."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
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
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                    "Sub-task 5: Make final decision on the correct product of the Cope rearrangement.", 
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent on product evaluation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
