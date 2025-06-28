async def forward_39(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and Identify Phrase Meaning
    # Sub-task 1: Extract and define the key phrase 'my compounds are on top of each other' from the chemists statement
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and define the key phrase 'my compounds are on top of each other' from the chemists statement, "
        "understanding its context within synthetic organic chemistry lab work and typical chemist communication."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting phrase meaning, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Identify and list possible meanings or interpretations of the phrase in synthetic organic chemistry context
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify and list possible meanings or interpretations of the phrase 'compounds are on top of each other' "
        "in the context of synthetic organic chemistry, focusing on physical and chemical properties relevant to compound separation and analysis."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    thinkingmapping_0_2 = {}
    answermapping_0_2 = {}
    for i in range(N_sc_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, listing phrase interpretations, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2.content)
        thinkingmapping_0_2[answer_0_2.content] = thinking_0_2
        answermapping_0_2[answer_0_2.content] = answer_0_2
    most_common_answer_0_2 = Counter(possible_answers_0_2).most_common(1)[0][0]
    thinking_0_2 = thinkingmapping_0_2[most_common_answer_0_2]
    answer_0_2 = answermapping_0_2[most_common_answer_0_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Analyze and Classify Given Choices
    # Sub-task 3: Analyze the four given choices to understand their chemical significance and relation to phrase
    cot_instruction_1_3 = (
        "Sub-task 3: Analyze the four given choices (non-covalent/van der Waals interactions, similar polarities, similar optical rotations, similar boiling points) "
        "to understand their chemical significance and how each could relate to compounds being 'on top of each other'."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, analyzing choices, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Classify each choice based on relevance to lab techniques where compounds might be described as on top of each other
    cot_instruction_1_4 = (
        "Sub-task 4: Classify each choice based on its relevance to common laboratory techniques or observations where compounds might be described as 'on top of each other', "
        "such as chromatography or physical separation methods."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, classifying choices, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Transform Understanding into Practical Scenarios
    # Sub-task 5: Transform understanding of each choice into practical lab scenarios to evaluate best fit
    debate_instruction_2_5 = (
        "Sub-task 5: Based on the classification from Sub-task 4, convert the understanding of each choice into practical scenarios in synthetic organic chemistry labs, "
        "such as how similar polarity affects chromatographic separation or how similar boiling points affect distillation, to evaluate which scenario best fits the phrase 'compounds are on top of each other'."
    )
    debate_agents_2_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_5 = self.max_round
    all_thinking_2_5 = [[] for _ in range(N_max_2_5)]
    all_answer_2_5 = [[] for _ in range(N_max_2_5)]
    for r in range(N_max_2_5):
        for i, agent in enumerate(debate_agents_2_5):
            if r == 0:
                thinking_2_5, answer_2_5 = await agent([taskInfo, thinking_1_4, answer_1_4], debate_instruction_2_5, r, is_sub_task=True)
            else:
                input_infos_2_5 = [taskInfo, thinking_1_4, answer_1_4] + all_thinking_2_5[r-1] + all_answer_2_5[r-1]
                thinking_2_5, answer_2_5 = await agent(input_infos_2_5, debate_instruction_2_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, transforming understanding into practical scenarios, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
            all_thinking_2_5[r].append(thinking_2_5)
            all_answer_2_5[r].append(answer_2_5)
    final_decision_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await final_decision_agent_2_5([taskInfo] + all_thinking_2_5[-1] + all_answer_2_5[-1], "Sub-task 5: Make final decision on which practical scenario best fits the phrase 'my compounds are on top of each other'.", is_sub_task=True)
    agents.append(f"Final Decision agent on practical scenarios, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    # Stage 3: Evaluate and Prioritize Choices
    # Sub-task 6: Evaluate and prioritize the four choices by comparing their likelihood and relevance to the phrase
    cot_reflect_instruction_3_6 = (
        "Sub-task 6: Evaluate and prioritize the four choices by comparing their likelihood and relevance to the phrase 'my compounds are on top of each other' "
        "as used by chemists discussing lab results, selecting the most plausible explanation."
    )
    cot_agent_3_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_3_6, answer_3_6 = await cot_agent_3_6([taskInfo, thinking_2_5, answer_2_5], cot_reflect_instruction_3_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_6.id}, evaluating and prioritizing choices, thinking: {thinking_3_6.content}; answer: {answer_3_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_3_6.content}; answer - {answer_3_6.content}")
    print("Step 3.6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_6, answer_3_6, sub_tasks, agents)
    return final_answer