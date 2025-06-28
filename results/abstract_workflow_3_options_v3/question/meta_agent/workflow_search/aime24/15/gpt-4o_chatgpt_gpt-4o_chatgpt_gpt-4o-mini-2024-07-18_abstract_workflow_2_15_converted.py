async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning

    # Sub-task 1: Calculate the total number of residents who own at least one of the three items.
    cot_instruction_1 = "Sub-task 1: Calculate the total number of residents who own at least one of the three items (diamond ring, golf clubs, garden spade), excluding the bag of candy hearts."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating total residents with at least one item, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Calculate the number of residents who own exactly one of the three items.
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, calculate the number of residents who own exactly one of the three items."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, determining residents with exactly one item, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Calculate the number of residents who own exactly two of the three items.
    cot_instruction_3 = "Sub-task 3: Calculate the number of residents who own exactly two of the three items."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, calculating residents with exactly two items, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Calculate the number of residents who own exactly three of the three items.
    cot_instruction_4 = "Sub-task 4: Calculate the number of residents who own exactly three of the three items."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, calculating residents with exactly three items, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Stage 2: Intermediate Inference and Validation

    # Sub-task 5: Calculate the number of residents who own at least one of the three items using the inclusion-exclusion principle.
    cot_reflect_instruction_5 = "Sub-task 5: Based on the outputs from Sub-task 1, Sub-task 2, Sub-task 3, and Sub-task 4, calculate the number of residents who own at least one of the three items using the inclusion-exclusion principle."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating residents with at least one item using inclusion-exclusion, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review the inclusion-exclusion calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining inclusion-exclusion calculation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Sub-task 6: Calculate the number of residents who own all three items (diamond ring, golf clubs, garden spade).
    cot_instruction_6 = "Sub-task 6: Calculate the number of residents who own all three items (diamond ring, golf clubs, garden spade)."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, calculating residents with all three items, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Stage 3: Final Output Generation and Integration

    # Sub-task 7: Calculate the number of residents who own all four items (including the bag of candy hearts).
    debate_instruction_7 = "Sub-task 7: Based on the output of sub-task 6, calculate the number of residents who own all four items (including the bag of candy hearts)."
    debate_agents_7 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round

    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for agent in debate_agents_7:
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7.extend(all_thinking7[r-1])
            thinking7, answer7 = agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating residents with all four items, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_answers_7 = [ans.content for ans in all_answer7[-1]]
    final_answercontent_7 = Counter(final_answers_7).most_common(1)[0][0]
    index_7 = final_answers_7.index(final_answercontent_7)
    thinking7 = all_thinking7[-1][index_7]
    answer7 = all_answer7[-1][index_7]
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer