async def forward(self, taskInfo):
    from collections import Counter
    from math import comb

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Calculate the total number of ways to choose 4 numbers from the set S.
    cot_instruction_1 = "Sub-task 1: Calculate the total number of ways to choose 4 numbers from the set S."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, calculating total ways to choose 4 numbers, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Calculate the number of ways to win a prize (at least 2 of Jen's numbers are chosen).
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, calculate the number of ways to win a prize (at least 2 of Jen's numbers are chosen)."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, determining ways to win a prize, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate the number of ways to win the grand prize (all 4 of Jen's numbers are chosen).
    cot_sc_instruction_3 = "Sub-task 3: Based on the output from Sub-task 1, calculate the number of ways to win the grand prize (all 4 of Jen's numbers are chosen)."
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}

    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, determining ways to win the grand prize, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3

    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 4: Calculate the probability of winning a prize.
    cot_reflect_instruction_4 = "Sub-task 4: Based on the outputs from Sub-task 2 and Sub-task 1, calculate the probability of winning a prize."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking1, answer1]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, calculating probability of winning a prize, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback4, correct4 = critic_agent_4([taskInfo, thinking4, answer4], "please review the probability of winning a prize calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining probability of winning a prize, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate the probability of winning the grand prize.
    cot_reflect_instruction_5 = "Sub-task 5: Based on the outputs from Sub-task 3 and Sub-task 1, calculate the probability of winning the grand prize."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking3, answer3, thinking1, answer1]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, calculating probability of winning the grand prize, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback5, correct5 = critic_agent_5([taskInfo, thinking5, answer5], "please review the probability of winning the grand prize calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining probability of winning the grand prize, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration
    # Sub-task 6: Calculate the probability of winning the grand prize given that a prize is won.
    cot_reflect_instruction_6 = "Sub-task 6: Based on the outputs from Sub-task 4 and Sub-task 5, calculate the probability of winning the grand prize given that a prize is won."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round

    cot_inputs_6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, calculating conditional probability, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback6, correct6 = critic_agent_6([taskInfo, thinking6, answer6], "please review the conditional probability calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}')
        if correct6.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining conditional probability, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Express the probability from subtask_6 in the form m/n and find m+n.
    cot_reflect_instruction_7 = "Sub-task 7: Based on the output from Sub-task 6, express the probability in the form m/n and find m+n."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round

    cot_inputs_7 = [taskInfo, thinking6, answer6]
    thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_7.id}, expressing probability in m/n form, thinking: {thinking7.content}; answer: {answer7.content}')

    for i in range(N_max_7):
        feedback7, correct7 = critic_agent_7([taskInfo, thinking7, answer7], "please review the m/n expression and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}')
        if correct7.content == 'True':
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_7.id}, refining m/n expression, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer