async def forward(self, taskInfo):
    from collections import Counter
    import math

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Identify condition for a number n to be b-beautiful
    cot_instruction_1 = "Sub-task 1: Identify the condition for a number n to be b-beautiful: it must have exactly two digits in base b and the sum of these digits must equal the square root of n."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, identifying condition for b-beautiful, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine range of numbers with two digits in base b
    cot_sc_instruction_2 = "Sub-task 2: Determine the range of numbers that have exactly two digits in base b."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, determining range for two digits, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Calculate possible digit pairs for base b
    cot_sc_instruction_3 = "Sub-task 3: Calculate the possible digit pairs (d1, d2) for a given base b and check if their sum equals the square root of the number they represent."
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}

    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, calculating digit pairs, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3

    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 4: Count b-beautiful numbers for each base b
    cot_reflect_instruction_4 = "Sub-task 4: For each base b starting from 2, count the number of b-beautiful numbers by iterating through the range determined in stage 1 and checking the condition from subtask 3."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, counting b-beautiful numbers, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the count of b-beautiful numbers and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining count of b-beautiful numbers, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Validate the count of b-beautiful numbers
    cot_reflect_instruction_5 = "Sub-task 5: Validate the count of b-beautiful numbers for each base b to ensure accuracy."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, validating count, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review the validation of b-beautiful numbers count and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining validation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Stage 3: Final Output Generation and Integration
    # Sub-task 6: Identify the smallest base b for which there are more than ten b-beautiful numbers
    debate_instruction_6 = "Sub-task 6: Identify the smallest base b for which there are more than ten b-beautiful numbers."
    debate_agents_6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round

    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for agent in debate_agents_6:
            thinking6, answer6 = agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, identifying smallest base b, thinking: {thinking6.content}; answer: {answer6.content}')
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_answers_6 = [ans.content for ans in all_answer6[-1]]
    final_answercontent_6 = Counter(final_answers_6).most_common(1)[0][0]
    index_6 = final_answers_6.index(final_answercontent_6)
    thinking6 = all_thinking6[-1][index_6]
    answer6 = all_answer6[-1][index_6]
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer