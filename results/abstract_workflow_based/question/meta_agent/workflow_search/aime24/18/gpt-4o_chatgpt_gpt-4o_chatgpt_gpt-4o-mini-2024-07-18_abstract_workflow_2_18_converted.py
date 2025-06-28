async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Determine the equation of line AB and find the parametric representation of any point C on AB.
    cot_instruction_1 = "Sub-task 1: Determine the equation of line AB and find the parametric representation of any point C on AB."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determining line AB and parametric representation, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify the conditions for a segment PQ of unit length with P on the x-axis and Q on the y-axis.
    cot_sc_instruction_2 = "Sub-task 2: Based on the output from Sub-task 1, identify the conditions for a segment PQ of unit length with P on the x-axis and Q on the y-axis."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_2[i].id}, identifying conditions for segment PQ, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 3: Calculate the coordinates of point C such that it does not belong to any segment from the family F, except for AB itself.
    cot_reflect_instruction_3 = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate the coordinates of point C such that it does not belong to any segment from the family F, except for AB itself."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_3.id}, calculating coordinates of point C, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max_3):
        feedback, correct = critic_agent_3([taskInfo, thinking3, answer3], "please review the coordinates of point C calculation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_3.id}, refining coordinates of point C, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Validate that the point C found in subtask 3 is unique and satisfies the given conditions.
    cot_reflect_instruction_4 = "Sub-task 4: Validate that the point C found in subtask 3 is unique and satisfies the given conditions."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round

    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, validating uniqueness of point C, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the uniqueness validation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining uniqueness validation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Final Output Generation and Integration
    # Sub-task 5: Compute OC^2 and express it in the form p/q, where p and q are coprime integers.
    debate_instruction_5 = "Sub-task 5: Compute OC^2 and express it in the form p/q, where p and q are coprime integers."
    debate_agents_5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round

    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for agent in debate_agents_5:
            thinking5, answer5 = agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, computing OC^2, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_answers_5 = [ans.content for ans in all_answer5[-1]]
    final_answercontent_5 = Counter(final_answers_5).most_common(1)[0][0]
    index_5 = final_answers_5.index(final_answercontent_5)
    thinking5 = all_thinking5[-1][index_5]
    answer5 = all_answer5[-1][index_5]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Calculate p + q based on the result from subtask 5.
    cot_instruction_6 = "Sub-task 6: Calculate p + q based on the result from subtask 5."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, calculating p + q, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer