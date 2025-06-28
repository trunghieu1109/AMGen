async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Parametric Representation Identification
    # Sub-task 1: Identify parametric representation of z with |z| = 4
    cot_instruction = "Sub-task 1: Identify the parametric representation of the complex number z with |z| = 4."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying parametric representation of z, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify parametric representation of the expression
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, identify the parametric representation of the expression (75+117i)z + 9/6 + 144iz."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining parametric representation of the expression, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Expression Analysis and Transformation
    # Sub-task 3: Transform expression to isolate the real part
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, transform the expression to isolate the real part."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, transforming expression, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the expression transformation and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining transformation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Determine conditions for maximizing the real part
    debate_instruction_4 = "Sub-task 4: Determine the conditions under which the real part is maximized."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
            thinking4, answer4 = agent(input_infos_4, debate_instruction_4, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, determining conditions, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make a final decision on the conditions.", is_sub_task=True)
    agents.append(f'Final Decision agent on conditions, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Stage 3: Final Output Calculation
    # Sub-task 5: Calculate the maximum possible value of the real part
    cot_reflect_instruction_5 = "Sub-task 5: Calculate the maximum possible value of the real part based on the conditions identified."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking4, answer4]
    thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating maximum real part, thinking: {thinking5.content}; answer: {answer5.content}')
    for i in range(N_max_5):
        feedback, correct = critic_agent([taskInfo, thinking5, answer5], "please review the calculation of the maximum real part and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback on calculation, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent(cot_inputs, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining calculation, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Stage 4: Final Output Verification
    # Sub-task 6: Verify the calculated maximum real part
    cot_reflect_instruction_6 = "Sub-task 6: Verify the calculated maximum real part to ensure it meets all query conditions."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, verifying maximum real part, thinking: {thinking6.content}; answer: {answer6.content}')
    for i in range(N_max_6):
        feedback, correct = critic_agent([taskInfo, thinking6, answer6], "please review the verification of the maximum real part and correct if needed.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback on verification, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent(cot_inputs, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining verification, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer