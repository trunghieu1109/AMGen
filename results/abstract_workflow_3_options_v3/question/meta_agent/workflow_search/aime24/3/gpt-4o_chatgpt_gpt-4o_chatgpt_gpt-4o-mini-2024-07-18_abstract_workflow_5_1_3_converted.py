async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Stage 1: Core Information Extraction
    
    # Sub-task 1: Identify and define the function f(x) = ||x| - 1/2|.
    cot_instruction_1 = "Sub-task 1: Identify and define the function f(x) = ||x| - 1/2|."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, define f(x), thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Identify and define the function g(x) = ||x| - 1/4|.
    cot_instruction_2 = "Sub-task 2: Identify and define the function g(x) = ||x| - 1/4|."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, define g(x), thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Formulate the equation y = 4g(f(sin(2πx))) and understand its components.
    cot_instruction_3 = "Sub-task 3: Formulate the equation y = 4g(f(sin(2πx))) and understand its components."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, formulate y = 4g(f(sin(2πx))), thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Formulate the equation x = 4g(f(cos(3πy))) and understand its components.
    cot_instruction_4 = "Sub-task 4: Formulate the equation x = 4g(f(cos(3πy))) and understand its components."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, formulate x = 4g(f(cos(3πy))), thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Stage 2: Intermediate Value Computation
    
    # Sub-task 5: Analyze the behavior of f(x) and g(x) to determine their range and periodicity.
    cot_instruction_5 = "Sub-task 5: Analyze the behavior of f(x) and g(x) to determine their range and periodicity."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, analyze behavior of f(x) and g(x), thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Sub-task 6: Compute the intermediate values of 4g(f(sin(2πx))) for various x to understand its graph.
    cot_reflect_instruction_6 = "Sub-task 6: Compute the intermediate values of 4g(f(sin(2πx))) for various x to understand its graph."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking3, answer3, thinking5, answer5]
    
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, compute 4g(f(sin(2πx))), thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "please review the computation of 4g(f(sin(2πx))) and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining computation of 4g(f(sin(2πx))), thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Sub-task 7: Compute the intermediate values of 4g(f(cos(3πy))) for various y to understand its graph.
    cot_reflect_instruction_7 = "Sub-task 7: Compute the intermediate values of 4g(f(cos(3πy))) for various y to understand its graph."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking4, answer4, thinking5, answer5]
    
    thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_7.id}, compute 4g(f(cos(3πy))), thinking: {thinking7.content}; answer: {answer7.content}')

    for i in range(N_max_7):
        feedback, correct = critic_agent_7([taskInfo, thinking7, answer7], "please review the computation of 4g(f(cos(3πy))) and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_7.id}, refining computation of 4g(f(cos(3πy))), thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    # Stage 3: Final Output Derivation
    
    # Sub-task 8: Determine the points of intersection between the graphs of y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy))).
    debate_instruction_8 = "Sub-task 8: Determine the points of intersection between the graphs of y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))."
    debate_agents_8 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            input_infos_8 = [taskInfo, thinking6, answer6, thinking7, answer7]
            if r > 0:
                input_infos_8.extend(all_thinking8[r-1])
            thinking8, answer8 = agent(input_infos_8, debate_instruction_8, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, determining intersection points, thinking: {thinking8.content}; answer: {answer8.content}')
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    
    final_decision_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on intersection points.", is_sub_task=True)
    agents.append(f'Final Decision agent, determining intersection points, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')

    # Sub-task 9: Count the number of intersection points to find the final answer.
    cot_instruction_9 = "Sub-task 9: Count the number of intersection points to find the final answer."
    cot_agent_9 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking9, answer9 = cot_agent_9([taskInfo, thinking8, answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_9.id}, count intersection points, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer