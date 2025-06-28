async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Core Information and Constraint Extraction
    # Sub-task 1: Identify the definitions and properties of the functions f(x) and g(x).
    cot_instruction_1 = "Sub-task 1: Identify the definitions and properties of the functions f(x) = ||x| - 1/2| and g(x) = ||x| - 1/4|."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, determine definitions and properties, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract the equations y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy))).
    cot_instruction_2 = "Sub-task 2: Extract the equations y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, extracting equations, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Identify the conditions for intersection of the graphs of the two equations.
    cot_instruction_3 = "Sub-task 3: Identify the conditions for intersection of the graphs of the equations y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy)))."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, identifying intersection conditions, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Parametric Representation Derivation
    # Sub-task 4: Derive the parametric representation of f(x) = ||x| - 1/2|.
    cot_instruction_4 = "Sub-task 4: Derive the parametric representation of f(x) = ||x| - 1/2|."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_4.id}, deriving parametric representation of f(x), thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Derive the parametric representation of g(x) = ||x| - 1/4|.
    cot_instruction_5 = "Sub-task 5: Derive the parametric representation of g(x) = ||x| - 1/4|."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_5.id}, deriving parametric representation of g(x), thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Derive the parametric representation of y = 4g(f(sin(2πx))).
    cot_instruction_6 = "Sub-task 6: Derive the parametric representation of y = 4g(f(sin(2πx)))."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_6.id}, deriving parametric representation of y, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Derive the parametric representation of x = 4g(f(cos(3πy))).
    cot_instruction_7 = "Sub-task 7: Derive the parametric representation of x = 4g(f(cos(3πy)))."
    cot_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = cot_agent_7([taskInfo, thinking4, answer4, thinking5, answer5], cot_instruction_7, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_7.id}, deriving parametric representation of x, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    print("Step 7: ", sub_tasks[-1])

    # Stage 3: Expression Transformation and Alignment
    # Sub-task 8: Transform the expression y = 4g(f(sin(2πx))) to a clearer form.
    cot_reflect_instruction_8 = "Sub-task 8: Transform the expression y = 4g(f(sin(2πx))) to a clearer form."
    cot_agent_8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_8 = [taskInfo, thinking6, answer6]
    thinking8, answer8 = cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_8.id}, transforming expression y, thinking: {thinking8.content}; answer: {answer8.content}')

    for i in range(N_max):
        feedback, correct = critic_agent_8([taskInfo, thinking8, answer8], "please review the transformation of y and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_8.extend([thinking8, answer8, feedback])
        thinking8, answer8 = cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_8.id}, refining transformation of y, thinking: {thinking8.content}; answer: {answer8.content}')
    sub_tasks.append(f'Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}')
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Transform the expression x = 4g(f(cos(3πy))) to a clearer form.
    cot_reflect_instruction_9 = "Sub-task 9: Transform the expression x = 4g(f(cos(3πy))) to a clearer form."
    cot_agent_9 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_9 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs_9 = [taskInfo, thinking7, answer7]
    thinking9, answer9 = cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_9.id}, transforming expression x, thinking: {thinking9.content}; answer: {answer9.content}')

    for i in range(N_max):
        feedback, correct = critic_agent_9([taskInfo, thinking9, answer9], "please review the transformation of x and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_9.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_9.extend([thinking9, answer9, feedback])
        thinking9, answer9 = cot_agent_9(cot_inputs_9, cot_reflect_instruction_9, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_9.id}, refining transformation of x, thinking: {thinking9.content}; answer: {answer9.content}')
    sub_tasks.append(f'Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}')
    print("Step 9: ", sub_tasks[-1])

    # Stage 4: Final Output Calculation
    # Sub-task 10: Calculate the number of intersections of the graphs of the transformed expressions.
    debate_instruction_10 = "Sub-task 10: Calculate the number of intersections of the graphs of the transformed expressions."
    debate_agents_10 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.4) for role in self.debate_role]
    N_max_10 = self.max_round
    all_thinking10 = [[] for _ in range(N_max_10)]
    all_answer10 = [[] for _ in range(N_max_10)]

    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            input_infos_10 = [taskInfo, thinking8, answer8, thinking9, answer9]
            if r > 0:
                input_infos_10.extend(all_thinking10[r-1])
            thinking10, answer10 = agent(input_infos_10, debate_instruction_10, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating intersections, thinking: {thinking10.content}; answer: {answer10.content}')
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)

    final_decision_agent_10 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking10, answer10 = final_decision_agent_10([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make a final decision on the number of intersections.", is_sub_task=True)
    agents.append(f'Final Decision agent on calculating intersections, thinking: {thinking10.content}; answer: {answer10.content}')
    sub_tasks.append(f'Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}')
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer