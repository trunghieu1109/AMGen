async def forward(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []

    # Stage 1: Scenario Exploration and Enumeration
    # Sub-task 1: Enumerate all possible ways to fill the 2x3 grid with digits from 0 to 9.
    cot_instruction_1 = "Sub-task 1: Enumerate all possible ways to fill the 2x3 grid with digits from 0 to 9."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, enumerating grid configurations, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Calculate the sum of the two numbers formed by reading left to right for each grid configuration.
    cot_instruction_2 = "Sub-task 2: Calculate the sum of the two numbers formed by reading left to right for each grid configuration."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, calculating left-to-right sums, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Calculate the sum of the three numbers formed by reading top to bottom for each grid configuration.
    cot_instruction_3 = "Sub-task 3: Calculate the sum of the three numbers formed by reading top to bottom for each grid configuration."
    cot_agent_3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_3.id}, calculating top-to-bottom sums, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Stage 2: Case Aggregation and Filtering
    # Sub-task 4: Filter the grid configurations where the sum of the two numbers formed by reading left to right equals 999.
    cot_reflect_instruction_4 = "Sub-task 4: Filter the grid configurations where the sum of the two numbers formed by reading left to right equals 999."
    cot_agent_4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2]
    thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_4.id}, filtering left-to-right valid scenarios, thinking: {thinking4.content}; answer: {answer4.content}')

    for i in range(N_max_4):
        feedback, correct = critic_agent_4([taskInfo, thinking4, answer4], "please review the left-to-right filtering and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_4.id}, refining left-to-right valid scenarios, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Filter the grid configurations where the sum of the three numbers formed by reading top to bottom equals 99.
    cot_reflect_instruction_5 = "Sub-task 5: Filter the grid configurations where the sum of the three numbers formed by reading top to bottom equals 99."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking3, answer3]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, filtering top-to-bottom valid scenarios, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback, correct = critic_agent_5([taskInfo, thinking5, answer5], "please review the top-to-bottom filtering and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining top-to-bottom valid scenarios, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Stage 3: Intermediate Output Calculation
    # Sub-task 6: Identify the grid configurations that satisfy both conditions from stage 2.
    cot_reflect_instruction_6 = "Sub-task 6: Identify the grid configurations that satisfy both conditions from stage 2."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, identifying valid configurations, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback, correct = critic_agent_6([taskInfo, thinking6, answer6], "please review the identification of valid configurations and provide its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining valid configurations, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Stage 4: Final Answer Derivation
    # Sub-task 7: Count the number of valid grid configurations identified in stage 3.
    debate_instruction_7 = "Sub-task 7: Count the number of valid grid configurations identified in stage 3."
    debate_agents_7 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = agent([taskInfo, thinking6, answer6], debate_instruction_7, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = agent(input_infos_7, debate_instruction_7, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, counting valid configurations, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)

    final_decision_agent_7 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = final_decision_agent_7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the number of valid configurations.", is_sub_task=True)
    agents.append(f'Final Decision agent, counting valid configurations, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer