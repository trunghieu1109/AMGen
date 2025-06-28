async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Sub-task 1: Analyze the function f(x) = ||x| - 1/2|.
    cot_instruction_1 = "Sub-task 1: Analyze the function f(x) = ||x| - 1/2| to understand its behavior and critical points."
    cot_agent_1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_1.id}, analyzing f(x), thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Analyze the function g(x) = ||x| - 1/4|.
    cot_instruction_2 = "Sub-task 2: Analyze the function g(x) = ||x| - 1/4| to understand its behavior and critical points."
    cot_agent_2 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking2, answer2 = cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent_2.id}, analyzing g(x), thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Determine the composition of functions in y = 4g(f(sin(2πx))) and analyze its behavior.
    cot_sc_instruction_3 = "Sub-task 3: Determine the composition of functions in y = 4g(f(sin(2πx))) and analyze its behavior."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}

    for i in range(N):
        thinking3, answer3 = cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_3[i].id}, analyzing y composition, thinking: {thinking3.content}; answer: {answer3.content}')
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3

    answer3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3]
    answer3 = answermapping_3[answer3]
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Determine the composition of functions in x = 4g(f(cos(3πy))) and analyze its behavior.
    cot_sc_instruction_4 = "Sub-task 4: Determine the composition of functions in x = 4g(f(cos(3πy))) and analyze its behavior."
    cot_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.7) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}

    for i in range(N):
        thinking4, answer4 = cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents_4[i].id}, analyzing x composition, thinking: {thinking4.content}; answer: {answer4.content}')
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4

    answer4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4]
    answer4 = answermapping_4[answer4]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Identify the conditions under which y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy))) intersect.
    cot_reflect_instruction_5 = "Sub-task 5: Identify the conditions under which y = 4g(f(sin(2πx))) and x = 4g(f(cos(3πy))) intersect."
    cot_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking3, answer3, thinking4, answer4]
    thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_5.id}, identifying intersection conditions, thinking: {thinking5.content}; answer: {answer5.content}')

    for i in range(N_max_5):
        feedback5, correct5 = critic_agent_5([taskInfo, thinking5, answer5], "please review the intersection conditions and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback5.content}; answer: {correct5.content}')
        if correct5.content == 'True':
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_5.id}, refining intersection conditions, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    # Sub-task 6: Validate the identified intersection conditions by checking the periodicity and symmetry of the functions.
    cot_reflect_instruction_6 = "Sub-task 6: Validate the identified intersection conditions by checking the periodicity and symmetry of the functions."
    cot_agent_6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round

    cot_inputs_6 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent_6.id}, validating intersection conditions, thinking: {thinking6.content}; answer: {answer6.content}')

    for i in range(N_max_6):
        feedback6, correct6 = critic_agent_6([taskInfo, thinking6, answer6], "please review the validation of intersection conditions and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}')
        if correct6.content == 'True':
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent_6.id}, refining validation, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')

    # Sub-task 7: Calculate the number of intersection points based on the validated conditions.
    debate_instruction_7 = "Sub-task 7: Calculate the number of intersection points based on the validated conditions."
    debate_agents_7 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    N_max_7 = self.max_round

    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]

    for r in range(N_max_7):
        for agent in debate_agents_7:
            input_infos_7 = [taskInfo, thinking6, answer6]
            if r > 0:
                input_infos_7.extend(all_thinking7[r-1])
            thinking7, answer7 = agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, calculating intersection points, thinking: {thinking7.content}; answer: {answer7.content}')
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